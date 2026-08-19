from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import stage5d_attribution as s

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_stage5d"
MODELS = ROOT / "trained_v3"
POSITIONS = ["QB", "RB", "WR", "TE"]


def corrected_family(feature: str) -> str:
    f = feature.lower()
    if f.startswith("landing_"):
        return "NFL landing spot"
    if feature in {"draft_age", "young_for_position"} or "years_before" in f or f.endswith("_seasons"):
        return "Age / experience"
    if feature in s.CAPITAL or f in {"draft_round", "draft_pick", "log_pick", "pick_inv_sqrt", "day1", "day2"}:
        return "Draft capital"
    if any(k in f for k in s.ATHLETIC_KEYS):
        return "Athletic profile"
    if f.startswith("scout_"):
        return "Scouting surprise"
    if f.startswith("pass_team_") or f.startswith("rush_team_") or f.startswith("rec_team_") or f.endswith("power_conf"):
        return "College team context"
    if f.startswith("qb_"):
        return "Rushing production"
    if f.startswith("pass_"):
        return "Passing production"
    if f.startswith("rush_") or f.startswith("rb_"):
        return "Rushing production"
    if f.startswith("rec_"):
        return "Receiving production"
    if "team_" in f:
        return "College team context"
    return "Other production"


def main():
    # Override only the explanation taxonomy. The exact frozen scoring functions/models are unchanged.
    s.family = corrected_family
    s.main()

    family_rows = []
    scout_rows = []
    for pos in POSITIONS:
        job = joblib.load(MODELS / f"{pos}.joblib")
        primary = list(job.get("features", []))
        secondary = list(job.get("features_b") or [])
        union = s.ordered_union(primary, secondary)
        classifier = set(primary)
        families = sorted({corrected_family(f) for f in union})
        for fam in families:
            fs = [f for f in union if corrected_family(f) == fam]
            family_rows.append({
                "position": pos,
                "feature_family": fam,
                "union_feature_count": len(fs),
                "primary_regression_count": sum(f in primary for f in fs),
                "secondary_regression_count": sum(f in secondary for f in fs),
                "hit_classifier_count": sum(f in classifier for f in fs),
            })
        for f in union:
            if f.startswith("scout_"):
                scout_rows.append({
                    "position": pos,
                    "feature": f,
                    "in_primary_regression": int(f in primary),
                    "in_secondary_regression": int(f in secondary),
                    "in_hit_classifier": int(f in classifier),
                    "blend_weight_primary": float(job.get("blend", {}).get("weight", 1.0)),
                })

    family_audit = pd.DataFrame(family_rows)
    scout_audit = pd.DataFrame(scout_rows, columns=[
        "position", "feature", "in_primary_regression", "in_secondary_regression",
        "in_hit_classifier", "blend_weight_primary"
    ])
    family_audit.to_csv(OUT / "selected_feature_family_audit.csv", index=False)
    scout_audit.to_csv(OUT / "selected_scout_feature_audit.csv", index=False)

    report_path = OUT / "REPORT.md"
    with report_path.open("a") as f:
        f.write("\n## Selected feature-family audit\n\n")
        f.write("This audit uses the corrected football taxonomy: draft age is age/experience; QB-derived rushing features are rushing; receiving-per-team-pass metrics remain receiving production; explicit pass/rush/rec team-context fields remain college context.\n\n")
        f.write("|Pos|Family|Union N|Primary reg|Secondary reg|Hit clf|\n|---|---|---:|---:|---:|---:|\n")
        for _, r in family_audit.iterrows():
            f.write(f"|{r.position}|{r.feature_family}|{int(r.union_feature_count)}|{int(r.primary_regression_count)}|{int(r.secondary_regression_count)}|{int(r.hit_classifier_count)}|\n")
        f.write("\n## Scouting-surprise membership audit\n\n")
        if scout_audit.empty:
            f.write("No selected frozen model contains explicit scouting-surprise inputs.\n")
        else:
            f.write("The rows below are exact membership in the frozen jobs. Their presence is diagnostic, not a Stage 5D model change.\n\n")
            f.write("|Pos|Feature|Primary reg?|Secondary reg?|Hit clf?|Primary blend weight|\n|---|---|---:|---:|---:|---:|\n")
            for _, r in scout_audit.iterrows():
                f.write(f"|{r.position}|{r.feature}|{int(r.in_primary_regression)}|{int(r.in_secondary_regression)}|{int(r.in_hit_classifier)}|{r.blend_weight_primary:.2f}|\n")

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["taxonomy_corrected"] = True
    manifest["taxonomy_note"] = "Family labels corrected without changing feature values, frozen models, predictions, probabilities, or per-feature perturbation deltas. Family-neutralization outputs were regenerated under the corrected taxonomy."
    manifest["outputs"] = list(dict.fromkeys(manifest.get("outputs", []) + [
        "selected_feature_family_audit.csv", "selected_scout_feature_audit.csv"
    ]))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "stage": "5D-finalized",
        "taxonomy_corrected": True,
        "scout_feature_rows": len(scout_audit),
        "family_audit_rows": len(family_audit),
    }, indent=2))


if __name__ == "__main__":
    main()
