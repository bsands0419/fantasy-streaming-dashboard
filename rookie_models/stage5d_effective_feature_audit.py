from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import stage5d_finalize as fin
import stage5d_attribution as s

ROOT = Path(__file__).resolve().parent
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CUR_POOL = ROOT / "results_2026" / "prospect_pool_2026.csv"
MODELS = ROOT / "trained_v3"
OUT = ROOT / "results_stage5d"
POSITIONS = ["QB", "RB", "WR", "TE"]


def main():
    hist = pd.read_csv(HIST_POOL, low_memory=False)
    cur = pd.read_csv(CUR_POOL, low_memory=False)
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    cur["season"] = pd.to_numeric(cur["season"], errors="coerce")
    if "target_valid" in hist.columns:
        hist = hist[pd.to_numeric(hist.target_valid, errors="coerce").eq(1)].copy()
    hist = hist[hist.season.le(2023)].copy()
    cur = cur[cur.season.eq(2026)].copy()

    rows = []
    scouts = []
    for pos in POSITIONS:
        job = joblib.load(MODELS / f"{pos}.joblib")
        primary = list(job.get("features", []))
        secondary = list(job.get("features_b") or [])
        union = s.ordered_union(primary, secondary)
        h = hist[hist.position.eq(pos)]
        c = cur[cur.position.eq(pos)]

        info = []
        for feature in union:
            hv = pd.to_numeric(h[feature], errors="coerce") if feature in h.columns else pd.Series(dtype=float)
            cv = pd.to_numeric(c[feature], errors="coerce") if feature in c.columns else pd.Series(dtype=float)
            hn = int(hv.notna().sum())
            cn = int(cv.notna().sum())
            sd = float(hv.std()) if hn >= 2 else np.nan
            historically_observed = int(hn > 0)
            similarity_usable = int(hn >= 20 and np.isfinite(sd) and sd > 1e-12 and feature in c.columns)
            rec = {
                "position": pos,
                "feature": feature,
                "feature_family": fin.corrected_family(feature),
                "in_primary_regression": int(feature in primary),
                "in_secondary_regression": int(feature in secondary),
                "in_hit_classifier": int(feature in primary),
                "historical_nonmissing_n": hn,
                "current_2026_nonmissing_n": cn,
                "historically_observed": historically_observed,
                "similarity_usable_stage5b_rule": similarity_usable,
            }
            info.append(rec)
            if feature.startswith("scout_"):
                scouts.append(rec | {"blend_weight_primary": float(job.get("blend", {}).get("weight", 1.0))})

        z = pd.DataFrame(info)
        for fam, g in z.groupby("feature_family", sort=True):
            rows.append({
                "position": pos,
                "feature_family": fam,
                "stored_union_feature_count": len(g),
                "historically_observed_feature_count": int(g.historically_observed.sum()),
                "stage5b_similarity_usable_feature_count": int(g.similarity_usable_stage5b_rule.sum()),
                "stored_primary_regression_count": int(g.in_primary_regression.sum()),
                "observed_primary_regression_count": int((g.in_primary_regression * g.historically_observed).sum()),
                "stored_secondary_regression_count": int(g.in_secondary_regression.sum()),
                "observed_secondary_regression_count": int((g.in_secondary_regression * g.historically_observed).sum()),
                "stored_hit_classifier_count": int(g.in_hit_classifier.sum()),
                "observed_hit_classifier_count": int((g.in_hit_classifier * g.historically_observed).sum()),
            })

    audit = pd.DataFrame(rows)
    scout = pd.DataFrame(scouts)
    audit.to_csv(OUT / "selected_feature_family_effective_audit.csv", index=False)
    scout.to_csv(OUT / "selected_scout_feature_effective_audit.csv", index=False)

    report = OUT / "REPORT.md"
    with report.open("a") as f:
        f.write("\n## Effective frozen-feature audit\n\n")
        f.write("Frozen job feature lists contain some position-irrelevant columns that were entirely missing for that position and were skipped by scikit-learn's median imputer. The table below separates stored names from historically observed inputs. `Stage5B usable` additionally requires at least 20 historical observations and non-zero variance, matching the similarity-space rule.\n\n")
        f.write("|Pos|Family|Stored|Observed|Stage5B usable|Primary observed|Secondary observed|Hit-clf observed|\n|---|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in audit.iterrows():
            f.write(f"|{r.position}|{r.feature_family}|{int(r.stored_union_feature_count)}|{int(r.historically_observed_feature_count)}|{int(r.stage5b_similarity_usable_feature_count)}|{int(r.observed_primary_regression_count)}|{int(r.observed_secondary_regression_count)}|{int(r.observed_hit_classifier_count)}|\n")
        f.write("\n### Effective scouting-surprise coverage\n\n")
        if scout.empty:
            f.write("No scouting-surprise fields are present in selected jobs.\n")
        else:
            f.write("|Pos|Feature|Primary?|Secondary?|Hit clf?|Historical N|2026 N|Primary blend wt|\n|---|---|---:|---:|---:|---:|---:|---:|\n")
            for _, r in scout.iterrows():
                f.write(f"|{r.position}|{r.feature}|{int(r.in_primary_regression)}|{int(r.in_secondary_regression)}|{int(r.in_hit_classifier)}|{int(r.historical_nonmissing_n)}|{int(r.current_2026_nonmissing_n)}|{r.blend_weight_primary:.2f}|\n")

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["effective_feature_audit_generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["effective_feature_note"] = "Stored frozen feature names are distinguished from historically observed/effective inputs because SimpleImputer skips all-missing columns."
    manifest["outputs"] = list(dict.fromkeys(manifest.get("outputs", []) + [
        "selected_feature_family_effective_audit.csv",
        "selected_scout_feature_effective_audit.csv",
    ]))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "stage": "5D-effective-feature-audit",
        "family_rows": len(audit),
        "scout_rows": len(scout),
    }, indent=2))


if __name__ == "__main__":
    main()
