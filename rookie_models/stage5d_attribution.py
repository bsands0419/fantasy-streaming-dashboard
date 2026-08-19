from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CUR_POOL = ROOT / "results_2026" / "prospect_pool_2026.csv"
RANK = ROOT / "results_stage5c" / "rookie_rankings_2026_stage5c.csv"
MODELS = ROOT / "trained_v3"
OUT = ROOT / "results_stage5d"
OUT.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
PRED_TOL = 1e-6
PROB_TOL = 1e-6
TOP_N = 5

CAPITAL = {"draft_round", "draft_pick", "log_pick", "pick_inv_sqrt", "day1", "day2"}
ATHLETIC_KEYS = ["forty", "vertical", "bench", "broad", "cone", "shuttle", "speed_score", "height_inches", "weight_lbs", "bmi", "burst_proxy"]


def ordered_union(a, b):
    out = []
    for x in list(a or []) + list(b or []):
        if x not in out:
            out.append(x)
    return out


def n(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def family(feature: str) -> str:
    f = feature.lower()
    if feature in CAPITAL or f.startswith("draft_") or f in {"log_pick", "pick_inv_sqrt", "day1", "day2"}:
        return "Draft capital"
    if f.startswith("landing_"):
        return "NFL landing spot"
    if any(k in f for k in ATHLETIC_KEYS):
        return "Athletic profile"
    if feature in {"draft_age", "young_for_position"} or "years_before" in f or f.endswith("_seasons"):
        return "Age / experience"
    if "team_" in f or f.endswith("power_conf"):
        return "College team context"
    if f.startswith("pass_") or f.startswith("qb_"):
        return "Passing production"
    if f.startswith("rush_") or f.startswith("rb_"):
        return "Rushing production"
    if f.startswith("rec_"):
        return "Receiving production"
    if f.startswith("scout_"):
        return "Scouting surprise"
    return "Other production"


def pretty_feature(feature: str) -> str:
    s = feature
    replacements = [
        ("_expadj_peak", " experience-adjusted peak"),
        ("_early_peak", " early-career peak"),
        ("_years_before_first", " years before first season"),
        ("_years_before_last", " years before final season"),
        ("_seasons", " college seasons"),
        ("_final", " final season"),
        ("_peak", " career peak"),
        ("_mean", " career mean"),
        ("_slope", " career trend"),
    ]
    suffix = ""
    for raw, lab in replacements:
        if s.endswith(raw):
            s = s[: -len(raw)]
            suffix = lab
            break
    words = {
        "draft_round": "draft round",
        "draft_pick": "draft pick",
        "log_pick": "log draft pick",
        "pick_inv_sqrt": "inverse-root draft pick",
        "day1": "first-round indicator",
        "day2": "day-two indicator",
        "draft_age": "draft age",
        "young_for_position": "youth for position",
        "yardsdropback": "yards per dropback",
        "epaplay": "EPA per play",
        "epagame": "EPA per game",
        "yardsgame": "yards per game",
        "yardsplay": "yards per play",
        "comppct": "completion percentage",
        "sack_rate": "sack rate",
        "int_rate": "interception rate",
        "pass_td_rate": "passing TD rate",
        "qb_rush_fantasy_pg": "QB rushing fantasy points per game",
        "rush_share": "rushing share",
        "rush_yard_share": "rushing-yard share",
        "target_share": "target share",
        "rec_yard_share": "receiving-yard share",
        "rec_td_share": "receiving-TD share",
        "reception_share": "reception share",
        "yards_per_target": "yards per target",
        "targets_per_game": "targets per game",
        "rec_per_game": "receptions per game",
        "rec_per_team_pass_att": "receiving production per team pass attempt",
        "receptions_per_team_pass_att": "receptions per team pass attempt",
        "scrim_yards_per_team_play": "scrimmage yards per team play",
        "scrim_ypg": "scrimmage yards per game",
        "rec_yards_per_team_pass": "receiving yards per team pass attempt",
        "speed_score_v2": "speed score",
        "burst_proxy": "burst proxy",
    }
    base = s
    for prefix in ["pass_", "rush_", "rec_", "rb_", "qb_", "landing_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    base = words.get(base, base.replace("_", " "))
    if feature.startswith("landing_"):
        base = "landing spot: " + base
    return (base + suffix).strip()


def add_missing_columns(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in features:
        if c not in out.columns:
            out[c] = np.nan
    return out


def score_row(row: pd.DataFrame, job) -> float:
    a, bb = job["fitted"]["primary_ppg"]
    pa = float(a.predict(row[job["features"]])[0])
    if bb is None:
        return pa
    pb = float(bb.predict(row[job["features_b"]])[0])
    w = float(job["blend"]["weight"])
    return w * pa + (1.0 - w) * pb


def hit_row(row: pd.DataFrame, job) -> float:
    return float(job["classifier"].predict_proba(row[job["features"]])[:, 1][0])


def percentile(hist_values: pd.Series, value: float) -> float:
    h = pd.to_numeric(hist_values, errors="coerce").dropna().to_numpy(float)
    if not len(h) or not np.isfinite(value):
        return np.nan
    return 100.0 * float(np.mean(h <= value))


def fmt_driver(r, metric: str) -> str:
    if metric == "score":
        return f"{r.pretty_feature} ({r.score_delta_ppg:+.2f} PPG; {r.feature_family})"
    return f"{r.pretty_feature} ({r.hit_delta_pp:+.1f} pp; {r.feature_family})"


def top_string(d: pd.DataFrame, col: str, positive: bool, metric: str, k: int = TOP_N) -> str:
    z = d[np.isfinite(pd.to_numeric(d[col], errors="coerce"))].copy()
    z = z[z[col] > 0] if positive else z[z[col] < 0]
    if z.empty:
        return ""
    z = z.sort_values(col, ascending=not positive).head(k)
    return "; ".join(fmt_driver(r, metric) for _, r in z.iterrows())


def family_string(d: pd.DataFrame, col: str, positive: bool, k: int = 3) -> str:
    z = d[np.isfinite(pd.to_numeric(d[col], errors="coerce"))].copy()
    z = z[z[col] > 0] if positive else z[z[col] < 0]
    if z.empty:
        return ""
    z = z.sort_values(col, ascending=not positive).head(k)
    unit = "PPG" if col == "score_delta_ppg" else "pp"
    return "; ".join(f"{r.feature_family} ({r[col]:+.2f} {unit})" for _, r in z.iterrows())


def main():
    hist = pd.read_csv(HIST_POOL, low_memory=False)
    cur = pd.read_csv(CUR_POOL, low_memory=False)
    rankings = pd.read_csv(RANK, low_memory=False)

    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    cur["season"] = pd.to_numeric(cur["season"], errors="coerce")
    if "target_valid" in hist.columns:
        hist = hist[pd.to_numeric(hist.target_valid, errors="coerce").eq(1)].copy()
    hist = hist[hist.season.le(2023)].copy()
    cur = cur[cur.season.eq(2026)].copy()

    feat_rows = []
    family_rows = []
    audits = []
    explanation_rows = []

    for pos in POSITIONS:
        job = joblib.load(MODELS / f"{pos}.joblib")
        all_feats = ordered_union(job.get("features", []), job.get("features_b", []))
        hp_feats = list(job.get("features", []))
        h = hist[hist.position.eq(pos)].copy().reset_index(drop=True)
        c = cur[cur.position.eq(pos)].copy().reset_index(drop=True)
        h = add_missing_columns(h, all_feats)
        c = add_missing_columns(c, all_feats)

        medians = {f: n(pd.to_numeric(h[f], errors="coerce").median()) for f in all_feats}
        counts = {f: int(pd.to_numeric(h[f], errors="coerce").notna().sum()) for f in all_feats}
        rank_pos = rankings[rankings.position.eq(pos)].copy()

        pred_errs = []
        prob_errs = []

        for _, player in c.iterrows():
            name = player.pfr_name
            one = player.to_frame().T.copy()
            for f in all_feats:
                one[f] = pd.to_numeric(one[f], errors="coerce")

            base_score = score_row(one, job)
            base_hit = hit_row(one, job)
            rr = rank_pos[rank_pos.pfr_name.eq(name)]
            listed_score = n(rr.iloc[0].pred_best2of3_ppg) if len(rr) else np.nan
            listed_hit = n(rr.iloc[0].hit_probability) if len(rr) else np.nan
            pred_errs.append(abs(base_score - listed_score) if np.isfinite(listed_score) else np.nan)
            prob_errs.append(abs(base_hit - listed_hit) if np.isfinite(listed_hit) else np.nan)

            player_feature_rows = []
            for f in all_feats:
                med = medians[f]
                raw = n(one.iloc[0][f])
                if not np.isfinite(med):
                    continue
                pert = one.copy()
                pert.at[pert.index[0], f] = med
                pert_score = score_row(pert, job)
                pert_hit = hit_row(pert, job) if f in hp_feats else base_hit
                rec = {
                    "position": pos,
                    "pfr_name": name,
                    "feature": f,
                    "pretty_feature": pretty_feature(f),
                    "feature_family": family(f),
                    "feature_in_regression_primary": int(f in job.get("features", [])),
                    "feature_in_regression_secondary": int(f in (job.get("features_b") or [])),
                    "feature_in_hit_classifier": int(f in hp_feats),
                    "raw_value": raw,
                    "historical_median": med,
                    "historical_nonmissing_n": counts[f],
                    "raw_value_percentile": percentile(h[f], raw),
                    "raw_value_missing": int(not np.isfinite(raw)),
                    "base_pred_best2of3_ppg": base_score,
                    "neutralized_pred_best2of3_ppg": pert_score,
                    "score_delta_ppg": base_score - pert_score,
                    "base_hit_probability": base_hit,
                    "neutralized_hit_probability": pert_hit,
                    "hit_delta_pp": 100.0 * (base_hit - pert_hit),
                }
                rec["direction_conflict"] = int(
                    np.sign(rec["score_delta_ppg"]) != 0 and
                    np.sign(rec["hit_delta_pp"]) != 0 and
                    np.sign(rec["score_delta_ppg"]) != np.sign(rec["hit_delta_pp"])
                )
                feat_rows.append(rec)
                player_feature_rows.append(rec)

            pfd = pd.DataFrame(player_feature_rows)

            # Joint family neutralization captures within-family interactions. These deltas are local sensitivities, not additive shares.
            player_family_rows = []
            fams = sorted(set(family(f) for f in all_feats))
            for fam in fams:
                fs = [f for f in all_feats if family(f) == fam and np.isfinite(medians[f])]
                if not fs:
                    continue
                pert = one.copy()
                for f in fs:
                    pert.at[pert.index[0], f] = medians[f]
                pert_score = score_row(pert, job)
                hit_fs = [f for f in fs if f in hp_feats]
                if hit_fs:
                    pert_hit = hit_row(pert, job)
                else:
                    pert_hit = base_hit
                frec = {
                    "position": pos,
                    "pfr_name": name,
                    "feature_family": fam,
                    "family_feature_count": len(fs),
                    "family_hit_feature_count": len(hit_fs),
                    "base_pred_best2of3_ppg": base_score,
                    "neutralized_pred_best2of3_ppg": pert_score,
                    "score_delta_ppg": base_score - pert_score,
                    "base_hit_probability": base_hit,
                    "neutralized_hit_probability": pert_hit,
                    "hit_delta_pp": 100.0 * (base_hit - pert_hit),
                }
                frec["direction_conflict"] = int(
                    np.sign(frec["score_delta_ppg"]) != 0 and
                    np.sign(frec["hit_delta_pp"]) != 0 and
                    np.sign(frec["score_delta_ppg"]) != np.sign(frec["hit_delta_pp"])
                )
                family_rows.append(frec)
                player_family_rows.append(frec)

            pfa = pd.DataFrame(player_family_rows)
            meta = rr.iloc[0].to_dict() if len(rr) else {}
            explanation_rows.append({
                "position": pos,
                "rank": meta.get("rank"),
                "pfr_name": name,
                "prospect_model_percentile": meta.get("prospect_model_percentile"),
                "pred_best2of3_ppg": base_score,
                "hit_probability": base_hit,
                "historical_hit_rate": meta.get("historical_hit_rate"),
                "nearest5_historical_hit_rate": meta.get("nearest5_historical_hit_rate"),
                "model_confidence": meta.get("model_confidence"),
                "out_of_distribution_flag": meta.get("out_of_distribution_flag"),
                "diagnostic_priority": meta.get("diagnostic_priority"),
                "diagnostic_notes": meta.get("diagnostic_notes"),
                "closest_historical_comps": meta.get("closest_historical_comps"),
                "top_score_positive_features": top_string(pfd, "score_delta_ppg", True, "score"),
                "top_score_negative_features": top_string(pfd, "score_delta_ppg", False, "score"),
                "top_hit_positive_features": top_string(pfd, "hit_delta_pp", True, "hit"),
                "top_hit_negative_features": top_string(pfd, "hit_delta_pp", False, "hit"),
                "top_score_positive_families": family_string(pfa, "score_delta_ppg", True),
                "top_score_negative_families": family_string(pfa, "score_delta_ppg", False),
                "top_hit_positive_families": family_string(pfa, "hit_delta_pp", True),
                "top_hit_negative_families": family_string(pfa, "hit_delta_pp", False),
                "opposite_direction_feature_count": int(pfd.direction_conflict.sum()) if len(pfd) else 0,
                "opposite_direction_family_count": int(pfa.direction_conflict.sum()) if len(pfa) else 0,
            })

        max_pred_err = float(np.nanmax(pred_errs)) if np.isfinite(pd.to_numeric(pd.Series(pred_errs), errors="coerce")).any() else np.nan
        max_prob_err = float(np.nanmax(prob_errs)) if np.isfinite(pd.to_numeric(pd.Series(prob_errs), errors="coerce")).any() else np.nan
        audits.append({
            "position": pos,
            "current_rows": len(c),
            "historical_rows": len(h),
            "regression_primary_features": len(job.get("features", [])),
            "regression_secondary_features": len(job.get("features_b") or []),
            "union_features": len(all_feats),
            "hit_classifier_features": len(hp_feats),
            "max_abs_regression_reproduction_error": max_pred_err,
            "max_abs_hit_probability_reproduction_error": max_prob_err,
            "regression_reproduced": int(np.isfinite(max_pred_err) and max_pred_err <= PRED_TOL),
            "hit_probability_reproduced": int(np.isfinite(max_prob_err) and max_prob_err <= PROB_TOL),
        })

    features = pd.DataFrame(feat_rows)
    families = pd.DataFrame(family_rows)
    audit = pd.DataFrame(audits)
    explanations = pd.DataFrame(explanation_rows)

    if not audit.regression_reproduced.eq(1).all() or not audit.hit_probability_reproduced.eq(1).all():
        raise RuntimeError("Stage 5D attribution failed exact frozen-model reproduction audit")

    features["abs_score_delta_ppg"] = features.score_delta_ppg.abs()
    features["abs_hit_delta_pp"] = features.hit_delta_pp.abs()
    families["abs_score_delta_ppg"] = families.score_delta_ppg.abs()
    families["abs_hit_delta_pp"] = families.hit_delta_pp.abs()

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    explanations["_priority"] = explanations.diagnostic_priority.map(priority_order).fillna(3)
    explanations = explanations.sort_values(["_priority", "position", "rank"], kind="stable").drop(columns="_priority")

    features.to_csv(OUT / "feature_attribution_2026.csv", index=False)
    families.to_csv(OUT / "family_attribution_2026.csv", index=False)
    explanations.to_csv(OUT / "prospect_explanations_2026.csv", index=False)
    audit.to_csv(OUT / "model_reproduction_audit.csv", index=False)

    # Concise report focuses on high-priority cases plus the strongest family-level drivers.
    report = []
    report.append("# Rookie Models Stage 5D: Feature Attribution and Disagreement Diagnosis\n")
    report.append(f"Generated {datetime.now(timezone.utc).isoformat()}\n")
    report.append("Stage 5D does not change the frozen model. It measures local sensitivity by replacing one current-player input, or one football feature family, with the historical same-position median and re-running the exact frozen Stage 3 regression and hit classifier. Positive deltas mean the observed value raises the player's frozen-model output relative to that neutral median. These are perturbation sensitivities, not causal effects and not additive SHAP values.\n")
    report.append("## Exact reproduction audit\n")
    report.append("|Pos|Current N|Union features|Regression max error|Hit-prob max error|Pass?|\n|---|---:|---:|---:|---:|---|\n")
    for _, r in audit.iterrows():
        ok = "Yes" if r.regression_reproduced == 1 and r.hit_probability_reproduced == 1 else "No"
        report.append(f"|{r.position}|{int(r.current_rows)}|{int(r.union_features)}|{r.max_abs_regression_reproduction_error:.2e}|{r.max_abs_hit_probability_reproduction_error:.2e}|{ok}|\n")

    report.append("\n## High-priority 2026 diagnoses\n")
    high = explanations[explanations.diagnostic_priority.eq("High")].copy()
    for _, r in high.iterrows():
        report.append(f"### {r.pfr_name} ({r.position})\n")
        report.append(f"- Grade: {n(r.prospect_model_percentile):.1f}th percentile; projected best-2-of-3 PPG: {n(r.pred_best2of3_ppg):.2f}; classifier hit: {100*n(r.hit_probability):.1f}%.\n")
        report.append(f"- Stage 5C flag: {r.diagnostic_notes}. Confidence: {r.model_confidence}.\n")
        report.append(f"- Regression-positive families: {r.top_score_positive_families or 'none'}.\n")
        report.append(f"- Regression-negative families: {r.top_score_negative_families or 'none'}.\n")
        report.append(f"- Classifier-positive families: {r.top_hit_positive_families or 'none'}.\n")
        report.append(f"- Classifier-negative families: {r.top_hit_negative_families or 'none'}.\n")
        report.append(f"- Strongest positive score features: {r.top_score_positive_features or 'none'}.\n")
        report.append(f"- Strongest negative score features: {r.top_score_negative_features or 'none'}.\n\n")

    report.append("## Interpretation guardrails\n")
    report.append("- A positive local delta means that neutralizing that observed input to the historical position median lowers the output. It does not prove the feature causes NFL success.\n")
    report.append("- Tree-model interactions mean single-feature deltas do not sum to the full prediction. Family-level neutralization is included to expose interaction-sensitive football themes.\n")
    report.append("- Missing-value indicators can matter. When a current input is missing, replacing it with the historical median measures both value imputation and removal of the model's missingness signal.\n")
    report.append("- Regression grade and hit classifier are intentionally explained separately because Stage 5C showed meaningful disagreement for several prospects.\n")
    (OUT / "REPORT.md").write_text("".join(report))

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "5D",
        "model_changed": False,
        "model_selection_changed": False,
        "method": "Local historical-median neutralization sensitivity on exact frozen Stage 3 regression and hit classifier, both per feature and per football feature family.",
        "interpretation": "Positive delta means the player's observed input raises the frozen output versus replacing that input with the historical same-position median. Effects are local, non-causal, and non-additive.",
        "outputs": [
            "feature_attribution_2026.csv",
            "family_attribution_2026.csv",
            "prospect_explanations_2026.csv",
            "model_reproduction_audit.csv",
            "REPORT.md",
        ],
        "inputs": [
            "results_v3/prospect_pool_v3.csv",
            "results_2026/prospect_pool_2026.csv",
            "results_stage5c/rookie_rankings_2026_stage5c.csv",
            "trained_v3/{position}.joblib",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
