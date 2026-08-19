from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import stage5b_comps as s5b

ROOT = Path(__file__).resolve().parent
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CAL = ROOT / "results_stage5" / "historical_prospect_calibration_rows.csv"
RANK = ROOT / "results_stage5b" / "rookie_rankings_2026_stage5b.csv"
MODELS = ROOT / "trained_v3"
OUT = ROOT / "results_stage5c"
OUT.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
MIN_GROUP_N = 15
RHO_THRESHOLD = -0.10
P_THRESHOLD = 0.10
MIN_Q80_SPREAD_FRAC = 0.10
MIN_SCALE = 0.75
MAX_SCALE = 1.35


def pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def q(v: pd.Series, quantile: float) -> float:
    x = pd.to_numeric(v, errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(x, quantile)) if len(x) else np.nan


def historical_confidence_rows(hist_pool: pd.DataFrame, cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = [
        "season", "pfr_name", "position", "prospect_model_score", "prospect_model_percentile",
        "primary_ppg", "hit3", "star3", "best_rank3"
    ]
    labels = cal[[c for c in label_cols if c in cal.columns]].drop_duplicates(["season", "pfr_name", "position"])
    hist = hist_pool.merge(labels, on=["season", "pfr_name", "position"], how="left", suffixes=("", "_cal"))

    all_rows = []
    feature_rows = []
    for pos in POSITIONS:
        job = joblib.load(MODELS / f"{pos}.joblib")
        features = s5b.ordered_union(job.get("features", []), job.get("features_b", []))
        h = hist[hist.position.eq(pos)].copy().reset_index(drop=True)
        if len(h) <= s5b.K:
            continue

        usable, hx, _, coverage = s5b.prepare_space(h, h, features)
        d5 = s5b.historical_density_reference(hx, s5b.K)
        p90 = float(np.quantile(d5, .90))

        feature_rows.append({
            "position": pos,
            "historical_rows": len(h),
            "frozen_model_features": len(features),
            "usable_similarity_features": len(usable),
            "historical_loo_mean_5nn_distance": float(np.mean(d5)),
            "historical_loo_p90_5nn_distance": p90,
        })

        ppg_col = pick_col(h, ["primary_ppg_cal", "primary_ppg"])
        for i, r in h.iterrows():
            density_pct = 100.0 * float(np.mean(d5 >= d5[i]))
            coverage_pct = 100.0 * float(np.clip(coverage[i], 0, 1))
            score = 0.70 * density_pct + 0.30 * coverage_pct
            ood = bool(d5[i] > p90 or coverage[i] < 0.75)
            if score >= 70 and not ood:
                label = "High"
            elif score >= 35 and not ood:
                label = "Medium"
            else:
                label = "Low"

            pred = pd.to_numeric(pd.Series([r.get("prospect_model_score")]), errors="coerce").iloc[0]
            actual = pd.to_numeric(pd.Series([r.get(ppg_col) if ppg_col else np.nan]), errors="coerce").iloc[0]
            signed = actual - pred if np.isfinite(actual) and np.isfinite(pred) else np.nan
            all_rows.append({
                "season": r.get("season"),
                "position": pos,
                "pfr_name": r.get("pfr_name"),
                "prospect_model_score": pred,
                "primary_ppg": actual,
                "signed_error": signed,
                "abs_error": abs(signed) if np.isfinite(signed) else np.nan,
                "feature_coverage": float(coverage[i]),
                "mean_5nn_distance": float(d5[i]),
                "neighbor_density_percentile": density_pct,
                "model_confidence_score": score,
                "model_confidence": label,
                "out_of_distribution_flag": int(ood),
                "similarity_feature_count": len(usable),
            })

    return pd.DataFrame(all_rows), pd.DataFrame(feature_rows)


def confidence_calibration(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    policies = []

    for pos in POSITIONS:
        d = hist[hist.position.eq(pos)].dropna(subset=["abs_error", "model_confidence_score"]).copy()
        if d.empty:
            continue

        rho_obj = spearmanr(d.model_confidence_score, d.abs_error)
        rho = float(rho_obj.statistic) if np.isfinite(rho_obj.statistic) else np.nan
        pval = float(rho_obj.pvalue) if np.isfinite(rho_obj.pvalue) else np.nan
        overall_q80 = q(d.abs_error, .80)
        overall_q90 = q(d.abs_error, .90)

        rows.append({
            "position": pos, "confidence_group": "All", "n": len(d),
            "mae": float(d.abs_error.mean()), "median_abs_error": float(d.abs_error.median()),
            "q80_abs_error": overall_q80, "q90_abs_error": overall_q90,
            "mean_signed_error": float(d.signed_error.mean()),
        })

        group_stats = {}
        for label in ["High", "Medium", "Low"]:
            g = d[d.model_confidence.eq(label)]
            row = {
                "position": pos, "confidence_group": label, "n": len(g),
                "mae": float(g.abs_error.mean()) if len(g) else np.nan,
                "median_abs_error": float(g.abs_error.median()) if len(g) else np.nan,
                "q80_abs_error": q(g.abs_error, .80), "q90_abs_error": q(g.abs_error, .90),
                "mean_signed_error": float(g.signed_error.mean()) if len(g) else np.nan,
            }
            rows.append(row)
            group_stats[label] = row

        ood = d[d.out_of_distribution_flag.eq(1)]
        rows.append({
            "position": pos, "confidence_group": "OOD", "n": len(ood),
            "mae": float(ood.abs_error.mean()) if len(ood) else np.nan,
            "median_abs_error": float(ood.abs_error.median()) if len(ood) else np.nan,
            "q80_abs_error": q(ood.abs_error, .80), "q90_abs_error": q(ood.abs_error, .90),
            "mean_signed_error": float(ood.signed_error.mean()) if len(ood) else np.nan,
        })

        eligible_q80 = [v["q80_abs_error"] for v in group_stats.values() if v["n"] >= MIN_GROUP_N and np.isfinite(v["q80_abs_error"])]
        q80_spread = max(eligible_q80) - min(eligible_q80) if len(eligible_q80) >= 2 else 0.0
        material = bool(np.isfinite(overall_q80) and overall_q80 > 0 and q80_spread >= MIN_Q80_SPREAD_FRAC * overall_q80)
        supported = bool(
            np.isfinite(rho) and rho <= RHO_THRESHOLD and
            np.isfinite(pval) and pval <= P_THRESHOLD and
            len(eligible_q80) >= 2 and material
        )

        job = joblib.load(MODELS / f"{pos}.joblib")
        policies.append({
            "position": pos,
            "historical_error_n": len(d),
            "spearman_confidence_vs_abs_error": rho,
            "spearman_p_value": pval,
            "historical_overall_q80": overall_q80,
            "historical_overall_q90": overall_q90,
            "eligible_confidence_groups": len(eligible_q80),
            "q80_group_spread": q80_spread,
            "q80_spread_material": int(material),
            "adaptive_uncertainty_supported": int(supported),
            "stage3_base_q80": float(job.get("interval_abs_error_q80", np.nan)),
            "stage3_base_q90": float(job.get("interval_abs_error_q90", np.nan)),
        })

    return pd.DataFrame(rows), pd.DataFrame(policies)


def grade_tier(x: float) -> str:
    if not np.isfinite(x):
        return "Unknown"
    if x >= 95: return "Elite"
    if x >= 80: return "Strong"
    if x >= 60: return "Above average"
    if x >= 40: return "Average"
    return "Below average"


def apply_uncertainty_and_diagnostics(rankings: pd.DataFrame, cal: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    r = rankings.copy()
    cal_map = {(x.position, x.confidence_group): x for _, x in cal.iterrows()}
    pol_map = {x.position: x for _, x in policy.iterrows()}

    out_rows = []
    for _, row in r.iterrows():
        z = row.copy()
        pos = z.position
        pol = pol_map.get(pos)
        label = str(z.get("model_confidence", "Low"))
        grp = cal_map.get((pos, label))
        allg = cal_map.get((pos, "All"))

        scale80 = scale90 = 1.0
        applied = False
        if pol is not None and int(pol.adaptive_uncertainty_supported) == 1 and grp is not None and allg is not None and int(grp.n) >= MIN_GROUP_N:
            if np.isfinite(grp.q80_abs_error) and np.isfinite(allg.q80_abs_error) and allg.q80_abs_error > 0:
                scale80 = float(np.clip(grp.q80_abs_error / allg.q80_abs_error, MIN_SCALE, MAX_SCALE))
            if np.isfinite(grp.q90_abs_error) and np.isfinite(allg.q90_abs_error) and allg.q90_abs_error > 0:
                scale90 = float(np.clip(grp.q90_abs_error / allg.q90_abs_error, MIN_SCALE, MAX_SCALE))
            applied = True

        pred = float(z.get("pred_best2of3_ppg", np.nan))
        base80 = float(pol.stage3_base_q80) if pol is not None else float(z.get("pred80_high", np.nan) - pred)
        base90 = float(pol.stage3_base_q90) if pol is not None else float(z.get("pred90_high", np.nan) - pred)
        width80 = base80 * scale80
        width90 = base90 * scale90

        z["stage5c_pred80_low"] = max(0.0, pred - width80) if np.isfinite(pred) and np.isfinite(width80) else np.nan
        z["stage5c_pred80_high"] = pred + width80 if np.isfinite(pred) and np.isfinite(width80) else np.nan
        z["stage5c_pred90_low"] = max(0.0, pred - width90) if np.isfinite(pred) and np.isfinite(width90) else np.nan
        z["stage5c_pred90_high"] = pred + width90 if np.isfinite(pred) and np.isfinite(width90) else np.nan
        z["uncertainty_scale80"] = scale80
        z["uncertainty_scale90"] = scale90
        z["uncertainty_adjustment_applied"] = int(applied)
        z["uncertainty_policy"] = "confidence-adjusted" if applied else "original-position-band"

        hp = pd.to_numeric(pd.Series([z.get("hit_probability")]), errors="coerce").iloc[0]
        bh = pd.to_numeric(pd.Series([z.get("historical_hit_rate")]), errors="coerce").iloc[0]
        ch = pd.to_numeric(pd.Series([z.get("nearest5_historical_hit_rate")]), errors="coerce").iloc[0]
        probs = [v for v in [hp, bh, ch] if np.isfinite(v)]
        diffs = [abs(probs[i] - probs[j]) for i in range(len(probs)) for j in range(i + 1, len(probs))]
        z["hit_signal_disagreement_pp"] = 100.0 * float(np.mean(diffs)) if diffs else np.nan
        z["classifier_minus_bucket_hit_pp"] = 100.0 * (hp - bh) if np.isfinite(hp) and np.isfinite(bh) else np.nan
        z["comps_minus_bucket_hit_pp"] = 100.0 * (ch - bh) if np.isfinite(ch) and np.isfinite(bh) else np.nan
        z["classifier_minus_comps_hit_pp"] = 100.0 * (hp - ch) if np.isfinite(hp) and np.isfinite(ch) else np.nan
        z["grade_tier"] = grade_tier(float(z.get("prospect_model_percentile", np.nan)))
        out_rows.append(z)

    out = pd.DataFrame(out_rows)
    out["hit_signal_disagreement_percentile_2026"] = out.groupby("position")["hit_signal_disagreement_pp"].rank(method="average", pct=True) * 100.0

    priorities = []
    notes = []
    summaries = []
    for _, z in out.iterrows():
        n = []
        if int(z.get("out_of_distribution_flag", 0)) == 1:
            n.append("out-of-distribution profile")
        if str(z.get("model_confidence", "")) == "Low":
            n.append("low model familiarity")
        cb = z.get("classifier_minus_bucket_hit_pp")
        compb = z.get("comps_minus_bucket_hit_pp")
        if np.isfinite(cb) and cb <= -20: n.append("classifier materially below bucket hit rate")
        if np.isfinite(cb) and cb >= 20: n.append("classifier materially above bucket hit rate")
        if np.isfinite(compb) and compb <= -25: n.append("five comps materially below bucket hit rate")
        if np.isfinite(compb) and compb >= 25: n.append("five comps materially above bucket hit rate")
        pct = float(z.get("prospect_model_percentile", np.nan))
        hp = float(z.get("hit_probability", np.nan))
        if np.isfinite(pct) and pct >= 90 and np.isfinite(hp) and hp < .35:
            n.append("elite grade but modest classifier probability")

        dp = float(z.get("hit_signal_disagreement_percentile_2026", np.nan))
        if int(z.get("out_of_distribution_flag", 0)) == 1 or (np.isfinite(dp) and dp >= 80):
            priority = "High"
        elif str(z.get("model_confidence", "")) == "Low" or (np.isfinite(dp) and dp >= 50):
            priority = "Medium"
        else:
            priority = "Low"
        priorities.append(priority)
        notes.append("; ".join(n) if n else "signals broadly consistent")

        hit_txt = f"{100*float(z.historical_hit_rate):.0f}% historical bucket hit" if np.isfinite(z.get("historical_hit_rate", np.nan)) else "historical hit unavailable"
        clf_txt = f"{100*float(z.hit_probability):.0f}% classifier hit" if np.isfinite(z.get("hit_probability", np.nan)) else "classifier unavailable"
        summaries.append(
            f"{z.grade_tier} {z.position} prospect ({float(z.prospect_model_percentile):.1f}th pct); "
            f"{hit_txt}; {clf_txt}; {z.model_confidence} confidence; "
            f"80% range {float(z.stage5c_pred80_low):.1f}-{float(z.stage5c_pred80_high):.1f} PPG."
        )

    out["diagnostic_priority"] = priorities
    out["diagnostic_notes"] = notes
    out["diagnosis_summary"] = summaries
    return out.sort_values(["position", "rank"], kind="stable")


def write_report(policy: pd.DataFrame, rankings: pd.DataFrame) -> None:
    lines = [
        "# Rookie Models Stage 5C: Uncertainty and Diagnosis",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
        "Stage 5C tests whether Stage 5B model-familiarity confidence historically relates to smaller frozen-model OOF errors. Confidence changes displayed prediction bands only when a pre-specified diagnostic rule is satisfied; otherwise the original Stage 3 position-level bands are retained.",
        "",
        "## Confidence/error test",
        "",
        "|Pos|N|Spearman(confidence, abs error)|p|Adaptive bands?|",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in policy.iterrows():
        lines.append(f"|{r.position}|{int(r.historical_error_n)}|{r.spearman_confidence_vs_abs_error:.3f}|{r.spearman_p_value:.3f}|{'Yes' if int(r.adaptive_uncertainty_supported) else 'No'}|")

    lines += ["", "## Highest diagnostic disagreement in 2026", ""]
    z = rankings.sort_values(["diagnostic_priority", "hit_signal_disagreement_pp"], ascending=[True, False])
    z = rankings.sort_values("hit_signal_disagreement_pp", ascending=False).head(20)
    lines += ["|Pos|Player|Prospect pct|Classifier hit|Bucket hit|5-comp hit|Confidence|Priority|", "|---|---|---:|---:|---:|---:|---|---|"]
    for _, r in z.iterrows():
        lines.append(
            f"|{r.position}|{r.pfr_name}|{r.prospect_model_percentile:.1f}|{100*r.hit_probability:.1f}%|"
            f"{100*r.historical_hit_rate:.1f}%|{100*r.nearest5_historical_hit_rate:.1f}%|{r.model_confidence}|{r.diagnostic_priority}|"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Prospect percentile is a grade, not a success probability. Historical bucket hit rate is empirical calibration. The classifier is a separate supervised probability model. Five-neighbor hit rate is a descriptive comp summary. Model confidence measures familiarity with the frozen feature space. Stage 5C deliberately keeps these signals separate and surfaces disagreement rather than averaging them into a false single probability.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    hist_pool = pd.read_csv(HIST_POOL, low_memory=False)
    cal = pd.read_csv(CAL)
    rankings = pd.read_csv(RANK)

    hist_pool["season"] = pd.to_numeric(hist_pool["season"], errors="coerce")
    if "target_valid" in hist_pool.columns:
        hist_pool = hist_pool[pd.to_numeric(hist_pool.target_valid, errors="coerce").eq(1)].copy()
    hist_pool = hist_pool[hist_pool.season <= 2023].copy()

    hist_rows, feature_audit = historical_confidence_rows(hist_pool, cal)
    err_cal, policy = confidence_calibration(hist_rows)
    current = apply_uncertainty_and_diagnostics(rankings, err_cal, policy)

    hist_rows.to_csv(OUT / "historical_confidence_error_rows.csv", index=False)
    feature_audit.to_csv(OUT / "historical_confidence_feature_audit.csv", index=False)
    err_cal.to_csv(OUT / "confidence_error_calibration.csv", index=False)
    policy.to_csv(OUT / "uncertainty_policy_by_position.csv", index=False)
    current.to_csv(OUT / "rookie_rankings_2026_stage5c.csv", index=False)
    current[[
        "position", "rank", "pfr_name", "prospect_model_percentile", "historical_hit_rate", "historical_star_rate",
        "hit_probability", "nearest5_historical_hit_rate", "nearest5_historical_star_rate", "model_confidence",
        "out_of_distribution_flag", "hit_signal_disagreement_pp", "hit_signal_disagreement_percentile_2026",
        "diagnostic_priority", "diagnostic_notes", "uncertainty_policy", "stage5c_pred80_low", "stage5c_pred80_high",
        "stage5c_pred90_low", "stage5c_pred90_high", "closest_historical_comps", "diagnosis_summary"
    ]].to_csv(OUT / "prospect_diagnosis_2026.csv", index=False)
    write_report(policy, current)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "5C",
        "model_changed": False,
        "model_selection_changed": False,
        "purpose": "Test whether Stage 5B model familiarity predicts historical OOF error, conditionally calibrate displayed uncertainty widths, and surface disagreements among grade, calibrated hit rate, classifier probability, and nearest-neighbor outcomes.",
        "confidence_error_test": {
            "definition": "Pooled historical leave-one-out Stage 5B confidence versus absolute leakage-safe historical prospect-model OOF error.",
            "adaptive_rule": f"Use confidence-specific scaling only when Spearman rho <= {RHO_THRESHOLD}, p <= {P_THRESHOLD}, at least two confidence groups have n >= {MIN_GROUP_N}, and q80 spread is at least {MIN_Q80_SPREAD_FRAC:.0%} of overall q80.",
            "scaling": f"Scale original Stage 3 position-level 80/90 bands by confidence-group historical q80/q90 divided by overall historical q80/q90, clipped to {MIN_SCALE}-{MAX_SCALE}.",
        },
        "important_note": "No probability signals are averaged. Prospect percentile remains a grade; bucket hit rate is empirical calibration; classifier hit probability is a separate supervised model; five-comp hit rate is descriptive; confidence is feature-space familiarity.",
        "inputs": [
            "results_v3/prospect_pool_v3.csv",
            "results_stage5/historical_prospect_calibration_rows.csv",
            "results_stage5b/rookie_rankings_2026_stage5b.csv",
            "trained_v3/{position}.joblib",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
