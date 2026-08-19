from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import stage5c_uncertainty as s5c


def add_tail_monotonic_guard(err_cal: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    monotonic_flags = []
    reason = []
    for _, p in out.iterrows():
        pos = p.position
        d = err_cal[err_cal.position.eq(pos)].set_index("confidence_group")
        needed = ["High", "Medium", "Low"]
        enough = all(g in d.index and int(d.loc[g, "n"]) >= s5c.MIN_GROUP_N for g in needed)
        if enough:
            q80 = [float(d.loc[g, "q80_abs_error"]) for g in needed]
            q90 = [float(d.loc[g, "q90_abs_error"]) for g in needed]
            monotonic = bool(
                all(np.isfinite(q80)) and all(np.isfinite(q90)) and
                q80[0] <= q80[1] <= q80[2] and
                q90[0] <= q90[1] <= q90[2]
            )
        else:
            monotonic = False
        monotonic_flags.append(int(monotonic))
        if not enough:
            reason.append("insufficient confidence-group sample")
        elif not monotonic:
            reason.append("confidence tail errors are not monotonic High <= Medium <= Low")
        elif int(p.adaptive_uncertainty_supported) != 1:
            reason.append("correlation/significance/material-spread rule not satisfied")
        else:
            reason.append("all adaptive uncertainty criteria satisfied")

    out["confidence_tail_error_monotonic"] = monotonic_flags
    out["adaptive_guard_reason"] = reason
    out["adaptive_uncertainty_supported_pre_tail_guard"] = out["adaptive_uncertainty_supported"].astype(int)
    out["adaptive_uncertainty_supported"] = (
        out["adaptive_uncertainty_supported"].astype(int) * out["confidence_tail_error_monotonic"].astype(int)
    )
    return out


def write_report_v2(policy: pd.DataFrame, rankings: pd.DataFrame) -> None:
    lines = [
        "# Rookie Models Stage 5C: Uncertainty and Diagnosis",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
        "Stage 5C tests whether Stage 5B model-familiarity confidence historically relates to smaller frozen-model OOF errors. Confidence may change displayed prediction bands only if both average-error and tail-error diagnostics support it.",
        "",
        "## Confidence/error test",
        "",
        "|Pos|N|Spearman(confidence, abs error)|p|Tail errors monotonic?|Adaptive bands?|",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, r in policy.iterrows():
        lines.append(
            f"|{r.position}|{int(r.historical_error_n)}|{r.spearman_confidence_vs_abs_error:.3f}|"
            f"{r.spearman_p_value:.3f}|{'Yes' if int(r.confidence_tail_error_monotonic) else 'No'}|"
            f"{'Yes' if int(r.adaptive_uncertainty_supported) else 'No'}|"
        )

    lines += [
        "",
        "The tail guard requires the historical 80th- and 90th-percentile absolute errors to satisfy High <= Medium <= Low with at least 15 observations in each group. This prevents a significant average-error correlation from being used to narrow/widen intervals when the actual prediction tails do not behave monotonically.",
        "",
        "## Highest diagnostic disagreement in 2026",
        "",
        "|Pos|Player|Prospect pct|Classifier hit|Bucket hit|5-comp hit|Confidence|Priority|",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    z = rankings.sort_values("hit_signal_disagreement_pp", ascending=False).head(20)
    for _, r in z.iterrows():
        lines.append(
            f"|{r.position}|{r.pfr_name}|{r.prospect_model_percentile:.1f}|{100*r.hit_probability:.1f}%|"
            f"{100*r.historical_hit_rate:.1f}%|{100*r.nearest5_historical_hit_rate:.1f}%|{r.model_confidence}|{r.diagnostic_priority}|"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Prospect percentile is a grade, not a success probability. Historical bucket hit rate is empirical calibration. The classifier is a separate supervised probability model. Five-neighbor hit rate is a descriptive comp summary. Model confidence measures familiarity with the frozen feature space. Stage 5C keeps these signals separate and surfaces disagreement rather than averaging them into a false single probability.",
    ]
    (s5c.OUT / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    hist_pool = pd.read_csv(s5c.HIST_POOL, low_memory=False)
    cal = pd.read_csv(s5c.CAL)
    rankings = pd.read_csv(s5c.RANK)

    hist_pool["season"] = pd.to_numeric(hist_pool["season"], errors="coerce")
    if "target_valid" in hist_pool.columns:
        hist_pool = hist_pool[pd.to_numeric(hist_pool.target_valid, errors="coerce").eq(1)].copy()
    hist_pool = hist_pool[hist_pool.season <= 2023].copy()

    hist_rows, feature_audit = s5c.historical_confidence_rows(hist_pool, cal)
    err_cal, policy = s5c.confidence_calibration(hist_rows)
    policy = add_tail_monotonic_guard(err_cal, policy)
    current = s5c.apply_uncertainty_and_diagnostics(rankings, err_cal, policy)

    hist_rows.to_csv(s5c.OUT / "historical_confidence_error_rows.csv", index=False)
    feature_audit.to_csv(s5c.OUT / "historical_confidence_feature_audit.csv", index=False)
    err_cal.to_csv(s5c.OUT / "confidence_error_calibration.csv", index=False)
    policy.to_csv(s5c.OUT / "uncertainty_policy_by_position.csv", index=False)
    current.to_csv(s5c.OUT / "rookie_rankings_2026_stage5c.csv", index=False)
    current[[
        "position", "rank", "pfr_name", "prospect_model_percentile", "historical_hit_rate", "historical_star_rate",
        "hit_probability", "nearest5_historical_hit_rate", "nearest5_historical_star_rate", "model_confidence",
        "out_of_distribution_flag", "hit_signal_disagreement_pp", "hit_signal_disagreement_percentile_2026",
        "diagnostic_priority", "diagnostic_notes", "uncertainty_policy", "stage5c_pred80_low", "stage5c_pred80_high",
        "stage5c_pred90_low", "stage5c_pred90_high", "closest_historical_comps", "diagnosis_summary"
    ]].to_csv(s5c.OUT / "prospect_diagnosis_2026.csv", index=False)
    write_report_v2(policy, current)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "5C",
        "revision": "tail-monotonic guard",
        "model_changed": False,
        "model_selection_changed": False,
        "purpose": "Test whether Stage 5B model familiarity predicts historical OOF error, conditionally calibrate displayed uncertainty widths, and surface disagreements among grade, calibrated hit rate, classifier probability, and nearest-neighbor outcomes.",
        "adaptive_uncertainty_rule": {
            "average_error_gate": f"Spearman rho <= {s5c.RHO_THRESHOLD}, p <= {s5c.P_THRESHOLD}, at least two confidence groups n >= {s5c.MIN_GROUP_N}, material q80 spread.",
            "tail_error_gate": f"High <= Medium <= Low for both q80 and q90 absolute errors, with all three confidence groups n >= {s5c.MIN_GROUP_N}.",
            "result": "Only if both gates pass may Stage 5C scale the original Stage 3 position-level bands.",
        },
        "important_note": "No probability signals are averaged. Prospect percentile remains a grade; bucket hit rate is empirical calibration; classifier hit probability is a separate supervised model; five-comp hit rate is descriptive; confidence is feature-space familiarity.",
        "inputs": [
            "results_v3/prospect_pool_v3.csv",
            "results_stage5/historical_prospect_calibration_rows.csv",
            "results_stage5b/rookie_rankings_2026_stage5b.csv",
            "trained_v3/{position}.joblib",
        ],
    }
    (s5c.OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
