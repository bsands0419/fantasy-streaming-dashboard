from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import stage6a_scout_ablation as s

ROOT = Path(__file__).resolve().parent
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CUR_POOL = ROOT / "results_2026" / "prospect_pool_2026.csv"
CURRENT_RANK = ROOT / "results_stage5c" / "rookie_rankings_2026_stage5c.csv"
V3 = ROOT / "results_v3"
MODELS = ROOT / "trained_v3"
OUT = ROOT / "results_stage6a"
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ["WR", "TE"]


def saved_frozen_metrics(pos: str) -> list[dict]:
    out = []
    for split, reg_file, clf_file in [
        ("validation_2019_2022", f"{pos}_validation_predictions.csv", f"{pos}_validation_hit_probs.csv"),
        ("final_2023", f"{pos}_final_2023_predictions.csv", f"{pos}_final_2023_hit_probs.csv"),
    ]:
        reg = pd.read_csv(V3 / reg_file)
        clf = pd.read_csv(V3 / clf_file)
        rm = s.regression_metrics(reg.primary_ppg, reg.pred)
        cm = s.classifier_metrics(clf.hit3, clf.prob)
        out.append({
            "position": pos,
            "variant": "serialized_frozen_reference",
            "split": split,
            "removed_primary_scout_features": 0,
            "removed_secondary_scout_features": 0,
            **{f"reg_{k}": v for k, v in rm.items()},
            **{f"hit_{k}": v for k, v in cm.items()},
        })
    return out


def main():
    hist = pd.read_csv(HIST_POOL, low_memory=False)
    cur = pd.read_csv(CUR_POOL, low_memory=False)
    published = pd.read_csv(CURRENT_RANK, low_memory=False)
    hist["season"] = pd.to_numeric(hist.season, errors="coerce")
    cur["season"] = pd.to_numeric(cur.season, errors="coerce")
    if "target_valid" in hist.columns:
        hist = hist[pd.to_numeric(hist.target_valid, errors="coerce").eq(1)].copy()
    hist = hist[hist.season.le(2023)].copy()
    cur = cur[cur.season.eq(2026)].copy()

    metrics = []
    pred_rows = []
    drift_rows = []
    feature_rows = []

    for pos in POSITIONS:
        job = joblib.load(MODELS / f"{pos}.joblib")
        all_features = list(dict.fromkeys(list(job.get("features", [])) + list(job.get("features_b") or [])))
        d = s.add_missing(hist[hist.position.eq(pos)].copy().reset_index(drop=True), all_features)
        c = s.add_missing(cur[cur.position.eq(pos)].copy().reset_index(drop=True), all_features)

        metrics.extend(saved_frozen_metrics(pos))

        for variant, remove_scout in [("snapshot_refit_with_scout", False), ("snapshot_refit_no_scout", True)]:
            primary, secondary = s.variant_features(job, remove_scout)
            removed_primary = sum(f in job.get("features", []) for f in s.SCOUT_FEATURES) if remove_scout else 0
            removed_secondary = sum(f in (job.get("features_b") or []) for f in s.SCOUT_FEATURES) if remove_scout else 0
            feature_rows.append({
                "position": pos,
                "variant": variant,
                "primary_feature_count": len(primary),
                "secondary_feature_count": len(secondary),
                "removed_primary_scout_features": removed_primary,
                "removed_secondary_scout_features": removed_secondary,
                "scout_features_in_frozen_primary": "; ".join(f for f in s.SCOUT_FEATURES if f in job.get("features", [])),
                "scout_features_in_frozen_secondary": "; ".join(f for f in s.SCOUT_FEATURES if f in (job.get("features_b") or [])),
            })

            val_reg, val_clf = s.walk_forward(d, job, primary, secondary)
            final_reg, final_clf = s.final_2023(d, job, primary, secondary)
            metrics.append(s.metric_row(pos, variant, "validation_2019_2022", val_reg, val_clf, removed_primary, removed_secondary))
            metrics.append(s.metric_row(pos, variant, "final_2023", final_reg, final_clf, removed_primary, removed_secondary))
            current = s.score_current(d, c, job, primary, secondary)
            current["variant"] = variant
            pred_rows.append(current)

        refit = pred_rows[-2]
        pub = published[published.position.eq(pos)][["position", "pfr_name", "pred_best2of3_ppg", "hit_probability"]].copy()
        chk = refit.merge(pub, on=["position", "pfr_name"], suffixes=("_refit", "_serialized"))
        drift_rows.append({
            "position": pos,
            "n": len(chk),
            "max_abs_score_refit_vs_serialized": float((chk.pred_best2of3_ppg_refit - chk.pred_best2of3_ppg_serialized).abs().max()),
            "mean_abs_score_refit_vs_serialized": float((chk.pred_best2of3_ppg_refit - chk.pred_best2of3_ppg_serialized).abs().mean()),
            "max_abs_hit_prob_refit_vs_serialized": float((chk.hit_probability_refit - chk.hit_probability_serialized).abs().max()),
            "mean_abs_hit_prob_refit_vs_serialized": float((chk.hit_probability_refit - chk.hit_probability_serialized).abs().mean()),
        })

    metric_df = pd.DataFrame(metrics)
    predictions = pd.concat(pred_rows, ignore_index=True)
    drift = pd.DataFrame(drift_rows)
    feature_audit = pd.DataFrame(feature_rows)

    refit = metric_df[metric_df.variant.eq("snapshot_refit_with_scout")].drop(columns="variant")
    no = metric_df[metric_df.variant.eq("snapshot_refit_no_scout")].drop(columns="variant")
    paired = no.merge(refit, on=["position", "split"], suffixes=("_no_scout", "_with_scout"))
    deltas = []
    for _, r in paired.iterrows():
        deltas.append({
            "position": r.position,
            "split": r.split,
            "primary_scout_features_removed": int(r.removed_primary_scout_features_no_scout),
            "secondary_scout_features_removed": int(r.removed_secondary_scout_features_no_scout),
            "delta_mae_no_scout_minus_with_scout": r.reg_mae_no_scout - r.reg_mae_with_scout,
            "delta_rmse_no_scout_minus_with_scout": r.reg_rmse_no_scout - r.reg_rmse_with_scout,
            "delta_spearman_no_scout_minus_with_scout": r.reg_spearman_no_scout - r.reg_spearman_with_scout,
            "delta_pearson_no_scout_minus_with_scout": r.reg_pearson_no_scout - r.reg_pearson_with_scout,
            "delta_hit_auc_no_scout_minus_with_scout": r.hit_auc_no_scout - r.hit_auc_with_scout,
            "delta_hit_brier_no_scout_minus_with_scout": r.hit_brier_no_scout - r.hit_brier_with_scout,
        })
    delta_df = pd.DataFrame(deltas)

    p_with = predictions[predictions.variant.eq("snapshot_refit_with_scout")].drop(columns="variant")
    p_no = predictions[predictions.variant.eq("snapshot_refit_no_scout")].drop(columns="variant")
    impact = p_no.merge(p_with, on=["position", "pfr_name"], suffixes=("_no_scout", "_with_scout"))
    impact["score_delta_no_scout_minus_with_scout"] = impact.pred_best2of3_ppg_no_scout - impact.pred_best2of3_ppg_with_scout
    impact["hit_probability_delta_pp_no_scout_minus_with_scout"] = 100.0 * (impact.hit_probability_no_scout - impact.hit_probability_with_scout)
    pub = published[published.position.isin(POSITIONS)][["position", "pfr_name", "pred_best2of3_ppg", "hit_probability"]]
    impact = impact.merge(pub, on=["position", "pfr_name"], how="left")
    impact = impact.rename(columns={"pred_best2of3_ppg": "serialized_frozen_pred_best2of3_ppg", "hit_probability": "serialized_frozen_hit_probability"})
    impact["abs_score_scout_ablation_effect"] = impact.score_delta_no_scout_minus_with_scout.abs()
    impact = impact.sort_values(["position", "abs_score_scout_ablation_effect"], ascending=[True, False], kind="stable")

    metric_df.to_csv(OUT / "scout_ablation_metrics.csv", index=False)
    delta_df.to_csv(OUT / "scout_ablation_deltas.csv", index=False)
    predictions.to_csv(OUT / "scout_ablation_2026_predictions.csv", index=False)
    impact.to_csv(OUT / "scout_ablation_2026_impact.csv", index=False)
    drift.to_csv(OUT / "serialized_refit_drift_audit.csv", index=False)
    feature_audit.to_csv(OUT / "scout_ablation_feature_audit.csv", index=False)

    report = [
        "# Rookie Models Stage 6A: Scouting-Surprise Ablation\n\n",
        f"Generated {datetime.now(timezone.utc).isoformat()}\n\n",
        "Stage 6A holds model classes, hyperparameters, blend weights, target definitions, and walk-forward windows fixed. The experimental comparison is between two fresh refits on the identical saved historical snapshot: one with the frozen scout fields and one with only `scout_expected_log_pick` and `scout_boost` removed. The serialized Stage 3 model is retained as a separate published reference. This avoids confusing tree-model refit drift after CSV round-tripping with the effect of the scout variables.\n\n",
        "The scout fields are pre-NFL and leakage-safe with respect to NFL outcomes. This is a feature-taxonomy and robustness test, not an outcome-leakage correction.\n\n",
        "## Serialized model versus snapshot refit drift\n\n",
        "|Pos|N|Max score drift|Mean score drift|Max hit-prob drift|Mean hit-prob drift|\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in drift.iterrows():
        report.append(f"|{r.position}|{int(r.n)}|{r.max_abs_score_refit_vs_serialized:.4f}|{r.mean_abs_score_refit_vs_serialized:.4f}|{r.max_abs_hit_prob_refit_vs_serialized:.4f}|{r.mean_abs_hit_prob_refit_vs_serialized:.4f}|\n")

    report.extend([
        "\n## Scout ablation deltas\n\n",
        "The table below isolates the scout-variable effect because both sides are fit from the same snapshot. Negative MAE/RMSE/Brier and positive Spearman/Pearson/AUC deltas favor removing the scout fields.\n\n",
        "|Pos|Split|Removed P/S|ΔMAE|ΔRMSE|ΔSpearman|ΔPearson|ΔHit AUC|ΔHit Brier|\n|---|---|---|---:|---:|---:|---:|---:|---:|\n",
    ])
    for _, r in delta_df.iterrows():
        report.append(
            f"|{r.position}|{r.split}|{int(r.primary_scout_features_removed)}/{int(r.secondary_scout_features_removed)}|"
            f"{r.delta_mae_no_scout_minus_with_scout:+.3f}|{r.delta_rmse_no_scout_minus_with_scout:+.3f}|"
            f"{r.delta_spearman_no_scout_minus_with_scout:+.3f}|{r.delta_pearson_no_scout_minus_with_scout:+.3f}|"
            f"{r.delta_hit_auc_no_scout_minus_with_scout:+.3f}|{r.delta_hit_brier_no_scout_minus_with_scout:+.3f}|\n"
        )

    report.append("\n## Largest 2026 ablation effects\n\n")
    report.append("|Pos|Player|With-scout refit|No-scout refit|ΔPPG|With-scout hit|No-scout hit|Δ hit pp|Serialized frozen PPG|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in impact.sort_values("abs_score_scout_ablation_effect", ascending=False).head(30).iterrows():
        report.append(
            f"|{r.position}|{r.pfr_name}|{r.pred_best2of3_ppg_with_scout:.2f}|{r.pred_best2of3_ppg_no_scout:.2f}|"
            f"{r.score_delta_no_scout_minus_with_scout:+.2f}|{100*r.hit_probability_with_scout:.1f}%|{100*r.hit_probability_no_scout:.1f}%|"
            f"{r.hit_probability_delta_pp_no_scout_minus_with_scout:+.1f}|{r.serialized_frozen_pred_best2of3_ppg:.2f}|\n"
        )

    report.extend([
        "\n## Decision framework\n\n",
        "No Stage 6 experiment replaces the frozen model automatically. Pre-2023 validation is the selection evidence; the 2023 class remains a frozen confirmation check. If scout removal is better or effectively neutral on validation, Stage 6B should explicitly exclude scout fields from generic production groups. If scout inclusion is clearly beneficial, Stage 6B may retain them, but only as an explicitly named scouting-surprise feature family so their contribution is deliberate and auditable.\n",
    ])
    (OUT / "REPORT.md").write_text("".join(report))

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "6A",
        "experiment": "scouting-surprise ablation",
        "affected_positions": POSITIONS,
        "published_frozen_model_changed": False,
        "comparison": "snapshot_refit_with_scout versus snapshot_refit_no_scout",
        "held_fixed": ["model classes", "hyperparameters", "blend weights", "target", "validation years", "final-test year", "historical snapshot"],
        "removed_only": s.SCOUT_FEATURES,
        "important_note": "Serialized frozen model is a reference, not one side of the ablation, because fresh tree-model refits from CSV can drift slightly. Scout fields are pre-NFL and outcome-leakage-safe.",
        "outputs": ["scout_ablation_metrics.csv", "scout_ablation_deltas.csv", "scout_ablation_2026_predictions.csv", "scout_ablation_2026_impact.csv", "serialized_refit_drift_audit.csv", "scout_ablation_feature_audit.csv", "REPORT.md"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
