from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CUR_POOL = ROOT / "results_2026" / "prospect_pool_2026.csv"
CURRENT_RANK = ROOT / "results_stage5c" / "rookie_rankings_2026_stage5c.csv"
MODELS = ROOT / "trained_v3"
OUT = ROOT / "results_stage6a"
OUT.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
VALID_YEARS = [2019, 2020, 2021, 2022]
FINAL_TEST_YEAR = 2023
SCOUT_FEATURES = ["scout_expected_log_pick", "scout_boost"]
MIN_TRAIN = 35


def add_missing(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in features:
        if c not in out.columns:
            out[c] = np.nan
    return out


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 4:
        return {"n": len(y), "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
    }


def classifier_metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    auc = float(roc_auc_score(y, p)) if len(y) and len(np.unique(y)) > 1 else np.nan
    brier = float(brier_score_loss(y, p)) if len(y) else np.nan
    return {"n": len(y), "auc": auc, "brier": brier}


def job_templates(job):
    a_fitted, b_fitted = job["fitted"]["primary_ppg"]
    return clone(a_fitted), clone(b_fitted) if b_fitted is not None else None, clone(job["classifier"])


def variant_features(job, remove_scout: bool):
    primary = list(job.get("features", []))
    secondary = list(job.get("features_b") or [])
    if remove_scout:
        primary = [f for f in primary if f not in SCOUT_FEATURES]
        secondary = [f for f in secondary if f not in SCOUT_FEATURES]
    return primary, secondary


def fit_predict_regression(train, test, job, primary, secondary):
    a_template, b_template, _ = job_templates(job)
    a = clone(a_template)
    a.fit(train[primary], train.primary_ppg)
    pa = np.asarray(a.predict(test[primary]), float)
    if b_template is None:
        return pa
    b = clone(b_template)
    b.fit(train[secondary], train.primary_ppg)
    pb = np.asarray(b.predict(test[secondary]), float)
    w = float(job["blend"]["weight"])
    return w * pa + (1.0 - w) * pb


def fit_predict_classifier(train, test, job, primary):
    _, _, clf_template = job_templates(job)
    clf = clone(clf_template)
    clf.fit(train[primary], train.hit3.astype(int))
    return np.asarray(clf.predict_proba(test[primary])[:, 1], float)


def walk_forward(d, job, primary, secondary):
    reg_rows = []
    clf_rows = []
    for year in VALID_YEARS:
        tr = d[(d.season < year) & d.primary_ppg.notna()].copy()
        te = d[(d.season == year) & d.primary_ppg.notna()].copy()
        if len(tr) < MIN_TRAIN or te.empty:
            continue
        pred = fit_predict_regression(tr, te, job, primary, secondary)
        for (_, r), p in zip(te.iterrows(), pred):
            reg_rows.append({"season": year, "pfr_name": r.pfr_name, "primary_ppg": r.primary_ppg, "pred": float(p)})

        htr = d[(d.season < year) & d.hit3.notna()].copy()
        hte = d[(d.season == year) & d.hit3.notna()].copy()
        if len(htr) >= MIN_TRAIN and not hte.empty and htr.hit3.nunique() > 1:
            prob = fit_predict_classifier(htr, hte, job, primary)
            for (_, r), p in zip(hte.iterrows(), prob):
                clf_rows.append({"season": year, "pfr_name": r.pfr_name, "hit3": int(r.hit3), "prob": float(p)})
    return pd.DataFrame(reg_rows), pd.DataFrame(clf_rows)


def final_2023(d, job, primary, secondary):
    tr = d[(d.season < FINAL_TEST_YEAR) & d.primary_ppg.notna()].copy()
    te = d[(d.season == FINAL_TEST_YEAR) & d.primary_ppg.notna()].copy()
    pred = fit_predict_regression(tr, te, job, primary, secondary) if len(tr) >= MIN_TRAIN and not te.empty else np.array([])
    reg = te[["season", "pfr_name", "primary_ppg"]].copy()
    reg["pred"] = pred

    htr = d[(d.season < FINAL_TEST_YEAR) & d.hit3.notna()].copy()
    hte = d[(d.season == FINAL_TEST_YEAR) & d.hit3.notna()].copy()
    clf = hte[["season", "pfr_name", "hit3"]].copy()
    if len(htr) >= MIN_TRAIN and not hte.empty and htr.hit3.nunique() > 1:
        clf["prob"] = fit_predict_classifier(htr, hte, job, primary)
    else:
        clf["prob"] = np.nan
    return reg, clf


def score_current(d, cur, job, primary, secondary):
    tr = d[d.primary_ppg.notna()].copy()
    pred = fit_predict_regression(tr, cur, job, primary, secondary)
    prob = fit_predict_classifier(d[d.hit3.notna()].copy(), cur, job, primary)
    out = cur[["position", "pfr_name"]].copy()
    out["pred_best2of3_ppg"] = pred
    out["hit_probability"] = prob
    return out


def metric_row(position, variant, split, reg, clf, removed_primary, removed_secondary):
    rm = regression_metrics(reg.primary_ppg, reg.pred) if len(reg) else regression_metrics([], [])
    cm = classifier_metrics(clf.hit3, clf.prob) if len(clf) else classifier_metrics([], [])
    return {
        "position": position,
        "variant": variant,
        "split": split,
        "removed_primary_scout_features": removed_primary,
        "removed_secondary_scout_features": removed_secondary,
        **{f"reg_{k}": v for k, v in rm.items()},
        **{f"hit_{k}": v for k, v in cm.items()},
    }


def main():
    hist = pd.read_csv(HIST_POOL, low_memory=False)
    cur = pd.read_csv(CUR_POOL, low_memory=False)
    current_rank = pd.read_csv(CURRENT_RANK, low_memory=False)
    hist["season"] = pd.to_numeric(hist.season, errors="coerce")
    cur["season"] = pd.to_numeric(cur.season, errors="coerce")
    if "target_valid" in hist.columns:
        hist = hist[pd.to_numeric(hist.target_valid, errors="coerce").eq(1)].copy()
    hist = hist[hist.season.le(2023)].copy()
    cur = cur[cur.season.eq(2026)].copy()

    metrics = []
    pred_rows = []
    reproduction_rows = []
    feature_rows = []

    for pos in POSITIONS:
        job = joblib.load(MODELS / f"{pos}.joblib")
        all_features = list(dict.fromkeys(list(job.get("features", [])) + list(job.get("features_b") or [])))
        d = add_missing(hist[hist.position.eq(pos)].copy().reset_index(drop=True), all_features)
        c = add_missing(cur[cur.position.eq(pos)].copy().reset_index(drop=True), all_features)

        for variant, remove_scout in [("frozen_baseline", False), ("no_scout", True)]:
            primary, secondary = variant_features(job, remove_scout)
            removed_primary = sum(f in job.get("features", []) for f in SCOUT_FEATURES) if remove_scout else 0
            removed_secondary = sum(f in (job.get("features_b") or []) for f in SCOUT_FEATURES) if remove_scout else 0
            feature_rows.append({
                "position": pos,
                "variant": variant,
                "primary_feature_count": len(primary),
                "secondary_feature_count": len(secondary),
                "removed_primary_scout_features": removed_primary,
                "removed_secondary_scout_features": removed_secondary,
                "scout_features_in_frozen_primary": "; ".join(f for f in SCOUT_FEATURES if f in job.get("features", [])),
                "scout_features_in_frozen_secondary": "; ".join(f for f in SCOUT_FEATURES if f in (job.get("features_b") or [])),
            })

            val_reg, val_clf = walk_forward(d, job, primary, secondary)
            final_reg, final_clf = final_2023(d, job, primary, secondary)
            metrics.append(metric_row(pos, variant, "validation_2019_2022", val_reg, val_clf, removed_primary, removed_secondary))
            metrics.append(metric_row(pos, variant, "final_2023", final_reg, final_clf, removed_primary, removed_secondary))

            current = score_current(d, c, job, primary, secondary)
            current["variant"] = variant
            pred_rows.append(current)

        # Deterministic reproduction check: refit the frozen architecture through 2023 and compare to published 2026 values.
        base = pd.concat(pred_rows[-2:-1], ignore_index=True)
        published = current_rank[current_rank.position.eq(pos)][["position", "pfr_name", "pred_best2of3_ppg", "hit_probability"]].copy()
        chk = base.merge(published, on=["position", "pfr_name"], suffixes=("_refit", "_published"))
        max_score_err = float((chk.pred_best2of3_ppg_refit - chk.pred_best2of3_ppg_published).abs().max()) if len(chk) else np.nan
        max_prob_err = float((chk.hit_probability_refit - chk.hit_probability_published).abs().max()) if len(chk) else np.nan
        reproduction_rows.append({
            "position": pos,
            "n": len(chk),
            "max_abs_score_reproduction_error": max_score_err,
            "max_abs_hit_probability_reproduction_error": max_prob_err,
            "score_reproduced": int(np.isfinite(max_score_err) and max_score_err <= 1e-6),
            "hit_probability_reproduced": int(np.isfinite(max_prob_err) and max_prob_err <= 1e-6),
        })

    metric_df = pd.DataFrame(metrics)
    predictions = pd.concat(pred_rows, ignore_index=True)
    reproduction = pd.DataFrame(reproduction_rows)
    feature_audit = pd.DataFrame(feature_rows)

    if not reproduction.score_reproduced.eq(1).all() or not reproduction.hit_probability_reproduced.eq(1).all():
        raise RuntimeError("Stage 6A baseline refit does not reproduce frozen published 2026 scores")

    # Pair baseline/no-scout metrics to create an interpretable delta table. Positive Spearman/AUC deltas favor no-scout;
    # negative MAE/RMSE/Brier deltas favor no-scout.
    b = metric_df[metric_df.variant.eq("frozen_baseline")].drop(columns="variant")
    a = metric_df[metric_df.variant.eq("no_scout")].drop(columns="variant")
    paired = a.merge(b, on=["position", "split"], suffixes=("_no_scout", "_baseline"))
    deltas = []
    for _, r in paired.iterrows():
        deltas.append({
            "position": r.position,
            "split": r.split,
            "primary_scout_features_removed": int(r.removed_primary_scout_features_no_scout),
            "secondary_scout_features_removed": int(r.removed_secondary_scout_features_no_scout),
            "delta_mae_no_scout_minus_baseline": r.reg_mae_no_scout - r.reg_mae_baseline,
            "delta_rmse_no_scout_minus_baseline": r.reg_rmse_no_scout - r.reg_rmse_baseline,
            "delta_spearman_no_scout_minus_baseline": r.reg_spearman_no_scout - r.reg_spearman_baseline,
            "delta_pearson_no_scout_minus_baseline": r.reg_pearson_no_scout - r.reg_pearson_baseline,
            "delta_hit_auc_no_scout_minus_baseline": r.hit_auc_no_scout - r.hit_auc_baseline,
            "delta_hit_brier_no_scout_minus_baseline": r.hit_brier_no_scout - r.hit_brier_baseline,
        })
    delta_df = pd.DataFrame(deltas)

    # 2026 player-level impact.
    pbase = predictions[predictions.variant.eq("frozen_baseline")].drop(columns="variant")
    pabl = predictions[predictions.variant.eq("no_scout")].drop(columns="variant")
    impact = pabl.merge(pbase, on=["position", "pfr_name"], suffixes=("_no_scout", "_baseline"))
    impact["score_delta_no_scout_minus_baseline"] = impact.pred_best2of3_ppg_no_scout - impact.pred_best2of3_ppg_baseline
    impact["hit_probability_delta_pp_no_scout_minus_baseline"] = 100.0 * (impact.hit_probability_no_scout - impact.hit_probability_baseline)
    impact = impact.sort_values(["position", "score_delta_no_scout_minus_baseline"], key=lambda s: s.abs() if "delta" in s.name else s, ascending=[True, False], kind="stable")

    metric_df.to_csv(OUT / "scout_ablation_metrics.csv", index=False)
    delta_df.to_csv(OUT / "scout_ablation_deltas.csv", index=False)
    predictions.to_csv(OUT / "scout_ablation_2026_predictions.csv", index=False)
    impact.to_csv(OUT / "scout_ablation_2026_impact.csv", index=False)
    reproduction.to_csv(OUT / "baseline_reproduction_audit.csv", index=False)
    feature_audit.to_csv(OUT / "scout_ablation_feature_audit.csv", index=False)

    report = [
        "# Rookie Models Stage 6A: Scouting-Surprise Ablation\n\n",
        f"Generated {datetime.now(timezone.utc).isoformat()}\n\n",
        "Stage 6A is the first model-enhancement experiment. It holds the frozen Stage 3 architecture, algorithms, blend weights, targets, and train/test windows fixed and removes only `scout_expected_log_pick` and `scout_boost`. This tests the predictive value of their implicit inclusion in broad WR/TE feature sets. The scouting-surprise fields are pre-NFL and leakage-safe with respect to NFL outcomes; this is a taxonomy/robustness ablation, not an outcome-leakage correction.\n\n",
        "## Baseline reproduction audit\n\n",
        "|Pos|N|Score max error|Hit-prob max error|Pass?|\n|---|---:|---:|---:|---|\n",
    ]
    for _, r in reproduction.iterrows():
        ok = "Yes" if r.score_reproduced == 1 and r.hit_probability_reproduced == 1 else "No"
        report.append(f"|{r.position}|{int(r.n)}|{r.max_abs_score_reproduction_error:.2e}|{r.max_abs_hit_probability_reproduction_error:.2e}|{ok}|\n")

    report.extend(["\n## Ablation deltas\n\n", "Negative MAE/RMSE/Brier deltas and positive Spearman/Pearson/AUC deltas favor removing the scout fields.\n\n",
                   "|Pos|Split|Removed P/S|ΔMAE|ΔRMSE|ΔSpearman|ΔPearson|ΔHit AUC|ΔHit Brier|\n|---|---|---|---:|---:|---:|---:|---:|---:|\n"])
    for _, r in delta_df.iterrows():
        report.append(
            f"|{r.position}|{r.split}|{int(r.primary_scout_features_removed)}/{int(r.secondary_scout_features_removed)}|"
            f"{r.delta_mae_no_scout_minus_baseline:+.3f}|{r.delta_rmse_no_scout_minus_baseline:+.3f}|"
            f"{r.delta_spearman_no_scout_minus_baseline:+.3f}|{r.delta_pearson_no_scout_minus_baseline:+.3f}|"
            f"{r.delta_hit_auc_no_scout_minus_baseline:+.3f}|{r.delta_hit_brier_no_scout_minus_baseline:+.3f}|\n"
        )

    report.append("\n## Largest 2026 score changes\n\n")
    changed = impact[impact.score_delta_no_scout_minus_baseline.abs() > 1e-9].copy()
    if changed.empty:
        report.append("No 2026 score changes.\n")
    else:
        report.append("|Pos|Player|Baseline PPG|No-scout PPG|ΔPPG|Baseline hit|No-scout hit|Δ hit pp|\n|---|---|---:|---:|---:|---:|---:|---:|\n")
        changed["abs_delta"] = changed.score_delta_no_scout_minus_baseline.abs()
        for _, r in changed.sort_values("abs_delta", ascending=False).head(25).iterrows():
            report.append(
                f"|{r.position}|{r.pfr_name}|{r.pred_best2of3_ppg_baseline:.2f}|{r.pred_best2of3_ppg_no_scout:.2f}|"
                f"{r.score_delta_no_scout_minus_baseline:+.2f}|{100*r.hit_probability_baseline:.1f}%|{100*r.hit_probability_no_scout:.1f}%|"
                f"{r.hit_probability_delta_pp_no_scout_minus_baseline:+.1f}|\n"
            )

    report.extend([
        "\n## Decision rule\n\n",
        "This stage does not automatically replace the frozen model. A scout-free variant should only advance if pre-2023 validation improves or remains essentially neutral without creating a meaningful 2023 degradation. If the scout fields help validation, they remain legitimate candidate predictors but will be isolated explicitly in the Stage 6 feature taxonomy rather than entering generic production sets accidentally.\n",
    ])
    (OUT / "REPORT.md").write_text("".join(report))

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "6A",
        "experiment": "scouting-surprise ablation",
        "baseline_model_changed": False,
        "experiment_changes": "Remove only scout_expected_log_pick and scout_boost while holding frozen architecture/model types/blend weights/train windows fixed.",
        "validation_years": VALID_YEARS,
        "final_test_year": FINAL_TEST_YEAR,
        "important_note": "Scout features are pre-NFL and leakage-safe with respect to NFL outcomes. This tests robustness and explicit feature taxonomy, not outcome leakage.",
        "outputs": [
            "scout_ablation_metrics.csv", "scout_ablation_deltas.csv", "scout_ablation_2026_predictions.csv",
            "scout_ablation_2026_impact.csv", "baseline_reproduction_audit.csv", "scout_ablation_feature_audit.csv", "REPORT.md"
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
