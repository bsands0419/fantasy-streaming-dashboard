from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
IN2026 = ROOT / "results_2026"
INV4 = ROOT / "results_v4"
OUT = ROOT / "results_stage5"
OUT.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
BUCKET_EDGES = [-1e-9, 20, 40, 60, 80, 90, 95, 100.000001]
BUCKET_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-90", "90-95", "95-100"]
THRESHOLDS = [50, 60, 70, 80, 90, 95]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def percentile_against(values: pd.Series, x: float) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(v) or not np.isfinite(x):
        return np.nan
    return 100.0 * float(np.mean(v <= x))


def add_percentiles(oof: pd.DataFrame) -> pd.DataFrame:
    x = oof.copy()
    x["prospect_model_percentile"] = np.nan
    x["temporal_prospect_percentile"] = np.nan
    x["temporal_reference_n"] = 0

    for pos in POSITIONS:
        idx = x.index[x.position.eq(pos)]
        vals = pd.to_numeric(x.loc[idx, "prospect_model_score"], errors="coerce")
        x.loc[idx, "prospect_model_percentile"] = vals.rank(method="average", pct=True).to_numpy() * 100.0
        for i in idx:
            yr = int(x.at[i, "season"])
            hist = x.loc[(x.position.eq(pos)) & (x.season < yr), "prospect_model_score"]
            x.at[i, "temporal_reference_n"] = int(pd.to_numeric(hist, errors="coerce").notna().sum())
            if x.at[i, "temporal_reference_n"] >= 30:
                x.at[i, "temporal_prospect_percentile"] = percentile_against(hist, float(x.at[i, "prospect_model_score"]))
    return x


def calibration_rows(x: pd.DataFrame, percentile_col: str, calibration_type: str) -> pd.DataFrame:
    z = x[x[percentile_col].notna()].copy()
    z["percentile_bucket"] = pd.cut(
        z[percentile_col], bins=BUCKET_EDGES, labels=BUCKET_LABELS, include_lowest=True, right=True
    )
    rows = []
    for (pos, bucket), g in z.groupby(["position", "percentile_bucket"], observed=True):
        n = len(g)
        hits = int(g.hit3.sum())
        stars = int(g.star3.sum())
        hlo, hhi = wilson(hits, n)
        slo, shi = wilson(stars, n)
        rows.append({
            "calibration_type": calibration_type,
            "position": pos,
            "percentile_bucket": str(bucket),
            "n": n,
            "hit_rate": hits / n if n else np.nan,
            "hit_rate_ci_low": hlo,
            "hit_rate_ci_high": hhi,
            "star_rate": stars / n if n else np.nan,
            "star_rate_ci_low": slo,
            "star_rate_ci_high": shi,
            "mean_primary_ppg": float(g.primary_ppg.mean()),
            "median_primary_ppg": float(g.primary_ppg.median()),
            "mean_prospect_score": float(g.prospect_model_score.mean()),
        })
    return pd.DataFrame(rows)


def threshold_rows(x: pd.DataFrame, percentile_col: str, calibration_type: str) -> pd.DataFrame:
    rows = []
    for pos in POSITIONS:
        d = x[(x.position.eq(pos)) & x[percentile_col].notna()].copy()
        for t in THRESHOLDS:
            g = d[d[percentile_col] >= t]
            n = len(g)
            hits = int(g.hit3.sum()) if n else 0
            stars = int(g.star3.sum()) if n else 0
            hlo, hhi = wilson(hits, n)
            slo, shi = wilson(stars, n)
            rows.append({
                "calibration_type": calibration_type,
                "position": pos,
                "minimum_percentile": t,
                "n": n,
                "hit_rate": hits / n if n else np.nan,
                "hit_rate_ci_low": hlo,
                "hit_rate_ci_high": hhi,
                "star_rate": stars / n if n else np.nan,
                "star_rate_ci_low": slo,
                "star_rate_ci_high": shi,
                "mean_primary_ppg": float(g.primary_ppg.mean()) if n else np.nan,
            })
    return pd.DataFrame(rows)


def summary_rows(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pos in POSITIONS:
        d = x[x.position.eq(pos)].dropna(subset=["prospect_model_percentile", "primary_ppg"])
        if d.empty:
            continue
        sp = spearmanr(d.prospect_model_percentile, d.primary_ppg).statistic if len(d) >= 4 else np.nan
        hit_auc = roc_auc_score(d.hit3, d.prospect_model_percentile) if d.hit3.nunique() > 1 else np.nan
        star_auc = roc_auc_score(d.star3, d.prospect_model_percentile) if d.star3.nunique() > 1 else np.nan
        td = d.dropna(subset=["temporal_prospect_percentile"])
        tsp = spearmanr(td.temporal_prospect_percentile, td.primary_ppg).statistic if len(td) >= 4 else np.nan
        thit = roc_auc_score(td.hit3, td.temporal_prospect_percentile) if len(td) and td.hit3.nunique() > 1 else np.nan
        tstar = roc_auc_score(td.star3, td.temporal_prospect_percentile) if len(td) and td.star3.nunique() > 1 else np.nan
        rows.append({
            "position": pos,
            "oof_n": len(d),
            "first_oof_year": int(d.season.min()),
            "last_oof_year": int(d.season.max()),
            "spearman_percentile_vs_primary_ppg": sp,
            "hit_auc": hit_auc,
            "star_auc": star_auc,
            "temporal_n": len(td),
            "temporal_spearman_vs_primary_ppg": tsp,
            "temporal_hit_auc": thit,
            "temporal_star_auc": tstar,
        })
    return pd.DataFrame(rows)


def attach_2026_calibration(rankings: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    r = rankings.copy()
    r["percentile_bucket"] = pd.cut(
        pd.to_numeric(r.prospect_model_percentile, errors="coerce"),
        bins=BUCKET_EDGES, labels=BUCKET_LABELS, include_lowest=True, right=True,
    ).astype(str)
    lookup = pooled[pooled.calibration_type.eq("pooled_reference")][
        ["position", "percentile_bucket", "n", "hit_rate", "hit_rate_ci_low", "hit_rate_ci_high",
         "star_rate", "star_rate_ci_low", "star_rate_ci_high", "mean_primary_ppg"]
    ].rename(columns={
        "n": "historical_bucket_n",
        "hit_rate": "historical_hit_rate",
        "hit_rate_ci_low": "historical_hit_rate_ci_low",
        "hit_rate_ci_high": "historical_hit_rate_ci_high",
        "star_rate": "historical_star_rate",
        "star_rate_ci_low": "historical_star_rate_ci_low",
        "star_rate_ci_high": "historical_star_rate_ci_high",
        "mean_primary_ppg": "historical_bucket_mean_primary_ppg",
    })
    return r.merge(lookup, on=["position", "percentile_bucket"], how="left", validate="many_to_one")


def main() -> None:
    oof = pd.read_csv(IN2026 / "historical_prospect_oof.csv")
    pool = pd.read_csv(INV4 / "prospect_pool_v4.csv", low_memory=False)
    rankings = pd.read_csv(IN2026 / "rookie_rankings_2026.csv")

    labels = pool[[c for c in ["season", "pfr_name", "position", "hit3", "star3", "best_rank3", "target_valid"] if c in pool.columns]].copy()
    labels = labels.drop_duplicates(["season", "pfr_name", "position"])
    x = oof.merge(labels, on=["season", "pfr_name", "position"], how="left", validate="one_to_one")
    for c in ["hit3", "star3"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[x.hit3.notna() & x.star3.notna()].copy()
    x["hit3"] = x.hit3.astype(int)
    x["star3"] = x.star3.astype(int)
    x = add_percentiles(x)

    pooled = calibration_rows(x, "prospect_model_percentile", "pooled_reference")
    temporal = calibration_rows(x, "temporal_prospect_percentile", "prior_only_reference")
    calibration = pd.concat([pooled, temporal], ignore_index=True)
    thresholds = pd.concat([
        threshold_rows(x, "prospect_model_percentile", "pooled_reference"),
        threshold_rows(x, "temporal_prospect_percentile", "prior_only_reference"),
    ], ignore_index=True)
    summary = summary_rows(x)
    current = attach_2026_calibration(rankings, calibration)

    x.to_csv(OUT / "historical_prospect_calibration_rows.csv", index=False)
    calibration.to_csv(OUT / "percentile_bucket_calibration.csv", index=False)
    thresholds.to_csv(OUT / "percentile_threshold_curves.csv", index=False)
    summary.to_csv(OUT / "calibration_summary.csv", index=False)
    current.to_csv(OUT / "rookie_rankings_2026_calibrated.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "5A",
        "model_changed": False,
        "model_selection_changed": False,
        "inputs": [
            "results_2026/historical_prospect_oof.csv",
            "results_v4/prospect_pool_v4.csv",
            "results_2026/rookie_rankings_2026.csv",
        ],
        "success_definitions": {
            "hit3": "At least one first-three-year league-wide positional finish of QB/RB/TE top 12 or WR top 24.",
            "star3": "At least one first-three-year league-wide positional finish of QB/RB/TE top 6 or WR top 12.",
        },
        "calibration_types": {
            "pooled_reference": "Matches the current 2026 percentile interpretation by ranking each historical OOF score inside the complete same-position OOF reference distribution.",
            "prior_only_reference": "Stricter temporal robustness check: each historical score is ranked only against OOF scores from earlier draft classes, requiring at least 30 earlier reference players.",
        },
        "percentile_buckets": BUCKET_LABELS,
        "cumulative_thresholds": THRESHOLDS,
        "note": "Stage 5A is interpretation/calibration only. Frozen Stage 3 models, Stage 4 audit logic, targets, feature selection, and 2026 predictions are unchanged.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
