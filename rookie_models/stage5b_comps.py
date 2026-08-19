from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CUR_POOL = ROOT / "results_2026" / "prospect_pool_2026.csv"
CAL = ROOT / "results_stage5" / "historical_prospect_calibration_rows.csv"
RANK = ROOT / "results_stage5" / "rookie_rankings_2026_calibrated.csv"
MODELS = ROOT / "trained_v3"
OUT = ROOT / "results_stage5b"
OUT.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
K = 5
MIN_HISTORY = 30
MIN_FEATURE_NONMISS = 20


def ordered_union(a, b):
    out = []
    for x in list(a or []) + list(b or []):
        if x not in out:
            out.append(x)
    return out


def prepare_space(hist: pd.DataFrame, cur: pd.DataFrame, features: list[str]):
    usable = []
    for c in features:
        if c not in hist.columns or c not in cur.columns:
            continue
        h = pd.to_numeric(hist[c], errors="coerce")
        if h.notna().sum() < MIN_FEATURE_NONMISS:
            continue
        sd = float(h.std())
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        usable.append(c)

    if not usable:
        raise RuntimeError("No usable frozen-model features for similarity space")

    hraw = hist[usable].apply(pd.to_numeric, errors="coerce")
    craw = cur[usable].apply(pd.to_numeric, errors="coerce")
    med = hraw.median(axis=0)
    mu = hraw.mean(axis=0)
    sd = hraw.std(axis=0).replace(0, np.nan)

    hz = ((hraw.fillna(med) - mu) / sd).clip(-6, 6).fillna(0.0)
    cz = ((craw.fillna(med) - mu) / sd).clip(-6, 6).fillna(0.0)
    coverage = craw.notna().mean(axis=1)
    return usable, hz.to_numpy(float), cz.to_numpy(float), coverage.to_numpy(float)


def rms_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def historical_density_reference(x: np.ndarray, k: int = K) -> np.ndarray:
    n = len(x)
    if n <= k:
        return np.array([], dtype=float)
    vals = []
    for i in range(n):
        d = rms_dist(x, x[i])
        d[i] = np.inf
        near = np.partition(d, k - 1)[:k]
        vals.append(float(np.mean(near)))
    return np.asarray(vals, dtype=float)


def confidence_from_density(cur_d5: float, hist_d5: np.ndarray, coverage: float):
    if not len(hist_d5) or not np.isfinite(cur_d5):
        return np.nan, "Low", True
    density_pct = 100.0 * float(np.mean(hist_d5 >= cur_d5))
    coverage_pct = 100.0 * float(np.clip(coverage, 0, 1))
    score = 0.70 * density_pct + 0.30 * coverage_pct
    ood = bool(cur_d5 > np.quantile(hist_d5, 0.90) or coverage < 0.75)
    if score >= 70 and not ood:
        label = "High"
    elif score >= 35 and not ood:
        label = "Medium"
    else:
        label = "Low"
    return score, label, ood


def main():
    hist_pool = pd.read_csv(HIST_POOL, low_memory=False)
    cur_pool = pd.read_csv(CUR_POOL, low_memory=False)
    cal = pd.read_csv(CAL)
    rankings = pd.read_csv(RANK)

    hist_pool["season"] = pd.to_numeric(hist_pool["season"], errors="coerce")
    cur_pool["season"] = pd.to_numeric(cur_pool["season"], errors="coerce")
    if "target_valid" in hist_pool.columns:
        hist_pool = hist_pool[pd.to_numeric(hist_pool.target_valid, errors="coerce").eq(1)].copy()
    hist_pool = hist_pool[hist_pool.season <= 2023].copy()
    cur_pool = cur_pool[cur_pool.season.eq(2026)].copy()

    label_cols = [
        "season", "pfr_name", "position", "prospect_model_score", "prospect_model_percentile",
        "primary_ppg", "hit3", "star3", "best_rank3"
    ]
    labels = cal[[c for c in label_cols if c in cal.columns]].drop_duplicates(["season", "pfr_name", "position"])
    hist_pool = hist_pool.merge(labels, on=["season", "pfr_name", "position"], how="left", suffixes=("", "_cal"))

    comp_rows = []
    summary_rows = []
    feature_audit = []

    for pos in POSITIONS:
        job = joblib.load(MODELS / f"{pos}.joblib")
        features = ordered_union(job.get("features", []), job.get("features_b", []))
        h = hist_pool[hist_pool.position.eq(pos)].copy().reset_index(drop=True)
        c = cur_pool[cur_pool.position.eq(pos)].copy().reset_index(drop=True)
        if len(h) < MIN_HISTORY or c.empty:
            continue

        usable, hx, cx, cover = prepare_space(h, c, features)
        h_d5 = historical_density_reference(hx, K)
        feature_audit.append({
            "position": pos,
            "frozen_model_features": len(features),
            "usable_similarity_features": len(usable),
            "historical_rows": len(h),
            "current_rows": len(c),
            "historical_loo_mean_5nn_distance": float(np.mean(h_d5)) if len(h_d5) else np.nan,
            "historical_loo_p90_5nn_distance": float(np.quantile(h_d5, .90)) if len(h_d5) else np.nan,
        })

        for i, r in c.iterrows():
            d = rms_dist(hx, cx[i])
            order = np.argsort(d)[:K]
            d5 = float(np.mean(d[order]))
            conf_score, conf_label, ood = confidence_from_density(d5, h_d5, float(cover[i]))

            neigh = h.iloc[order].copy()
            neigh["distance"] = d[order]
            hit_rate = pd.to_numeric(neigh.get("hit3"), errors="coerce").mean()
            star_rate = pd.to_numeric(neigh.get("star3"), errors="coerce").mean()
            ppg_mean = pd.to_numeric(neigh.get("primary_ppg"), errors="coerce").mean()

            comp_names = []
            for rank_i, (idx, hr) in enumerate(neigh.iterrows(), start=1):
                comp_names.append(f"{hr.pfr_name} ({int(hr.season)})")
                comp_rows.append({
                    "position": pos,
                    "current_player": r.pfr_name,
                    "current_draft_pick": r.get("draft_pick"),
                    "comp_rank": rank_i,
                    "comp_player": hr.pfr_name,
                    "comp_draft_year": int(hr.season),
                    "comp_draft_pick": hr.get("draft_pick"),
                    "feature_space_distance": float(hr.distance),
                    "comp_prospect_model_score": hr.get("prospect_model_score"),
                    "comp_prospect_model_percentile": hr.get("prospect_model_percentile"),
                    "comp_primary_ppg": hr.get("primary_ppg"),
                    "comp_hit3": hr.get("hit3"),
                    "comp_star3": hr.get("star3"),
                    "comp_best_rank3": hr.get("best_rank3"),
                })

            density_pct = 100.0 * float(np.mean(h_d5 >= d5)) if len(h_d5) else np.nan
            summary_rows.append({
                "position": pos,
                "pfr_name": r.pfr_name,
                "feature_coverage": float(cover[i]),
                "mean_5nn_distance": d5,
                "neighbor_density_percentile": density_pct,
                "model_confidence_score": conf_score,
                "model_confidence": conf_label,
                "out_of_distribution_flag": int(ood),
                "nearest5_historical_hit_rate": hit_rate,
                "nearest5_historical_star_rate": star_rate,
                "nearest5_historical_primary_ppg_mean": ppg_mean,
                "closest_historical_comps": "; ".join(comp_names),
                "closest_comp_1": comp_names[0] if len(comp_names) > 0 else "",
                "closest_comp_2": comp_names[1] if len(comp_names) > 1 else "",
                "closest_comp_3": comp_names[2] if len(comp_names) > 2 else "",
                "similarity_feature_count": len(usable),
                "historical_reference_n": len(h),
            })

    comps = pd.DataFrame(comp_rows)
    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(feature_audit)

    enriched = rankings.merge(summary, on=["position", "pfr_name"], how="left", validate="one_to_one")
    if "rank" in enriched.columns:
        enriched = enriched.sort_values(["position", "rank"], kind="stable")

    comps.to_csv(OUT / "historical_player_comps_2026.csv", index=False)
    summary.to_csv(OUT / "model_confidence_2026.csv", index=False)
    audit.to_csv(OUT / "similarity_feature_audit.csv", index=False)
    enriched.to_csv(OUT / "rookie_rankings_2026_stage5b.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "5B",
        "model_changed": False,
        "model_selection_changed": False,
        "comparison_population": "Historical valid same-position drafted prospects through 2023 versus the 2026 class.",
        "similarity_space": "Frozen Stage 3 model input features (primary plus blend feature set), numeric, historical-mean standardized after historical-median imputation.",
        "nearest_neighbors": K,
        "confidence_definition": {
            "neighbor_density_percentile": "Percentile of the current player's mean 5-neighbor distance relative to historical leave-one-out mean 5-neighbor distances; higher means denser/more familiar feature space.",
            "feature_coverage": "Share of usable frozen-model similarity features observed before imputation for the current player.",
            "score": "70% neighbor-density percentile + 30% feature coverage percentage.",
            "high": "Score >=70 and not OOD.",
            "medium": "Score >=35 and not OOD.",
            "low": "All other cases.",
            "ood": "Mean 5-neighbor distance above the historical 90th percentile or feature coverage below 75%.",
        },
        "important_note": "Confidence is familiarity with the historical model-input space, not probability of NFL success. Nearest-neighbor hit/star rates are descriptive five-player comp summaries, not calibrated probabilities.",
        "inputs": [
            "results_v3/prospect_pool_v3.csv",
            "results_2026/prospect_pool_2026.csv",
            "trained_v3/{position}.joblib",
            "results_stage5/historical_prospect_calibration_rows.csv",
            "results_stage5/rookie_rankings_2026_calibrated.csv",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
