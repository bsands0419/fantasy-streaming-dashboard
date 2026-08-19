from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import stage3 as s3

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_2023_2026"
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ["QB", "RB", "WR", "TE"]
YEARS = [2023, 2024, 2025, 2026]


def pct(hist: pd.Series, vals: pd.Series) -> np.ndarray:
    h = pd.to_numeric(hist, errors="coerce").dropna().to_numpy(float)
    v = pd.to_numeric(vals, errors="coerce").to_numpy(float)
    out = np.full(len(v), np.nan)
    if not len(h):
        return out
    for i, x in enumerate(v):
        if np.isfinite(x):
            out[i] = 100.0 * np.mean(h <= x)
    return out


def predict_frozen(cur: pd.DataFrame, job: dict) -> pd.DataFrame:
    q = cur.copy()
    a, bb = job["fitted"]["primary_ppg"]
    q["prospect_model_score"] = s3.predict_pair(a, bb, q, job)
    q["hit_probability"] = job["classifier"].predict_proba(q[job["features"]])[:, 1]
    return q


def main() -> None:
    pool = pd.read_csv(ROOT / "results_2026" / "prospect_pool_2026.csv")
    hist = pd.read_csv(ROOT / "results_2026" / "historical_prospect_oof.csv")
    pool["season"] = pd.to_numeric(pool["season"], errors="coerce")
    pool["draft_pick"] = pd.to_numeric(pool["draft_pick"], errors="coerce")
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")

    rows = []
    for pos in POSITIONS:
        ref = hist.loc[hist.position.eq(pos), "prospect_model_score"]
        h23 = hist[(hist.position.eq(pos)) & (hist.season.eq(2023))].copy()
        meta_cols = [c for c in ["season", "position", "pfr_name", "draft_team", "draft_round", "draft_pick", "college_match", "fuzzy_match", "scout_boost"] if c in pool.columns]
        meta = pool[pool.season.eq(2023) & pool.position.eq(pos)][meta_cols].copy()
        if not h23.empty:
            q23 = h23.merge(meta, on=["season", "position", "pfr_name"], how="left")
            q23["hit_probability"] = np.nan
            q23["primary_ppg_realized"] = q23.get("primary_ppg")
            rows.append(q23)

        job = joblib.load(ROOT / "trained_v3" / f"{pos}.joblib")
        for year in [2024, 2025, 2026]:
            cur = pool[(pool.position.eq(pos)) & (pool.season.eq(year))].copy()
            if cur.empty:
                continue
            cur = predict_frozen(cur, job)
            cur["primary_ppg_realized"] = pd.to_numeric(cur.get("primary_ppg"), errors="coerce")
            rows.append(cur)

    z = pd.concat(rows, ignore_index=True, sort=False)
    z = z[z.season.isin(YEARS) & z.position.isin(POSITIONS)].copy()

    z["prospect_model_percentile"] = np.nan
    for pos in POSITIONS:
        m = z.position.eq(pos)
        ref = hist.loc[hist.position.eq(pos), "prospect_model_score"]
        z.loc[m, "prospect_model_percentile"] = pct(ref, z.loc[m, "prospect_model_score"])

    z["position_year_rank"] = z.groupby(["season", "position"])["prospect_model_score"].rank(ascending=False, method="first").astype(int)
    z["position_2023_2026_rank"] = z.groupby("position")["prospect_model_percentile"].rank(ascending=False, method="first").astype(int)
    z["overall_2023_2026_rank"] = z["prospect_model_percentile"].rank(ascending=False, method="first").astype(int)

    keep = [
        "season", "position", "position_year_rank", "position_2023_2026_rank", "overall_2023_2026_rank",
        "pfr_name", "draft_team", "draft_round", "draft_pick",
        "prospect_model_score", "prospect_model_percentile", "hit_probability",
        "primary_ppg_realized", "college_match", "fuzzy_match", "scout_boost"
    ]
    z = z[[c for c in keep if c in z.columns]]

    z.sort_values(["season", "position", "position_year_rank"]).to_csv(OUT / "full_results_2023_2026.csv", index=False)
    z.sort_values(["position", "position_2023_2026_rank"]).to_csv(OUT / "all_years_ranked_by_position.csv", index=False)
    z.sort_values("overall_2023_2026_rank").to_csv(OUT / "all_positions_all_years_ranked.csv", index=False)

    for year in YEARS:
        z[z.season.eq(year)].sort_values(["position", "position_year_rank"]).to_csv(OUT / f"rookie_results_{year}.csv", index=False)
    for pos in POSITIONS:
        z[z.position.eq(pos)].sort_values("position_2023_2026_rank").to_csv(OUT / f"{pos}_2023_2026_ranked.csv", index=False)

    print(z.groupby(["season", "position"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
