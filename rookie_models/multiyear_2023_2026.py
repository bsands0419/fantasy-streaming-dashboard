from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import modeling as b
import stage2 as s2
import stage3 as s3
import score_2026 as s26

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


def current_recent_draft() -> pd.DataFrame:
    d = b.read_csv_url(s26.CURRENT_DRAFT_URL).copy()
    if "season" not in d.columns:
        for c in ("draft_year", "year"):
            if c in d.columns:
                d["season"] = d[c]
                break
    if "category" not in d.columns:
        for c in ("position", "pos"):
            if c in d.columns:
                d["category"] = d[c]
                break
    if "pfr_name" not in d.columns:
        for c in ("pfr_player_name", "player_name", "name"):
            if c in d.columns:
                d["pfr_name"] = d[c]
                break
    if "pfr_id" not in d.columns:
        for c in ("pfr_player_id", "player_id"):
            if c in d.columns:
                d["pfr_id"] = d[c]
                break

    required = ["season", "category", "pfr_name", "round", "pick", "team"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"Current draft release missing required columns: {missing}")

    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["round"] = pd.to_numeric(d["round"], errors="coerce")
    d["pick"] = pd.to_numeric(d["pick"], errors="coerce")
    d["category"] = d["category"].astype(str).str.upper().str.strip()
    return d[d["season"].isin([2025, 2026]) & d["category"].isin(POSITIONS)].copy()


def rebuild_recent_pool() -> pd.DataFrame:
    hist_pool = pd.read_csv(ROOT / "results_v4" / "prospect_pool_v4.csv")
    hist_pool["season"] = pd.to_numeric(hist_pool["season"], errors="coerce")
    hist_pool = hist_pool[hist_pool.season.le(2023)].copy()
    hist_pool = hist_pool.drop(columns=["scout_expected_log_pick", "scout_boost"], errors="ignore")

    print("Loading college data for 2024-2026 classes...")
    pa = b.load_cfb_table("passing")
    ru = b.load_cfb_table("rushing")
    re = b.load_cfb_table("receiving")
    team = b.load_cfb_table("team_summaries")
    pa, ru, re = s2.prep_college(pa, ru, re, team)

    print("Loading nflverse draft/meta/context data...")
    draft, players, combine, weekly = b.load_nflverse()
    draft["season"] = pd.to_numeric(draft["season"], errors="coerce")

    # Historical source contains 2024 but currently lags 2025 and 2026.
    # Replace both recent years with the current nflverse draft release.
    current = current_recent_draft()
    draft = draft[~draft["season"].isin([2025, 2026])].copy()
    draft = pd.concat([draft, current], ignore_index=True, sort=False)
    draft["season"] = pd.to_numeric(draft["season"], errors="coerce")

    recent_draft = draft[
        draft.category.isin(POSITIONS)
        & draft.season.isin([2024, 2025, 2026])
    ].copy()

    for year in [2024, 2025, 2026]:
        if recent_draft[recent_draft.season.eq(year)].empty:
            raise RuntimeError(f"No {year} draft rows in combined current draft source")

    recent = s2.build_profiles(recent_draft, pa, ru, re)
    recent = s2.add_nfl_meta(recent, players, combine)
    recent = s26.add_landing_features_by_pick(recent, draft, weekly)

    combined = pd.concat([hist_pool, recent], ignore_index=True, sort=False)
    combined = s3.add_scouting_surprise(combined)
    return combined[combined.season.isin([2023, 2024, 2025, 2026])].copy()


def main() -> None:
    hist = pd.read_csv(ROOT / "results_2026" / "historical_prospect_oof.csv")
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    pool = rebuild_recent_pool()
    pool["season"] = pd.to_numeric(pool["season"], errors="coerce")
    pool["draft_pick"] = pd.to_numeric(pool["draft_pick"], errors="coerce")

    rows = []
    for pos in POSITIONS:
        h23 = hist[(hist.position.eq(pos)) & (hist.season.eq(2023))].copy()
        meta_cols = [c for c in ["season", "position", "pfr_name", "draft_team", "draft_round", "draft_pick", "college_match", "fuzzy_match", "scout_boost"] if c in pool.columns]
        meta = pool[pool.season.eq(2023) & pool.position.eq(pos)][meta_cols].copy()
        if not h23.empty:
            q23 = h23.merge(meta, on=["season", "position", "pfr_name"], how="left")
            q23["hit_probability"] = np.nan
            q23["primary_ppg_realized"] = q23.get("primary_ppg")
            rows.append(q23)

        job = joblib.load(ROOT / "trained_v3" / f"{pos}.joblib")
        missing = [c for c in job["features"] if c not in pool.columns]
        missing_b = [c for c in (job.get("features_b") or []) if c not in pool.columns]
        if missing or missing_b:
            raise RuntimeError(f"{pos} missing frozen features: primary={missing}, blend={missing_b}")

        for year in [2024, 2025, 2026]:
            cur = pool[(pool.position.eq(pos)) & (pool.season.eq(year))].copy()
            if cur.empty:
                raise RuntimeError(f"No {year} {pos} rows after recent-class rebuild")
            cur = predict_frozen(cur, job)
            cur["primary_ppg_realized"] = np.nan
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

    counts = z.groupby(["season", "position"]).size().unstack(fill_value=0)
    counts.to_csv(OUT / "class_counts.csv")
    print(counts)


if __name__ == "__main__":
    main()
