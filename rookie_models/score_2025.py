from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import modeling as b
import stage2 as s2
import stage3 as s3
import score_2026 as s26

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_2025"
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ["QB", "RB", "WR", "TE"]
PREDICT_YEAR = 2025
CURRENT_DRAFT_URL = s26.CURRENT_DRAFT_URL
CFB_YEARS = range(2019, 2025)


def load_current_draft_year(year: int) -> pd.DataFrame:
    d = b.read_csv_url(CURRENT_DRAFT_URL).copy()
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
        raise RuntimeError(f"Current nflverse draft release missing required columns: {missing}")

    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["round"] = pd.to_numeric(d["round"], errors="coerce")
    d["pick"] = pd.to_numeric(d["pick"], errors="coerce")
    d["category"] = d["category"].astype(str).str.upper().str.strip()
    out = d[d["season"].eq(year) & d["category"].isin(POSITIONS)].copy()
    if out.empty:
        raise RuntimeError(f"Current nflverse draft release has no {year} QB/RB/WR/TE rows")
    return out


def load_cfb_table_limited(kind: str) -> pd.DataFrame:
    tag = f"espn_cfb_{kind}"
    assets = b.release_assets(tag)
    frames = []
    for year in CFB_YEARS:
        names = [f"cfb_{kind}_{year}.csv", f"{kind}_{year}.csv"]
        name = next((n for n in names if n in assets), None)
        if not name:
            continue
        fp = b.CACHE / name
        if not fp.exists():
            fp.write_bytes(b.get(assets[name], timeout=180).content)
        d = pd.read_csv(fp, low_memory=False)
        d["season"] = pd.to_numeric(d.get("season", year), errors="coerce").fillna(year).astype(int)
        frames.append(d)
    if not frames:
        raise RuntimeError(f"No CFB {kind} data for 2019-2024")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_nfl_context_2025(current: pd.DataFrame):
    players = b.read_csv_url("https://github.com/nflverse/nflverse-data/releases/download/players/players.csv")
    combine = b.read_csv_url("https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv")
    weekly = b.read_csv_url("https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2024.csv")
    weekly["season"] = 2024
    draft = b.read_csv_url(CURRENT_DRAFT_URL)
    if "season" not in draft.columns:
        for c in ("draft_year", "year"):
            if c in draft.columns:
                draft["season"] = draft[c]
                break
    draft["season"] = pd.to_numeric(draft["season"], errors="coerce")
    return draft, players, combine, weekly


def percentile(hist: pd.Series, vals: pd.Series) -> np.ndarray:
    h = pd.to_numeric(hist, errors="coerce").dropna().to_numpy(float)
    v = pd.to_numeric(vals, errors="coerce").to_numpy(float)
    out = np.full(len(v), np.nan)
    if not len(h):
        return out
    for i, x in enumerate(v):
        if np.isfinite(x):
            out[i] = 100.0 * np.mean(h <= x)
    return out


def predict_year(pool: pd.DataFrame, jobs: dict, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pos in POSITIONS:
        cur = pool[(pool["season"].eq(PREDICT_YEAR)) & (pool["position"].eq(pos))].copy()
        if cur.empty:
            raise RuntimeError(f"No {PREDICT_YEAR} {pos} rows after feature engineering")
        job = jobs[pos]
        missing = [c for c in job["features"] if c not in cur.columns]
        missing_b = [c for c in (job.get("features_b") or []) if c not in cur.columns]
        if missing or missing_b:
            raise RuntimeError(f"{pos} missing frozen features: primary={missing}, blend={missing_b}")

        a, bb = job["fitted"]["primary_ppg"]
        cur["prospect_model_score"] = s3.predict_pair(a, bb, cur, job)
        cur["hit_probability"] = job["classifier"].predict_proba(cur[job["features"]])[:, 1]
        ref = oof.loc[oof.position.eq(pos), "prospect_model_score"]
        cur["prospect_model_percentile"] = percentile(ref, cur["prospect_model_score"])
        cur["position_year_rank"] = cur["prospect_model_score"].rank(ascending=False, method="first").astype(int)
        rows.append(cur)

    z = pd.concat(rows, ignore_index=True, sort=False)
    keep = [
        "season", "position", "position_year_rank", "pfr_name", "draft_team", "draft_round", "draft_pick",
        "prospect_model_score", "prospect_model_percentile", "hit_probability",
        "college_match", "fuzzy_match", "scout_boost"
    ]
    return z[[c for c in keep if c in z.columns]].sort_values(["position", "position_year_rank"])


def main() -> None:
    print("Loading current 2025 nflverse draft class...")
    current = load_current_draft_year(PREDICT_YEAR)
    print(f"Found {len(current)} drafted 2025 QB/RB/WR/TE prospects")

    print("Loading only the six pre-draft CFB seasons used by the frozen feature builder...")
    pa = load_cfb_table_limited("passing")
    ru = load_cfb_table_limited("rushing")
    re = load_cfb_table_limited("receiving")
    team = load_cfb_table_limited("team_summaries")
    pa, ru, re = s2.prep_college(pa, ru, re, team)

    print("Loading 2025 nflverse player/combine data and 2024 landing context...")
    draft, players, combine, weekly = load_nfl_context_2025(current)
    recent = s2.build_profiles(current, pa, ru, re)
    recent = s2.add_nfl_meta(recent, players, combine)
    recent = s26.add_landing_features_by_pick(recent, draft, weekly)

    print("Loading validated historical pool for leakage-safe scouting surprise feature...")
    hist_pool = pd.read_csv(ROOT / "results_v4" / "prospect_pool_v4.csv")
    hist_pool["season"] = pd.to_numeric(hist_pool["season"], errors="coerce")
    hist_pool = hist_pool[hist_pool["season"].lt(PREDICT_YEAR)].copy()
    hist_pool = hist_pool.drop(columns=["scout_expected_log_pick", "scout_boost"], errors="ignore")
    recent = recent.drop(columns=["scout_expected_log_pick", "scout_boost"], errors="ignore")
    pool = pd.concat([hist_pool, recent], ignore_index=True, sort=False)
    pool = s3.add_scouting_surprise(pool)

    jobs = {}
    for pos in POSITIONS:
        path = ROOT / "trained_v3" / f"{pos}.joblib"
        if not path.exists():
            raise RuntimeError(f"Missing frozen Stage 3 model: {path}")
        jobs[pos] = joblib.load(path)

    oof = pd.read_csv(ROOT / "results_2026" / "historical_prospect_oof.csv")
    rankings = predict_year(pool, jobs, oof)
    rankings.to_csv(OUT / "rookie_rankings_2025.csv", index=False)
    pool[pool["season"].eq(PREDICT_YEAR)].to_csv(OUT / "prospect_pool_2025.csv", index=False)

    counts = rankings.groupby("position").size().to_dict()
    hist_max_year = int(pd.to_numeric(hist_pool["season"], errors="coerce").max()) if len(hist_pool) else None
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_year": PREDICT_YEAR,
        "model_source": "frozen Stage 3 trained_v3 models",
        "model_retrained": False,
        "current_draft_source": CURRENT_DRAFT_URL,
        "cfb_seasons_loaded": list(CFB_YEARS),
        "nfl_weekly_context_season": 2024,
        "historical_scout_reference_last_year": hist_max_year,
        "ranked_rows": int(len(rankings)),
        "counts_by_position": {p: int(counts.get(p, 0)) for p in POSITIONS},
        "percentile_reference": "same-position leakage-safe historical frozen-architecture OOF prospect scores from results_2026/historical_prospect_oof.csv",
        "note": "Score-only historical class reconstruction. No model selection, target, feature-selection, or audit logic changed."
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
