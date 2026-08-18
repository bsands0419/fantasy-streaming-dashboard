from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import modeling as b
import stage2 as s2
import stage3 as s3

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_2026"
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ["QB", "RB", "WR", "TE"]
PREDICT_YEAR = 2026
CURRENT_DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"


def load_current_2026_draft() -> pd.DataFrame:
    """Load the current nflverse draft release and normalize the columns Stage 2 expects.

    The historical modeling loader still points at nflverse/nfldata, which currently
    lags the just-completed draft. This score-only path uses nflverse's current
    draft_picks release without changing any frozen training or model logic.
    """
    d = b.read_csv_url(CURRENT_DRAFT_URL)
    d = d.copy()

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
        raise RuntimeError(f"Current nflverse draft release missing required columns: {missing}; columns={list(d.columns)}")

    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["round"] = pd.to_numeric(d["round"], errors="coerce")
    d["pick"] = pd.to_numeric(d["pick"], errors="coerce")
    d["category"] = d["category"].astype(str).str.upper().str.strip()
    d = d[d["season"].eq(PREDICT_YEAR) & d["category"].isin(POSITIONS)].copy()
    if d.empty:
        mx = pd.to_numeric(pd.Series(b.read_csv_url(CURRENT_DRAFT_URL).get("season")), errors="coerce").max()
        raise RuntimeError(f"Current nflverse draft_picks release has no {PREDICT_YEAR} QB/RB/WR/TE rows; max season={mx}")
    return d


def add_landing_features_by_pick(prof: pd.DataFrame, draft: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Attach draft team and prior-year landing context without relying on a PFR id.

    Draft pick is unique within a draft year and is already present in the frozen
    prospect feature set. This fallback matters for a just-completed draft class,
    where PFR identifiers can lag the nflverse draft feed.
    """
    p = prof.copy()
    d = draft[["season", "pick", "team"]].copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["draft_pick"] = pd.to_numeric(d["pick"], errors="coerce")
    d = d.dropna(subset=["season", "draft_pick"]).drop_duplicates(["season", "draft_pick"])
    d["draft_team"] = d["team"]
    d["draft_team_norm"] = d["team"].map(s3.team_norm)
    d = d[["season", "draft_pick", "draft_team", "draft_team_norm"]]

    p["season"] = pd.to_numeric(p["season"], errors="coerce")
    p["draft_pick"] = pd.to_numeric(p["draft_pick"], errors="coerce")
    p = p.merge(d, on=["season", "draft_pick"], how="left", validate="many_to_one")

    lt = s3.landing_table(weekly)
    if not lt.empty:
        p["prior_season"] = p["season"] - 1
        p = p.merge(lt, on=["prior_season", "draft_team_norm"], how="left", validate="many_to_one")
        p = p.drop(columns=["prior_season"], errors="ignore")
        for pos in POSITIONS:
            pre = f"landing_{pos.lower()}"
            if f"{pre}_top_ppr_share" in p.columns:
                p[f"{pre}_ppr_concentration"] = p[f"{pre}_top_ppr_share"]
            if f"{pre}_top_targets_share" in p.columns:
                p[f"{pre}_target_concentration"] = p[f"{pre}_top_targets_share"]
    return p


def main() -> None:
    print("Loading live cfbfastR/SportsDataverse data...")
    pa = b.load_cfb_table("passing")
    ru = b.load_cfb_table("rushing")
    re = b.load_cfb_table("receiving")
    team = b.load_cfb_table("team_summaries")
    pa, ru, re = s2.prep_college(pa, ru, re, team)

    print("Loading live nflverse data...")
    draft, players, combine, weekly = b.load_nflverse()
    draft["season"] = pd.to_numeric(draft["season"], errors="coerce")

    print("Loading current nflverse draft_picks release for 2026...")
    current_2026 = load_current_2026_draft()
    # Preserve the frozen historical draft input and replace/append only the scoring class.
    draft = draft[~draft["season"].eq(PREDICT_YEAR)].copy()
    draft = pd.concat([draft, current_2026], ignore_index=True, sort=False)
    draft["season"] = pd.to_numeric(draft["season"], errors="coerce")
    draft = draft[
        draft.category.isin(POSITIONS)
        & draft.season.between(b.TRAIN_DRAFT_START, PREDICT_YEAR)
    ].copy()

    live_2026 = draft[draft.season.eq(PREDICT_YEAR)].copy()
    if live_2026.empty:
        mx = pd.to_numeric(draft.season, errors="coerce").max()
        raise RuntimeError(f"Live nflverse draft feed has no {PREDICT_YEAR} rows after refresh; max season={mx}")
    print(f"Found {len(live_2026)} drafted 2026 QB/RB/WR/TE prospects")

    prof = s2.build_profiles(draft, pa, ru, re)
    prof = s2.add_nfl_meta(prof, players, combine)
    prof = s2.build_targets(prof, weekly)
    prof = add_landing_features_by_pick(prof, draft, weekly)
    prof = s3.add_scouting_surprise(prof)

    cur = prof[prof.season.eq(PREDICT_YEAR)].copy()
    if cur.empty:
        raise RuntimeError("2026 draft rows were present before feature engineering but disappeared from the prospect pool")

    jobs = {}
    for pos in POSITIONS:
        path = ROOT / "trained_v3" / f"{pos}.joblib"
        if not path.exists():
            raise RuntimeError(f"Missing frozen Stage 3 trained model: {path}")
        jobs[pos] = joblib.load(path)

    rankings = s3.current_rankings(prof, jobs)
    if rankings.empty:
        raise RuntimeError("Frozen Stage 3 models produced no 2026 rankings")

    rankings.to_csv(OUT / "rookie_rankings_2026.csv", index=False)
    cur.to_csv(OUT / "prospect_pool_2026.csv", index=False)

    audit = []
    for pos in POSITIONS:
        z = cur[cur.position.eq(pos)]
        r = rankings[rankings.position.eq(pos)]
        audit.append({
            "position": pos,
            "drafted_2026": int(len(z)),
            "ranked_2026": int(len(r)),
            "college_match_rate": float(pd.to_numeric(z.get("college_match"), errors="coerce").mean()) if len(z) else np.nan,
            "fuzzy_match_rate": float(pd.to_numeric(z.get("fuzzy_match"), errors="coerce").mean()) if len(z) else np.nan,
            "draft_team_match_rate": float(z.get("draft_team", pd.Series(index=z.index, dtype=object)).notna().mean()) if len(z) else np.nan,
        })
    pd.DataFrame(audit).to_csv(OUT / "scoring_audit.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_year": PREDICT_YEAR,
        "model_source": "frozen Stage 3 trained_v3 models",
        "model_retrained": False,
        "historical_draft_source": "nflverse/nfldata draft_picks loaded by modeling.load_nflverse",
        "current_draft_source": CURRENT_DRAFT_URL,
        "scoring_population_rows": int(len(cur)),
        "ranked_rows": int(len(rankings)),
        "counts_by_position": {
            pos: int((rankings.position == pos).sum()) for pos in POSITIONS
        },
        "note": "Score-only refresh. No target, feature-selection, validation, model-selection, or frozen-2023 logic was changed.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
