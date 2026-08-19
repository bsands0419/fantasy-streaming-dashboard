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
OUT = ROOT / "results_multiyear_2023_2026"
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ["QB", "RB", "WR", "TE"]
YEARS = [2023, 2024, 2025, 2026]


def pct(hist, vals):
    h = pd.to_numeric(hist, errors="coerce").dropna().to_numpy(float)
    v = pd.to_numeric(vals, errors="coerce").to_numpy(float)
    return np.array([100.0 * np.mean(h <= x) if np.isfinite(x) and len(h) else np.nan for x in v])


def build_profiles():
    pa = b.load_cfb_table("passing")
    ru = b.load_cfb_table("rushing")
    re = b.load_cfb_table("receiving")
    team = b.load_cfb_table("team_summaries")
    pa, ru, re = s2.prep_college(pa, ru, re, team)

    draft, players, combine, weekly = b.load_nflverse()
    draft["season"] = pd.to_numeric(draft["season"], errors="coerce")
    current_2026 = s26.load_current_2026_draft()
    draft = draft[~draft["season"].eq(2026)].copy()
    draft = pd.concat([draft, current_2026], ignore_index=True, sort=False)
    draft["season"] = pd.to_numeric(draft["season"], errors="coerce")
    draft = draft[draft.category.isin(POSITIONS) & draft.season.between(b.TRAIN_DRAFT_START, 2026)].copy()

    prof = s2.build_profiles(draft, pa, ru, re)
    prof = s2.add_nfl_meta(prof, players, combine)
    prof = s2.build_targets(prof, weekly)
    prof = s26.add_landing_features_by_pick(prof, draft, weekly)
    prof = s3.add_scouting_surprise(prof)
    return prof


def main():
    prof = build_profiles()
    jobs = {pos: joblib.load(ROOT / "trained_v3" / f"{pos}.joblib") for pos in POSITIONS}
    oof = pd.read_csv(ROOT / "results_2026" / "historical_prospect_oof.csv")
    rows = []

    for pos in POSITIONS:
        job = jobs[pos]
        hist_scores = pd.to_numeric(oof.loc[oof.position.eq(pos), "prospect_model_score"], errors="coerce").dropna()

        # 2023 stays truly out-of-fold: model fit only on earlier draft classes.
        z23 = oof[(oof.position == pos) & (oof.season == 2023)].copy()
        meta23 = prof[(prof.position == pos) & (prof.season == 2023)][[c for c in ["pfr_name","draft_round","draft_pick","draft_team"] if c in prof.columns]].copy()
        z23 = z23.merge(meta23, on=[c for c in ["pfr_name","draft_pick"] if c in z23.columns and c in meta23.columns], how="left")
        if not z23.empty:
            z23["year"] = 2023
            z23["pred_best2of3_ppg"] = z23["prospect_model_score"]
            z23["prospect_model_percentile"] = pct(hist_scores, z23["prospect_model_score"])
            z23["score_method"] = "walk-forward OOF, trained through 2022"
            rows.append(z23)

        # 2024-2026 use the same frozen models fitted through 2023.
        for year in [2024, 2025, 2026]:
            cur = prof[(prof.position == pos) & (prof.season == year)].copy()
            if cur.empty:
                continue
            a, bb = job["fitted"]["primary_ppg"]
            cur["pred_best2of3_ppg"] = s3.predict_pair(a, bb, cur, job)
            cur["prospect_model_score"] = cur["pred_best2of3_ppg"]
            cur["prospect_model_percentile"] = pct(hist_scores, cur["prospect_model_score"])
            cur["year"] = year
            cur["score_method"] = "frozen Stage 3 model trained through 2023"
            keep = [c for c in ["year","position","pfr_name","draft_team","draft_round","draft_pick","pred_best2of3_ppg","prospect_model_score","prospect_model_percentile","college_match","fuzzy_match","scout_boost","score_method"] if c in cur.columns]
            rows.append(cur[keep])

    allr = pd.concat(rows, ignore_index=True, sort=False)
    allr["year_rank"] = allr.groupby(["year","position"])["prospect_model_percentile"].rank(ascending=False, method="first").astype(int)
    allr["combined_2023_2026_rank"] = allr.groupby("position")["prospect_model_percentile"].rank(ascending=False, method="first").astype(int)
    cols = ["position","year","year_rank","combined_2023_2026_rank","pfr_name","prospect_model_percentile","pred_best2of3_ppg","draft_team","draft_round","draft_pick","score_method"]
    cols += [c for c in ["college_match","fuzzy_match","scout_boost"] if c in allr.columns]
    allr = allr[[c for c in cols if c in allr.columns]]

    allr.sort_values(["position","year","year_rank"]).to_csv(OUT / "all_years_separated.csv", index=False)
    allr.sort_values(["position","combined_2023_2026_rank"]).to_csv(OUT / "all_years_combined_ranked.csv", index=False)
    for pos in POSITIONS:
        p = allr[allr.position.eq(pos)].copy()
        p.sort_values(["year","year_rank"]).to_csv(OUT / f"{pos}_by_year.csv", index=False)
        p.sort_values("combined_2023_2026_rank").to_csv(OUT / f"{pos}_combined.csv", index=False)
    print(allr.groupby(["year","position"]).size())


if __name__ == "__main__":
    main()
