from __future__ import annotations

# Post-registration trigger for full-history export.
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_full_history"
OUT.mkdir(parents=True, exist_ok=True)

HIST = ROOT / "results_stage5" / "historical_prospect_calibration_rows.csv"
CURRENT_DIR = ROOT / "results_2023_2026_final"
CURRENT_YEARS = [2024, 2025, 2026]
POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def historical_rows() -> pd.DataFrame:
    h = pd.read_csv(HIST, low_memory=False)
    h = h[pd.to_numeric(h["season"], errors="coerce").le(2023)].copy()
    out = pd.DataFrame({
        "draft_year": pd.to_numeric(h["season"], errors="coerce").astype("Int64"),
        "position": h["position"],
        "player_name": h["pfr_name"],
        "result_type": "historical_oof",
        "draft_team": np.nan,
        "draft_round": np.nan,
        "draft_pick": pd.to_numeric(h["draft_pick"], errors="coerce"),
        "model_score_best2of3_ppg": pd.to_numeric(h["prospect_model_score"], errors="coerce"),
        "prospect_model_percentile_pooled": pd.to_numeric(h["prospect_model_percentile"], errors="coerce"),
        "prospect_model_percentile_temporal": pd.to_numeric(h["temporal_prospect_percentile"], errors="coerce"),
        "temporal_reference_n": pd.to_numeric(h["temporal_reference_n"], errors="coerce"),
        "hit_probability": np.nan,
        "realized_primary_ppg": pd.to_numeric(h["primary_ppg"], errors="coerce"),
        "hit3": pd.to_numeric(h["hit3"], errors="coerce"),
        "star3": pd.to_numeric(h["star3"], errors="coerce"),
        "best_rank3": pd.to_numeric(h["best_rank3"], errors="coerce"),
        "target_valid": pd.to_numeric(h["target_valid"], errors="coerce"),
        "training_rows_before_draft": pd.to_numeric(h["training_rows_before_draft"], errors="coerce"),
        "college_match": np.nan,
        "fuzzy_match": np.nan,
        "scout_boost": np.nan,
        "outcome_complete": 1,
    })
    return out


def current_rows() -> pd.DataFrame:
    frames = []
    for year in CURRENT_YEARS:
        p = CURRENT_DIR / f"rookie_results_{year}.csv"
        d = pd.read_csv(p, low_memory=False)
        frames.append(pd.DataFrame({
            "draft_year": pd.to_numeric(d["season"], errors="coerce").astype("Int64"),
            "position": d["position"],
            "player_name": d["pfr_name"],
            "result_type": "frozen_prospect",
            "draft_team": d.get("draft_team"),
            "draft_round": pd.to_numeric(d.get("draft_round"), errors="coerce"),
            "draft_pick": pd.to_numeric(d.get("draft_pick"), errors="coerce"),
            "model_score_best2of3_ppg": pd.to_numeric(d["prospect_model_score"], errors="coerce"),
            "prospect_model_percentile_pooled": pd.to_numeric(d["prospect_model_percentile"], errors="coerce"),
            "prospect_model_percentile_temporal": np.nan,
            "temporal_reference_n": np.nan,
            "hit_probability": pd.to_numeric(d.get("hit_probability"), errors="coerce"),
            "realized_primary_ppg": np.nan,
            "hit3": np.nan,
            "star3": np.nan,
            "best_rank3": np.nan,
            "target_valid": np.nan,
            "training_rows_before_draft": np.nan,
            "college_match": pd.to_numeric(d.get("college_match"), errors="coerce"),
            "fuzzy_match": pd.to_numeric(d.get("fuzzy_match"), errors="coerce"),
            "scout_boost": pd.to_numeric(d.get("scout_boost"), errors="coerce"),
            "outcome_complete": 0,
        }))
    return pd.concat(frames, ignore_index=True)


def add_ranks(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["position_year_rank"] = d.groupby(["draft_year", "position"])["model_score_best2of3_ppg"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    d["all_years_position_rank_by_percentile"] = d.groupby("position")["prospect_model_percentile_pooled"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    d["all_years_overall_rank_by_percentile"] = d["prospect_model_percentile_pooled"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    d["position_sort"] = d["position"].map(POS_ORDER).fillna(9)
    d = d.sort_values(["draft_year", "position_sort", "position_year_rank", "draft_pick", "player_name"], na_position="last")
    return d.drop(columns=["position_sort"]).reset_index(drop=True)


def write_summary(d: pd.DataFrame) -> None:
    s = d.groupby(["draft_year", "position", "result_type"], dropna=False).agg(
        player_count=("player_name", "size"),
        mean_model_score=("model_score_best2of3_ppg", "mean"),
        max_model_score=("model_score_best2of3_ppg", "max"),
        mean_prospect_percentile=("prospect_model_percentile_pooled", "mean"),
        realized_hit_rate=("hit3", "mean"),
        realized_star_rate=("star3", "mean"),
    ).reset_index()
    s.to_csv(OUT / "full_model_results_summary_by_year_position.csv", index=False)


def main() -> None:
    d = pd.concat([historical_rows(), current_rows()], ignore_index=True)
    d = add_ranks(d)

    ordered = [
        "draft_year", "position", "position_year_rank", "player_name", "result_type",
        "draft_team", "draft_round", "draft_pick",
        "model_score_best2of3_ppg", "prospect_model_percentile_pooled",
        "prospect_model_percentile_temporal", "temporal_reference_n",
        "all_years_position_rank_by_percentile", "all_years_overall_rank_by_percentile",
        "hit_probability", "realized_primary_ppg", "hit3", "star3", "best_rank3",
        "outcome_complete", "target_valid", "training_rows_before_draft",
        "college_match", "fuzzy_match", "scout_boost",
    ]
    d = d[ordered]
    d.to_csv(OUT / "all_model_results_2007_2026.csv", index=False)
    write_summary(d)

    counts = {
        str(int(y)): int(n) for y, n in d.groupby("draft_year").size().items()
    }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(d)),
        "first_draft_year": int(d.draft_year.min()),
        "last_draft_year": int(d.draft_year.max()),
        "positions": ["QB", "RB", "WR", "TE"],
        "counts_by_year": counts,
        "historical_definition": "2007/2008-2023 uses leakage-safe walk-forward/OOF prospect model scores and complete first-three-season outcomes.",
        "current_definition": "2024-2026 uses frozen prospect scores from the reconstructed class files; complete three-year outcomes are intentionally left blank.",
        "rank_note": "Cross-position all-years rank uses same-position pooled prospect percentile. Ties receive the same statistical rank.",
        "source_files": [
            "results_stage5/historical_prospect_calibration_rows.csv",
            "results_2023_2026_final/rookie_results_2024.csv",
            "results_2023_2026_final/rookie_results_2025.csv",
            "results_2023_2026_final/rookie_results_2026.csv",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    readme = f"""# Full Rookie Model Results, 2007-2026\n\nGenerated {manifest['generated_utc']}\n\nThe master CSV contains {len(d):,} QB/RB/WR/TE prospect rows.\n\n- Historical rows through 2023 use leakage-safe out-of-fold scores.\n- 2024-2026 use frozen prospect-class scores.\n- Realized `primary_ppg`, `hit3`, `star3`, and `best_rank3` are populated only when the full three-season outcome window is complete.\n- `prospect_model_percentile_pooled` is a same-position historical percentile, not a success probability.\n- `prospect_model_percentile_temporal` is the historical as-of-draft percentile when enough prior OOF reference rows existed.\n- `hit_probability` is available for the reconstructed 2024-2026 frozen prospect classes; it is intentionally blank in this master file for earlier OOF rows rather than mixing non-equivalent classifier estimates.\n"""
    (OUT / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
