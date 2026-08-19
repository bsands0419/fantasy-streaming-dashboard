from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_2023_2026_final"
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ["QB", "RB", "WR", "TE"]
YEARS = [2023, 2024, 2025, 2026]

SOURCES = {
    2023: ROOT / "results_2023_2026" / "rookie_results_2023.csv",
    2024: ROOT / "results_2023_2026" / "rookie_results_2024.csv",
    2025: ROOT / "results_2025" / "rookie_rankings_2025.csv",
    2026: ROOT / "results_2023_2026" / "rookie_results_2026.csv",
}

FIELDS = [
    "season", "position", "position_year_rank", "position_2023_2026_rank",
    "overall_percentile_rank_2023_2026", "pfr_name", "draft_team", "draft_round", "draft_pick",
    "prospect_model_score", "prospect_model_percentile", "hit_probability", "primary_ppg_realized",
    "college_match", "fuzzy_match", "scout_boost",
]


def fnum(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def clean_intish(value):
    x = fnum(value)
    if x is None:
        return ""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x)


def load_rows(year: int):
    path = SOURCES[year]
    if not path.exists():
        raise RuntimeError(f"Missing source file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Source file has no rows: {path}")

    out = []
    for r in rows:
        pos = (r.get("position") or "").strip().upper()
        score = fnum(r.get("prospect_model_score"))
        pct = fnum(r.get("prospect_model_percentile"))
        name = (r.get("pfr_name") or "").strip()
        if pos not in POSITIONS or score is None or pct is None or not name:
            continue
        out.append({
            "season": year,
            "position": pos,
            "position_year_rank": None,
            "position_2023_2026_rank": None,
            "overall_percentile_rank_2023_2026": None,
            "pfr_name": name,
            "draft_team": (r.get("draft_team") or "").strip(),
            "draft_round": clean_intish(r.get("draft_round")),
            "draft_pick": clean_intish(r.get("draft_pick")),
            "prospect_model_score": score,
            "prospect_model_percentile": pct,
            "hit_probability": fnum(r.get("hit_probability")),
            "primary_ppg_realized": fnum(r.get("primary_ppg_realized")),
            "college_match": fnum(r.get("college_match")),
            "fuzzy_match": fnum(r.get("fuzzy_match")),
            "scout_boost": fnum(r.get("scout_boost")),
        })
    if not out:
        raise RuntimeError(f"No valid modeled QB/RB/WR/TE rows in {path}")
    return out


def position_sort_key(r):
    return (-r["prospect_model_score"], -r["prospect_model_percentile"], r["season"], r["pfr_name"])


def overall_sort_key(r):
    return (-r["prospect_model_percentile"], r["position"], -r["prospect_model_score"], r["season"], r["pfr_name"])


def assign_ranks(rows):
    for year in YEARS:
        for pos in POSITIONS:
            block = sorted([r for r in rows if r["season"] == year and r["position"] == pos], key=position_sort_key)
            for i, r in enumerate(block, start=1):
                r["position_year_rank"] = i

    for pos in POSITIONS:
        block = sorted([r for r in rows if r["position"] == pos], key=position_sort_key)
        for i, r in enumerate(block, start=1):
            r["position_2023_2026_rank"] = i

    block = sorted(rows, key=overall_sort_key)
    last_pct = None
    current_rank = 0
    for i, r in enumerate(block, start=1):
        pct = r["prospect_model_percentile"]
        if last_pct is None or abs(pct - last_pct) > 1e-12:
            current_rank = i
            last_pct = pct
        r["overall_percentile_rank_2023_2026"] = current_rank


def serialize(r):
    out = {}
    for field in FIELDS:
        v = r.get(field)
        if v is None:
            out[field] = ""
        else:
            out[field] = v
    return out


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(serialize(r))


def main():
    rows = []
    counts_by_year = {}
    counts_by_year_position = {}
    for year in YEARS:
        yr = load_rows(year)
        rows.extend(yr)
        counts_by_year[str(year)] = len(yr)
        counts_by_year_position[str(year)] = {p: sum(1 for r in yr if r["position"] == p) for p in POSITIONS}

    assign_ranks(rows)

    for year in YEARS:
        yr = sorted([r for r in rows if r["season"] == year], key=lambda r: (POSITIONS.index(r["position"]), r["position_year_rank"]))
        write_csv(OUT / f"rookie_results_{year}.csv", yr)

    for pos in POSITIONS:
        block = sorted([r for r in rows if r["position"] == pos], key=lambda r: r["position_2023_2026_rank"])
        write_csv(OUT / f"{pos}_2023_2026_ranked.csv", block)

    all_pos = sorted(rows, key=overall_sort_key)
    write_csv(OUT / "all_players_2023_2026_by_percentile.csv", all_pos)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "years": YEARS,
        "positions": POSITIONS,
        "total_ranked_rows": len(rows),
        "counts_by_year": counts_by_year,
        "counts_by_year_position": counts_by_year_position,
        "position_rank_definition": "Raw frozen prospect_model_score, compared only within the same position across the 2023-2026 draft classes.",
        "overall_rank_definition": "Leakage-safe same-position historical prospect_model_percentile. Cross-position ties retain the same statistical rank; file order within a tie is deterministic but not intended as extra model precision.",
        "model_changes": "None. This script only combines and reranks already generated frozen-model scores.",
        "source_files": {str(y): str(SOURCES[y].relative_to(ROOT)) for y in YEARS},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = """# 2023-2026 Rookie Model Comparison\n\nThis directory is the clean comparison output for the 2023, 2024, 2025 and 2026 draft classes.\n\n- `rookie_results_YEAR.csv`: each class separated by year and ranked within QB/RB/WR/TE.\n- `QB_2023_2026_ranked.csv`, `RB_...`, `WR_...`, `TE_...`: all four draft classes ranked together within position using the frozen prospect-model score.\n- `all_players_2023_2026_by_percentile.csv`: all positions together using the leakage-safe position-normalized prospect-model percentile. Tied percentiles remain tied.\n\nNo model was retrained or reselected to create these files.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
