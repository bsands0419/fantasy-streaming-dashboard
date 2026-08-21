from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_stage6_sos"
CACHE = ROOT / ".cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

START_SEASON = 2007
END_SEASON = 2025
URL = "https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={date}"

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})


def norm_team(x: str) -> str:
    s = str(x).lower().replace("&", " and ")
    s = re.sub(r"\([^)]*\)$", "", s).strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def parse_date_snapshot(snapshot_date: str, college_season: int) -> pd.DataFrame:
    fp = CACHE / f"teamrankings_sos_exact_{snapshot_date}.html"
    if fp.exists():
        text = fp.read_text(errors="ignore")
    else:
        r = S.get(URL.format(date=snapshot_date), timeout=45)
        if r.status_code != 200:
            return pd.DataFrame()
        text = r.text
        fp.write_text(text)
    soup = BeautifulSoup(text, "html.parser")
    rows = []
    for table in soup.find_all("table"):
        headers = [re.sub(r"\s+", " ", th.get_text(" ", strip=True)) for th in table.find_all("th")]
        hs = " | ".join(headers).lower()
        if not ("rank" in hs and "team" in hs and "rating" in hs):
            continue
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 3:
                continue
            vals = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in cells]
            try:
                rank = int(re.sub(r"[^0-9]", "", vals[0]))
                rating = float(vals[2])
            except Exception:
                continue
            team = re.sub(r"\s*\([0-9]+-[0-9]+(?:-[0-9]+)?\)\s*$", "", vals[1]).strip()
            rows.append({
                "college_season": college_season,
                "draft_year_if_final_college_season": college_season + 1,
                "source_date": snapshot_date,
                "teamrankings_team": team,
                "team_key": norm_team(team),
                "tr_sos_rank": rank,
                "tr_sos_rating": rating,
                "source_url": URL.format(date=snapshot_date),
            })
        if rows:
            break
    if len(rows) < 80:
        return pd.DataFrame()
    d = pd.DataFrame(rows).drop_duplicates(["college_season", "team_key"])
    n = len(d)
    d["tr_sos_rank_pct"] = 1.0 - (d.tr_sos_rank - 1.0) / max(1.0, n - 1.0)
    sd = float(d.tr_sos_rating.std(ddof=0))
    d["tr_sos_z"] = (d.tr_sos_rating - float(d.tr_sos_rating.mean())) / sd if sd > 0 else 0.0
    return d


def valid_day(season: int, day: int) -> tuple[bool, pd.DataFrame]:
    ds = f"{season + 1}-01-{day:02d}"
    try:
        d = parse_date_snapshot(ds, season)
        return (not d.empty), d
    except Exception:
        return False, pd.DataFrame()


def discover_final_snapshot(season: int) -> tuple[str, pd.DataFrame]:
    # TeamRankings final snapshots are in January after the college season.
    # Validity is monotone over this period: pages exist through the final update,
    # then cease to return the season table. Binary search finds the last valid day.
    lo, hi = 1, 31
    best_day = None
    best_df = pd.DataFrame()
    cache = {}
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid not in cache:
            cache[mid] = valid_day(season, mid)
            time.sleep(0.15)
        ok, d = cache[mid]
        if ok:
            best_day, best_df = mid, d
            lo = mid + 1
        else:
            hi = mid - 1
    if best_day is None:
        # Fallback linear scan in case an older season violates monotonicity.
        for day in range(31, 0, -1):
            ok, d = cache.get(day, valid_day(season, day))
            if ok:
                best_day, best_df = day, d
                break
    if best_day is None:
        raise RuntimeError(f"No valid TeamRankings January SOS snapshot found for college season {season}")
    return f"{season + 1}-01-{best_day:02d}", best_df


def main():
    frames = []
    dates = []
    for season in range(START_SEASON, END_SEASON + 1):
        final_date, d = discover_final_snapshot(season)
        frames.append(d)
        dates.append({
            "college_season": season,
            "final_teamrankings_snapshot": final_date,
            "draft_year_if_final_college_season": season + 1,
            "teams": len(d),
        })
        print(season, final_date, len(d))

    history = pd.concat(frames, ignore_index=True)
    date_map = pd.DataFrame(dates)

    # Explicit regression tests for the dates already independently verified.
    got_2024 = date_map.loc[date_map.college_season.eq(2024), "final_teamrankings_snapshot"].iloc[0]
    got_2023 = date_map.loc[date_map.college_season.eq(2023), "final_teamrankings_snapshot"].iloc[0]
    got_2025 = date_map.loc[date_map.college_season.eq(2025), "final_teamrankings_snapshot"].iloc[0]
    if got_2024 != "2025-01-21":
        raise RuntimeError(f"2024 final snapshot should be 2025-01-21, got {got_2024}")
    if got_2023 != "2024-01-09":
        raise RuntimeError(f"2023 final snapshot should be 2024-01-09, got {got_2023}")
    if got_2025 != "2026-01-20":
        raise RuntimeError(f"2025 final snapshot should be 2026-01-20, got {got_2025}")

    history.to_csv(OUT / "teamrankings_sos_final_history.csv", index=False)
    date_map.to_csv(OUT / "teamrankings_sos_final_dates.csv", index=False)
    print(date_map.to_string(index=False))


if __name__ == "__main__":
    main()
