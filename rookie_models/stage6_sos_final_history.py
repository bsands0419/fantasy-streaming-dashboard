from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_stage6_sos"
CACHE = ROOT / ".cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

URL = "https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={date}"

# Exact final TeamRankings SOS snapshots for each completed college season.
# The college season is one year before the corresponding NFL draft class.
# Example: 2024 college season -> 2025-01-21 snapshot -> 2025 NFL Draft.
FINAL_SNAPSHOT = {
    2007: "2008-01-08",
    2008: "2009-01-09",
    2009: "2010-01-08",
    2010: "2011-01-11",
    2011: "2012-01-10",
    2012: "2013-01-08",
    2013: "2014-01-07",
    2014: "2015-01-13",
    2015: "2016-01-12",
    2016: "2017-01-10",
    2017: "2018-01-09",
    2018: "2019-01-08",
    2019: "2020-01-14",
    2020: "2021-01-12",
    2021: "2022-01-11",
    2022: "2023-01-10",
    2023: "2024-01-09",
    2024: "2025-01-21",
    2025: "2026-01-20",
}

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})


def norm_team(x: str) -> str:
    s = str(x).lower().replace("&", " and ")
    s = re.sub(r"\([^)]*\)$", "", s).strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def parse_snapshot(snapshot_date: str, college_season: int) -> pd.DataFrame:
    fp = CACHE / f"teamrankings_sos_exact_{snapshot_date}.html"
    if fp.exists():
        text = fp.read_text(errors="ignore")
    else:
        r = S.get(URL.format(date=snapshot_date), timeout=60)
        r.raise_for_status()
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

    d = pd.DataFrame(rows)
    if len(d) < 80:
        raise RuntimeError(
            f"TeamRankings snapshot {snapshot_date} for college season {college_season} "
            f"parsed only {len(d)} teams; refusing to publish"
        )
    d = d.drop_duplicates(["college_season", "team_key"])
    n = len(d)
    d["tr_sos_rank_pct"] = 1.0 - (d.tr_sos_rank - 1.0) / max(1.0, n - 1.0)
    sd = float(d.tr_sos_rating.std(ddof=0))
    d["tr_sos_z"] = (d.tr_sos_rating - float(d.tr_sos_rating.mean())) / sd if sd > 0 else 0.0
    return d


def main():
    frames = []
    audit = []
    for season, snapshot_date in FINAL_SNAPSHOT.items():
        d = parse_snapshot(snapshot_date, season)
        frames.append(d)
        audit.append({
            "college_season": season,
            "final_teamrankings_snapshot": snapshot_date,
            "draft_year_if_final_college_season": season + 1,
            "teams": len(d),
            "min_rating": float(d.tr_sos_rating.min()),
            "max_rating": float(d.tr_sos_rating.max()),
        })
        print(season, snapshot_date, len(d), d.tr_sos_rating.min(), d.tr_sos_rating.max())

    history = pd.concat(frames, ignore_index=True)
    date_map = pd.DataFrame(audit)

    # Critical mapping assertions requested for Stage 6.
    assert FINAL_SNAPSHOT[2023] == "2024-01-09"
    assert FINAL_SNAPSHOT[2024] == "2025-01-21"
    assert FINAL_SNAPSHOT[2025] == "2026-01-20"

    history.to_csv(OUT / "teamrankings_sos_final_history.csv", index=False)
    date_map.to_csv(OUT / "teamrankings_sos_final_dates.csv", index=False)
    print(date_map.to_string(index=False))


if __name__ == "__main__":
    main()
