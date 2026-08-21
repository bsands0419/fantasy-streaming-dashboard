from __future__ import annotations

import re
import time
from datetime import date, timedelta
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

# Expected final-update anchors are tied to each season's title-game calendar.
# We probe a small window around each anchor and keep the latest valid TeamRankings
# table. This avoids the false monotonicity assumption that broke the earlier
# January-wide binary search while still verifying the actual final valid date.
EXPECTED_FINAL = {
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


def discover_final_snapshot(season: int) -> tuple[str, pd.DataFrame, list[dict]]:
    anchor = date.fromisoformat(EXPECTED_FINAL[season])
    probes = []
    # Probe latest to earliest. +/- 3 days is deliberately wider than observed
    # historical differences between the title game and TeamRankings' final table.
    for offset in range(3, -4, -1):
        ds = (anchor + timedelta(days=offset)).isoformat()
        try:
            d = parse_date_snapshot(ds, season)
            ok = not d.empty
        except Exception as exc:
            d = pd.DataFrame()
            ok = False
            probes.append({"college_season": season, "probe_date": ds, "valid": 0, "error": repr(exc)})
            continue
        probes.append({"college_season": season, "probe_date": ds, "valid": int(ok), "error": ""})
        if ok:
            return ds, d, probes
        time.sleep(0.10)
    raise RuntimeError(f"No valid TeamRankings SOS table within +/-3 days of {anchor} for season {season}")


def main():
    frames = []
    dates = []
    probe_rows = []
    for season in sorted(EXPECTED_FINAL):
        final_date, d, probes = discover_final_snapshot(season)
        frames.append(d)
        probe_rows.extend(probes)
        dates.append({
            "college_season": season,
            "expected_anchor": EXPECTED_FINAL[season],
            "final_teamrankings_snapshot": final_date,
            "draft_year_if_final_college_season": season + 1,
            "teams": len(d),
        })
        print(season, final_date, len(d))

    history = pd.concat(frames, ignore_index=True)
    date_map = pd.DataFrame(dates)
    probe_audit = pd.DataFrame(probe_rows)

    # Hard regression tests for dates independently verified during Stage 6.
    expected_checks = {
        2016: "2017-01-10",
        2018: "2019-01-08",
        2019: "2020-01-14",
        2021: "2022-01-11",
        2022: "2023-01-10",
        2023: "2024-01-09",
        2024: "2025-01-21",
        2025: "2026-01-20",
    }
    for season, expected in expected_checks.items():
        got = date_map.loc[date_map.college_season.eq(season), "final_teamrankings_snapshot"].iloc[0]
        if got != expected:
            raise RuntimeError(f"Season {season} final snapshot should be {expected}, got {got}")

    history.to_csv(OUT / "teamrankings_sos_final_history.csv", index=False)
    date_map.to_csv(OUT / "teamrankings_sos_final_dates.csv", index=False)
    probe_audit.to_csv(OUT / "teamrankings_sos_probe_audit.csv", index=False)
    print(date_map.to_string(index=False))


if __name__ == "__main__":
    main()
