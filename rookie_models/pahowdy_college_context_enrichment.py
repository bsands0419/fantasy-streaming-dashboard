from __future__ import annotations

import base64
import gzip
import io
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

import modeling as b

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_pahowdy_context"
OUT.mkdir(parents=True, exist_ok=True)
IDX_B64 = ROOT / "pahowdy_context_player_index.csv.gz.b64"
SOS_EXISTING = ROOT / "results_stage6_sos" / "teamrankings_sos_final_history.csv"

# Final post-season TeamRankings snapshots. College season Y is always matched to
# a final table dated in January Y+1. This prevents draft-year off-by-one errors.
FINAL_SNAPSHOT = {
    2004: "2005-01-05",
    2005: "2006-01-05",
    2006: "2007-01-09",
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

TR_URLS = {
    "sos": "https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={date}",
    "own_plays": "https://www.teamrankings.com/college-football/stat/plays-per-game?date={date}",
    "opp_plays": "https://www.teamrankings.com/college-football/stat/opponent-plays-per-game?date={date}",
}

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})

POSITIONS = {"QB", "RB", "WR", "TE"}

TEAM_ALIASES = {
    "alabama birmingham": "uab",
    "uab blazers": "uab",
    "central florida": "ucf",
    "ucf knights": "ucf",
    "southern california": "usc",
    "usc trojans": "usc",
    "louisiana state": "lsu",
    "lsu tigers": "lsu",
    "texas christian": "tcu",
    "tcu horned frogs": "tcu",
    "brigham young": "byu",
    "byu cougars": "byu",
    "mississippi": "ole miss",
    "ole miss rebels": "ole miss",
    "louisiana lafayette": "louisiana",
    "ul lafayette": "louisiana",
    "louisiana monroe": "ulm",
    "ul monroe": "ulm",
    "miami florida": "miami fl",
    "miami hurricanes": "miami fl",
    "miami ohio": "miami oh",
    "north carolina state": "nc state",
    "n c state": "nc state",
    "southern methodist": "smu",
    "texas san antonio": "utsa",
    "texas el paso": "utep",
    "middle tennessee state": "middle tennessee",
    "appalachian state": "app state",
    "florida international": "fiu",
    "florida atlantic": "fau",
    "bowling green state": "bowling green",
    "san jose state": "san jose st",
    "fresno state": "fresno st",
    "boise state": "boise st",
    "kansas state": "kansas st",
    "iowa state": "iowa st",
    "oregon state": "oregon st",
    "washington state": "washington st",
    "colorado state": "colorado st",
    "utah state": "utah st",
    "arizona state": "arizona st",
    "oklahoma state": "oklahoma st",
    "mississippi state": "mississippi st",
    "michigan state": "michigan st",
    "pennsylvania state": "penn state",
    "ohio state": "ohio st",
}


def norm_name(x):
    s = unicodedata.normalize("NFKD", str(x or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def norm_team(x):
    s = unicodedata.normalize("NFKD", str(x or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\([^)]*\)$", "", s).strip()
    s = re.sub(r"\buniversity\b", "", s)
    s = re.sub(r"\bthe\b", "", s)
    s = re.sub(r"\bst\.?\b", "state", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = TEAM_ALIASES.get(s, s)
    return re.sub(r"[^a-z0-9]+", "", s)


def clean_id(x):
    s = str(x or "").strip()
    if not s or s in {"-", "UNK", "nan", "None"}:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_index():
    raw = gzip.decompress(base64.b64decode(IDX_B64.read_text().strip()))
    d = pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
    d["draft_year"] = pd.to_numeric(d["draft_year"], errors="coerce")
    d["cfb_id_clean"] = d["cfbfastR_id"].map(clean_id)
    d["name_norm"] = d["name"].map(norm_name)
    return d


def get_espn_team_map():
    maps = {}
    try:
        u = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams?limit=1000"
        j = S.get(u, timeout=60).json()
        entries = j.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        for e in entries:
            t = e.get("team", {})
            tid = clean_id(t.get("id"))
            labels = [t.get("displayName"), t.get("shortDisplayName"), t.get("location"), t.get("name"), t.get("abbreviation")]
            labels = [x for x in labels if x]
            if tid and labels:
                maps[tid] = labels
    except Exception as exc:
        print("ESPN team map unavailable", repr(exc))
    return maps


def parse_teamrankings(kind: str, season: int, snapshot: str) -> pd.DataFrame:
    url = TR_URLS[kind].format(date=snapshot)
    r = S.get(url, timeout=75)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    chosen_headers = None

    for table in soup.find_all("table"):
        headers = [re.sub(r"\s+", " ", th.get_text(" ", strip=True)) for th in table.find_all("th")]
        hlow = [h.lower() for h in headers]
        if "team" not in hlow or "rank" not in hlow:
            continue
        if kind == "sos" and not any("rating" in h for h in hlow):
            continue
        if kind != "sos" and not (str(season) in headers or len(headers) >= 3):
            continue

        chosen_headers = headers
        team_i = hlow.index("team") if "team" in hlow else 1
        rank_i = hlow.index("rank") if "rank" in hlow else 0
        if kind == "sos":
            value_i = next((i for i, h in enumerate(hlow) if "rating" in h), 2)
        else:
            value_i = headers.index(str(season)) if str(season) in headers else 2

        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all("td")]
            if len(cells) <= max(team_i, rank_i, value_i):
                continue
            try:
                rank = int(re.sub(r"[^0-9]", "", cells[rank_i]))
                value = float(cells[value_i])
            except Exception:
                continue
            team = re.sub(r"\s*\([0-9]+-[0-9]+(?:-[0-9]+)?\)\s*$", "", cells[team_i]).strip()
            rows.append({
                "college_season": season,
                "source_date": snapshot,
                "team": team,
                "team_key": norm_team(team),
                "rank": rank,
                "value": value,
                "source_url": url,
            })
        if rows:
            break

    d = pd.DataFrame(rows).drop_duplicates(["college_season", "team_key"])
    # This is also a guard against accidentally scraping only the visible/top section.
    if len(d) < 80:
        raise RuntimeError(f"{kind} {season} parsed only {len(d)} rows from full TeamRankings table; refusing to publish. headers={chosen_headers}")
    n = len(d)
    d["rank_pct"] = 1.0 - (d["rank"] - 1.0) / max(1.0, n - 1.0)
    sd = float(d["value"].std(ddof=0))
    d["z"] = (d["value"] - float(d["value"].mean())) / sd if sd > 0 else 0.0
    return d


def build_teamrankings_context():
    sos_existing = pd.read_csv(SOS_EXISTING) if SOS_EXISTING.exists() else pd.DataFrame()
    frames = []
    coverage = []
    for season, snapshot in FINAL_SNAPSHOT.items():
        print("TeamRankings season", season, snapshot)
        if not sos_existing.empty and season in set(pd.to_numeric(sos_existing.get("college_season"), errors="coerce").dropna().astype(int)):
            s = sos_existing[pd.to_numeric(sos_existing["college_season"], errors="coerce").eq(season)].copy()
            s = s.rename(columns={
                "teamrankings_team": "team",
                "tr_sos_rank": "rank",
                "tr_sos_rating": "value",
                "tr_sos_rank_pct": "rank_pct",
                "tr_sos_z": "z",
            })
            s = s[["college_season", "source_date", "team", "team_key", "rank", "value", "rank_pct", "z", "source_url"]]
        else:
            s = parse_teamrankings("sos", season, snapshot)

        own = parse_teamrankings("own_plays", season, snapshot)
        opp = parse_teamrankings("opp_plays", season, snapshot)
        s = s.rename(columns={"team": "tr_team", "rank": "sos_rank", "value": "sos_rating", "rank_pct": "sos_pct", "z": "sos_z", "source_url": "sos_url"})
        own = own.rename(columns={"rank": "own_plays_rank", "value": "own_plays_pg", "rank_pct": "own_plays_pct", "z": "own_plays_z", "source_url": "own_plays_url"})
        opp = opp.rename(columns={"rank": "opp_plays_rank", "value": "opp_plays_pg", "rank_pct": "opp_plays_pct", "z": "opp_plays_z", "source_url": "opp_plays_url"})
        own = own.drop(columns=["team", "source_date"], errors="ignore")
        opp = opp.drop(columns=["team", "source_date"], errors="ignore")
        z = s.merge(own, on=["college_season", "team_key"], how="outer").merge(opp, on=["college_season", "team_key"], how="outer")
        z["season_median_own_plays_pg"] = float(z["own_plays_pg"].median())
        z["season_mean_own_plays_pg"] = float(z["own_plays_pg"].mean())
        z["season_median_opp_plays_pg"] = float(z["opp_plays_pg"].median())
        z["pace_factor_to_median"] = z["season_median_own_plays_pg"] / z["own_plays_pg"]
        frames.append(z)
        coverage.append({
            "college_season": season,
            "snapshot_date": snapshot,
            "sos_rows": int(s["team_key"].nunique()),
            "own_plays_rows": int(own["team_key"].nunique()),
            "opp_plays_rows": int(opp["team_key"].nunique()),
            "merged_rows": int(z["team_key"].nunique()),
            "median_own_plays_pg": float(z["own_plays_pg"].median()),
            "median_opp_plays_pg": float(z["opp_plays_pg"].median()),
        })
    allc = pd.concat(frames, ignore_index=True)
    cov = pd.DataFrame(coverage)
    allc.to_csv(OUT / "teamrankings_college_context_history.csv", index=False)
    cov.to_csv(OUT / "teamrankings_full_page_coverage.csv", index=False)
    return allc


def first_col(df, names):
    return next((c for c in names if c in df.columns), None)


def row_team_label(row, team_map):
    for c in ["team_name", "team_display_name", "school", "team", "pos_team", "team_abbreviation", "team_slug"]:
        if c in row.index:
            v = row.get(c)
            if pd.notna(v) and str(v).strip():
                return str(v).strip(), c
    tid = clean_id(row.get("team_id"))
    if tid and tid in team_map:
        return team_map[tid][0], "espn_team_id_map"
    return tid, "team_id_only"


def prepare_component(df: pd.DataFrame, kind: str, team_map: dict) -> pd.DataFrame:
    if df.empty:
        return df
    if kind == "pass":
        namecol = "passer_player_name"
        stats = {"pass_yards": "yards", "pass_td": "passing_td", "pass_int": "pass_int", "pass_games": "games"}
    elif kind == "rush":
        namecol = "rusher_player_name"
        stats = {"rush_yards": "yards", "rush_td": "rushing_td", "rush_games": "games"}
    else:
        namecol = "receiver_player_name"
        stats = {"rec_yards": "yards", "receptions": "comp", "rec_td": "passing_td", "targets": "targets", "rec_games": "games"}

    d = df.copy()
    d["player_id_clean"] = d.get("player_id", pd.Series("", index=d.index)).map(clean_id)
    d["name_norm"] = d[namecol].map(norm_name)
    team_info = d.apply(lambda r: row_team_label(r, team_map), axis=1)
    d["team_label"] = [x[0] for x in team_info]
    d["team_label_source"] = [x[1] for x in team_info]
    d["team_id_clean"] = d.get("team_id", pd.Series("", index=d.index)).map(clean_id)
    keep = ["season", "player_id_clean", "name_norm", "team_id_clean", "team_label", "team_label_source"]
    for outc, inc in stats.items():
        d[outc] = pd.to_numeric(d.get(inc), errors="coerce")
        keep.append(outc)
    # Preserve one row per player/team/season and sum volume if raw input has duplicate lines.
    agg = {c: "sum" for c in keep if c.endswith(("yards", "td", "int", "targets", "receptions"))}
    for c in [x for x in keep if x.endswith("games")]:
        agg[c] = "max"
    agg["team_label"] = "last"
    agg["team_label_source"] = "last"
    out = d[keep].groupby(["season", "player_id_clean", "name_norm", "team_id_clean"], dropna=False).agg(agg).reset_index()
    return out


def match_team_context(team_label, team_id, season, ctx, team_map):
    z = ctx[ctx["college_season"].eq(season)].copy()
    if z.empty:
        return None, "no_teamrankings_season", 0.0
    candidates = z["team_key"].dropna().astype(str).tolist()

    labels = []
    if team_label:
        labels.append(str(team_label))
    tid = clean_id(team_id)
    if tid and tid in team_map:
        labels.extend(team_map[tid])

    for lab in labels:
        k = norm_team(lab)
        if k in set(candidates):
            return z[z["team_key"].eq(k)].iloc[0], "exact_team", 100.0

    best = None
    for lab in labels:
        k = norm_team(lab)
        if not k:
            continue
        m = process.extractOne(k, candidates, scorer=fuzz.ratio, score_cutoff=78)
        if m and (best is None or m[1] > best[1]):
            best = m
    if best:
        return z[z["team_key"].eq(best[0])].iloc[0], "fuzzy_team", float(best[1])
    return None, "team_unmatched", 0.0


def build_player_seasons(index: pd.DataFrame, ctx: pd.DataFrame):
    print("loading SportsDataverse college tables")
    pa = b.load_cfb_table("passing")
    ru = b.load_cfb_table("rushing")
    re_ = b.load_cfb_table("receiving")
    team_map = get_espn_team_map()

    pc = prepare_component(pa, "pass", team_map)
    rc = prepare_component(ru, "rush", team_map)
    cc = prepare_component(re_, "rec", team_map)
    components = {"pass": pc, "rush": rc, "rec": cc}

    # Name indexes make fallback deterministic and fast.
    name_maps = {}
    for kind, d in components.items():
        name_maps[kind] = defaultdict(list)
        for nn, g in d.groupby("name_norm"):
            name_maps[kind][nn] = g

    out = []
    player_audit = []

    for i, p in index.iterrows():
        dy = p["draft_year"]
        if not np.isfinite(dy) or p["pos"] not in POSITIONS:
            continue
        dy = int(dy)
        if dy < 2005 or dy > 2026:
            player_audit.append({"master_row": p["master_row"], "name": p["name"], "pos": p["pos"], "draft_year": dy, "status": "outside_college_source_window"})
            continue
        lo, hi = dy - 6, dy - 1
        cid = p["cfb_id_clean"]
        nn = p["name_norm"]

        if p["pos"] == "QB":
            kinds = ["pass", "rush", "rec"]
        elif p["pos"] == "RB":
            kinds = ["rush", "rec", "pass"]
        else:
            kinds = ["rec", "rush", "pass"]

        matched_parts = []
        methods = []
        for kind in kinds:
            d = components[kind]
            g = pd.DataFrame()
            method = "none"
            if cid:
                g = d[d["player_id_clean"].eq(cid) & d["season"].between(lo, hi)].copy()
                if not g.empty:
                    method = "exact_cfb_id"
            if g.empty and nn:
                g = d[d["name_norm"].eq(nn) & d["season"].between(lo, hi)].copy()
                if not g.empty:
                    method = "exact_name"
            if g.empty and nn:
                cands = [x for x in name_maps[kind].keys() if x]
                m = process.extractOne(nn, cands, scorer=fuzz.ratio, score_cutoff=95)
                if m:
                    gg = name_maps[kind][m[0]]
                    gg = gg[gg["season"].between(lo, hi)].copy()
                    if not gg.empty:
                        g = gg
                        method = f"fuzzy_name_{m[1]:.0f}"
            if not g.empty:
                g["component"] = kind
                matched_parts.append(g)
                methods.append(method)

        if not matched_parts:
            player_audit.append({"master_row": p["master_row"], "name": p["name"], "pos": p["pos"], "draft_year": dy, "status": "no_sportsdataverse_match"})
            continue

        # Merge pass/rush/receive contributions by actual season/team, preserving transfers.
        allp = pd.concat(matched_parts, ignore_index=True, sort=False)
        key = ["season", "team_id_clean", "team_label"]
        statcols = ["pass_yards", "pass_td", "pass_int", "pass_games", "rush_yards", "rush_td", "rush_games", "rec_yards", "receptions", "rec_td", "targets", "rec_games"]
        for c in statcols:
            if c not in allp.columns:
                allp[c] = np.nan
        agg = {c: "sum" for c in statcols if not c.endswith("games")}
        for c in [x for x in statcols if x.endswith("games")]:
            agg[c] = "max"
        g = allp.groupby(key, dropna=False).agg(agg).reset_index()
        for c in statcols:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)

        for _, r in g.iterrows():
            season = int(r["season"])
            tr, team_method, team_score = match_team_context(r["team_label"], r["team_id_clean"], season, ctx, team_map)
            games = max(float(r["pass_games"]), float(r["rush_games"]), float(r["rec_games"]), 0.0)
            # PPR-like college production proxy used only for context weighting. It is not
            # substituted for any original Pahowdy production field.
            raw_pts = (
                float(r["pass_yards"]) / 25.0 + float(r["pass_td"]) * 4.0 - float(r["pass_int"]) * 2.0
                + float(r["rush_yards"]) / 10.0 + float(r["rush_td"]) * 6.0
                + float(r["receptions"]) + float(r["rec_yards"]) / 10.0 + float(r["rec_td"]) * 6.0
            )
            raw_pg = raw_pts / games if games > 0 else raw_pts
            if tr is not None:
                pace_factor = float(tr.get("pace_factor_to_median")) if pd.notna(tr.get("pace_factor_to_median")) else np.nan
                pace_adj_pg = raw_pg * pace_factor if np.isfinite(pace_factor) else np.nan
                vals = {c: tr.get(c) for c in [
                    "tr_team", "source_date", "sos_rank", "sos_rating", "sos_pct", "sos_z",
                    "own_plays_rank", "own_plays_pg", "own_plays_pct", "own_plays_z",
                    "opp_plays_rank", "opp_plays_pg", "opp_plays_pct", "opp_plays_z",
                    "season_median_own_plays_pg", "pace_factor_to_median", "sos_url", "own_plays_url", "opp_plays_url"
                ]}
            else:
                pace_adj_pg = np.nan
                vals = {c: np.nan for c in [
                    "tr_team", "source_date", "sos_rank", "sos_rating", "sos_pct", "sos_z",
                    "own_plays_rank", "own_plays_pg", "own_plays_pct", "own_plays_z",
                    "opp_plays_rank", "opp_plays_pg", "opp_plays_pct", "opp_plays_z",
                    "season_median_own_plays_pg", "pace_factor_to_median", "sos_url", "own_plays_url", "opp_plays_url"
                ]}

            out.append({
                "master_row": p["master_row"], "name": p["name"], "pos": p["pos"], "draft_year": dy,
                "college_season": season, "sportsdataverse_team": r["team_label"], "team_id": r["team_id_clean"],
                "player_match_method": ";".join(sorted(set(methods))), "teamrankings_match_method": team_method,
                "teamrankings_match_score": team_score, "games": games,
                "pass_yards": r["pass_yards"], "pass_td": r["pass_td"], "pass_int": r["pass_int"],
                "rush_yards": r["rush_yards"], "rush_td": r["rush_td"], "receptions": r["receptions"],
                "rec_yards": r["rec_yards"], "rec_td": r["rec_td"], "targets": r["targets"],
                "raw_ppr_proxy": raw_pts, "raw_ppr_proxy_pg": raw_pg, "pace_adjusted_ppr_proxy_pg": pace_adj_pg,
                **vals,
            })
        player_audit.append({"master_row": p["master_row"], "name": p["name"], "pos": p["pos"], "draft_year": dy, "status": "matched", "player_match_methods": ";".join(sorted(set(methods)))})
        if (i + 1) % 250 == 0:
            print("processed", i + 1, "of", len(index))

    seas = pd.DataFrame(out)
    audit = pd.DataFrame(player_audit)
    seas.to_csv(OUT / "pahowdy_player_season_context.csv", index=False)
    audit.to_csv(OUT / "pahowdy_player_match_audit.csv", index=False)
    return seas, audit


def wavg(g, col, wcol):
    v = pd.to_numeric(g[col], errors="coerce")
    w = pd.to_numeric(g[wcol], errors="coerce").clip(lower=0)
    m = v.notna() & w.notna()
    if not m.any():
        return np.nan
    if float(w[m].sum()) <= 0:
        return float(v[m].mean())
    return float(np.average(v[m], weights=w[m]))


def summarize(index, seas, audit):
    rows = []
    for _, p in index.iterrows():
        mr = str(p["master_row"])
        g = seas[seas["master_row"].astype(str).eq(mr)].copy() if not seas.empty else pd.DataFrame()
        base = {"master_row": p["master_row"], "name": p["name"], "pos": p["pos"], "draft_year": p["draft_year"]}
        if g.empty:
            rows.append({**base, "college_context_status": "unavailable", "college_context_rows": 0})
            continue
        g = g.sort_values(["college_season", "sportsdataverse_team"])
        matched = g[g["sos_rating"].notna() & g["own_plays_pg"].notna()].copy()
        if matched.empty:
            rows.append({**base, "college_context_status": "player_matched_teamrankings_unmatched", "college_context_rows": len(g)})
            continue

        # Pace-adjusted production is the weighting basis so fast offenses do not receive
        # extra influence simply for running more plays.
        matched["context_weight"] = pd.to_numeric(matched["pace_adjusted_ppr_proxy_pg"], errors="coerce").clip(lower=0).fillna(0)
        if float(matched["context_weight"].sum()) <= 0:
            matched["context_weight"] = 1.0

        final_season = int(matched["college_season"].max())
        fg = matched[matched["college_season"].eq(final_season)].copy()
        final = fg.sort_values("context_weight", ascending=False).iloc[0]
        peak = matched.sort_values("pace_adjusted_ppr_proxy_pg", ascending=False).iloc[0]
        team_history = " | ".join(
            f"{int(r.college_season)}:{r.tr_team if pd.notna(r.tr_team) else r.sportsdataverse_team}"
            for _, r in matched.iterrows()
        )
        distinct_teams = matched["tr_team"].fillna(matched["sportsdataverse_team"]).astype(str).nunique()

        raw_sum = float(pd.to_numeric(matched["raw_ppr_proxy_pg"], errors="coerce").fillna(0).sum())
        adj_sum = float(pd.to_numeric(matched["pace_adjusted_ppr_proxy_pg"], errors="coerce").fillna(0).sum())
        rows.append({
            **base,
            "college_context_status": "complete" if len(matched) == len(g) else "partial",
            "college_context_rows": int(len(g)),
            "college_context_rows_matched": int(len(matched)),
            "college_seasons_matched": int(matched["college_season"].nunique()),
            "distinct_college_teams": int(distinct_teams),
            "transfer_count": int(max(0, distinct_teams - 1)),
            "college_team_history": team_history,
            "final_college_season": final_season,
            "final_college_team": final.get("tr_team"),
            "final_sos_rating": final.get("sos_rating"),
            "final_sos_pct": final.get("sos_pct"),
            "final_sos_z": final.get("sos_z"),
            "final_team_plays_pg": final.get("own_plays_pg"),
            "final_team_plays_pct": final.get("own_plays_pct"),
            "final_team_plays_z": final.get("own_plays_z"),
            "final_opp_plays_pg": final.get("opp_plays_pg"),
            "final_opp_plays_pct": final.get("opp_plays_pct"),
            "final_opp_plays_z": final.get("opp_plays_z"),
            "peak_production_college_season": int(peak.get("college_season")),
            "peak_production_college_team": peak.get("tr_team"),
            "peak_pace_adj_ppr_proxy_pg": peak.get("pace_adjusted_ppr_proxy_pg"),
            "peak_sos_rating": peak.get("sos_rating"),
            "peak_sos_pct": peak.get("sos_pct"),
            "peak_team_plays_pg": peak.get("own_plays_pg"),
            "career_avg_sos_rating": float(pd.to_numeric(matched["sos_rating"], errors="coerce").mean()),
            "career_prod_weighted_sos_rating": wavg(matched, "sos_rating", "context_weight"),
            "career_prod_weighted_sos_pct": wavg(matched, "sos_pct", "context_weight"),
            "career_prod_weighted_sos_z": wavg(matched, "sos_z", "context_weight"),
            "best_sos_rating": float(pd.to_numeric(matched["sos_rating"], errors="coerce").max()),
            "best_sos_pct": float(pd.to_numeric(matched["sos_pct"], errors="coerce").max()),
            "career_prod_weighted_team_plays_pg": wavg(matched, "own_plays_pg", "context_weight"),
            "career_prod_weighted_team_plays_pct": wavg(matched, "own_plays_pct", "context_weight"),
            "career_prod_weighted_team_plays_z": wavg(matched, "own_plays_z", "context_weight"),
            "career_prod_weighted_opp_plays_pg": wavg(matched, "opp_plays_pg", "context_weight"),
            "career_prod_weighted_opp_plays_pct": wavg(matched, "opp_plays_pct", "context_weight"),
            "career_prod_weighted_opp_plays_z": wavg(matched, "opp_plays_z", "context_weight"),
            "career_raw_ppr_proxy_pg_sum": raw_sum,
            "career_pace_adjusted_ppr_proxy_pg_sum": adj_sum,
            "career_pace_adjustment_ratio": (adj_sum / raw_sum) if raw_sum > 0 else np.nan,
            "context_weight_basis": "pace-adjusted PPR-like production per game",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "pahowdy_player_context_summary.csv", index=False)
    return out


def main():
    idx = load_index()
    ctx = build_teamrankings_context()
    seas, audit = build_player_seasons(idx, ctx)
    summary = summarize(idx, seas, audit)

    cov = pd.DataFrame([
        {"metric": "Pahowdy rows", "value": len(idx)},
        {"metric": "Player rows with season context", "value": int(summary["college_context_rows"].fillna(0).gt(0).sum())},
        {"metric": "Complete TeamRankings context", "value": int(summary["college_context_status"].eq("complete").sum())},
        {"metric": "Partial TeamRankings context", "value": int(summary["college_context_status"].eq("partial").sum())},
        {"metric": "Unavailable", "value": int(summary["college_context_status"].eq("unavailable").sum())},
        {"metric": "Player-season-team context rows", "value": len(seas)},
    ])
    cov.to_csv(OUT / "pahowdy_context_coverage_summary.csv", index=False)
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
