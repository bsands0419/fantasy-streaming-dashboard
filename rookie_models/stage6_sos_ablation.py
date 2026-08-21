from __future__ import annotations

import io
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

import modeling as b
import stage2 as s2
import stage6a_scout_ablation as eval6

ROOT = Path(__file__).resolve().parent
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CUR_POOL = ROOT / "results_2026" / "prospect_pool_2026.csv"
MODELS = ROOT / "trained_v3"
OUT = ROOT / "results_stage6_sos"
CACHE = ROOT / ".cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
SOS_START_SEASON = 2007
SOS_END_SEASON = 2025
TEAMRANKINGS_URL = "https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={date}"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})

TEAM_ALIASES = {
    "ohio st": "ohio state", "penn st": "penn state", "florida st": "florida state",
    "michigan st": "michigan state", "kansas st": "kansas state", "oklahoma st": "oklahoma state",
    "iowa st": "iowa state", "arizona st": "arizona state", "mississippi st": "mississippi state",
    "colorado st": "colorado state", "fresno st": "fresno state", "san diego st": "san diego state",
    "san jose st": "san jose state", "utah st": "utah state", "arkansas st": "arkansas state",
    "ball st": "ball state", "kent st": "kent state", "new mexico st": "new mexico state",
    "georgia st": "georgia state", "washington st": "washington state", "oregon st": "oregon state",
    "boise st": "boise state", "texas st": "texas state", "app state": "appalachian state",
    "s florida": "south florida", "e carolina": "east carolina", "n texas": "north texas",
    "w michigan": "western michigan", "e michigan": "eastern michigan", "c michigan": "central michigan",
    "n illinois": "northern illinois", "s alabama": "south alabama", "w kentucky": "western kentucky",
    "georgia so": "georgia southern", "coastal car": "coastal carolina", "middle tenn": "middle tennessee",
    "florida intl": "florida international", "ul monroe": "louisiana monroe", "j madison": "james madison",
    "miami oh": "miami ohio", "hawai i": "hawaii", "uconn": "connecticut", "umass": "massachusetts",
    "usc": "southern california", "mississippi": "ole miss", "la lafayette": "louisiana",
    "louisiana lafayette": "louisiana", "utsa": "texas san antonio", "utep": "texas el paso",
    "uab": "alabama birmingham", "ucf": "central florida", "smu": "southern methodist",
    "tcu": "texas christian", "byu": "brigham young", "miami fl": "miami",
}


def norm_team(x: str) -> str:
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\([^)]*\)$", "", s).strip()
    s = re.sub(r"\buniversity\b|\bthe\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = TEAM_ALIASES.get(s, s)
    return re.sub(r"[^a-z0-9]", "", s)


def scrape_teamrankings_season(season: int) -> pd.DataFrame:
    source_date = f"{season + 1}-01-20"
    url = TEAMRANKINGS_URL.format(date=source_date)
    fp = CACHE / f"teamrankings_sos_{season}_{source_date}.html"
    if fp.exists():
        text = fp.read_text(errors="ignore")
    else:
        last = None
        for attempt in range(4):
            try:
                r = SESSION.get(url, timeout=60)
                r.raise_for_status()
                text = r.text
                fp.write_text(text)
                break
            except Exception as exc:
                last = exc
                if attempt == 3:
                    raise RuntimeError(f"TeamRankings fetch failed for season={season}, date={source_date}: {exc}")
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(last)

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
                "college_season": season,
                "draft_year_if_final_college_season": season + 1,
                "source_date": source_date,
                "teamrankings_team": team,
                "team_key": norm_team(team),
                "tr_sos_rank": rank,
                "tr_sos_rating": rating,
                "source_url": url,
            })
        if rows:
            break
    if not rows:
        raise RuntimeError(f"No TeamRankings SOS table parsed for season={season}, date={source_date}")
    d = pd.DataFrame(rows).drop_duplicates(["college_season", "team_key"])
    n = len(d)
    d["tr_sos_rank_pct"] = 1.0 - (d.tr_sos_rank - 1.0) / max(1.0, n - 1.0)
    sd = float(d.tr_sos_rating.std(ddof=0))
    d["tr_sos_z"] = (d.tr_sos_rating - float(d.tr_sos_rating.mean())) / sd if sd > 0 else 0.0
    return d


def load_teamrankings_history() -> pd.DataFrame:
    frames = []
    failures = []
    for season in range(SOS_START_SEASON, SOS_END_SEASON + 1):
        try:
            d = scrape_teamrankings_season(season)
            frames.append(d)
            print(f"TeamRankings SOS {season}: {len(d)} teams")
        except Exception as exc:
            failures.append({"college_season": season, "error": str(exc)})
            print("SOS MISS", season, exc)
    if failures:
        pd.DataFrame(failures).to_csv(OUT / "teamrankings_fetch_failures.csv", index=False)
    if not frames:
        raise RuntimeError("No TeamRankings SOS seasons loaded")
    out = pd.concat(frames, ignore_index=True)
    # Validation needs college seasons through 2021; 2026 scoring needs 2025.
    required = {2018, 2019, 2020, 2021, 2022, 2025}
    missing = sorted(required - set(out.college_season.astype(int)))
    if missing:
        raise RuntimeError(f"Required TeamRankings SOS seasons missing: {missing}")
    out.to_csv(OUT / "teamrankings_sos_history.csv", index=False)
    return out


def team_name_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "team", "team_name", "pos_team", "school", "school_name", "display_name",
        "team_display_name", "location", "short_display_name", "team_location",
    ]
    for c in preferred:
        if c in df.columns:
            s = df[c].astype(str)
            if s.nunique(dropna=True) >= 20:
                return c
    return None


def add_team_name_from_summary(d: pd.DataFrame, team: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    out = d.copy()
    c = team_name_column(out)
    if c:
        return out, c
    tc = team_name_column(team)
    if tc and "team_id" in out.columns and "team_id" in team.columns:
        t = team[["season", "team_id", tc]].drop_duplicates(["season", "team_id"]).rename(columns={tc: "_sos_team_name"})
        out = out.merge(t, on=["season", "team_id"], how="left")
        return out, "_sos_team_name"
    return out, None


def build_name_map(active_names: list[str], sos_season: pd.DataFrame, season: int) -> list[dict]:
    sos_keys = sos_season.team_key.tolist()
    key_to_row = sos_season.set_index("team_key")
    audit = []
    for raw in sorted(set(x for x in active_names if str(x).strip())):
        key = norm_team(raw)
        method = "exact"
        score = 100.0
        match = key if key in key_to_row.index else None
        if match is None and key:
            found = process.extractOne(key, sos_keys, scorer=fuzz.ratio, score_cutoff=88)
            if found:
                match, score, _ = found
                method = "fuzzy"
        audit.append({
            "college_season": season,
            "source_team_name": raw,
            "source_team_key": key,
            "matched_team_key": match,
            "matched_teamrankings_name": key_to_row.loc[match, "teamrankings_team"] if match is not None else None,
            "match_method": method if match is not None else "unmatched",
            "match_score": score if match is not None else np.nan,
        })
    return audit


def attach_sos(d: pd.DataFrame, team: pd.DataFrame, sos: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, ncol = add_team_name_from_summary(d, team)
    if not ncol:
        raise RuntimeError(f"Cannot find team name column for {dataset}; columns={list(d.columns)[:80]}")
    audits = []
    out["_sos_source_team"] = out[ncol].astype(str)
    out["_sos_source_key"] = out["_sos_source_team"].map(norm_team)
    mapping_rows = []
    for season, g in out.groupby("season"):
        sy = int(season)
        ss = sos[sos.college_season.eq(sy)].copy()
        if ss.empty:
            continue
        a = build_name_map(g["_sos_source_team"].dropna().astype(str).unique().tolist(), ss, sy)
        for r in a:
            r["dataset"] = dataset
        audits.extend(a)
        mapping_rows.extend(a)
    mp = pd.DataFrame(mapping_rows)
    if mp.empty:
        out["tr_sos_rating"] = np.nan
        out["tr_sos_z"] = np.nan
        out["tr_sos_rank_pct"] = np.nan
        return out, pd.DataFrame(audits)
    mp = mp[["college_season", "source_team_key", "matched_team_key"]].drop_duplicates()
    out = out.merge(mp, left_on=["season", "_sos_source_key"], right_on=["college_season", "source_team_key"], how="left")
    ss = sos[["college_season", "team_key", "tr_sos_rating", "tr_sos_z", "tr_sos_rank_pct"]].copy()
    out = out.merge(ss, left_on=["season", "matched_team_key"], right_on=["college_season", "team_key"], how="left", suffixes=("", "_tr"))
    out = out.drop(columns=["college_season_x", "college_season_y", "college_season", "source_team_key", "matched_team_key", "team_key", "_sos_source_key"], errors="ignore")
    return out, pd.DataFrame(audits)


def combine_for_sos(d: pd.DataFrame, namecol: str, vol: list[str]) -> pd.DataFrame:
    return s2.combine_player_season(d, namecol, vol)


def aggregate_sos_profiles(master: pd.DataFrame, pa: pd.DataFrame, ru: pd.DataFrame, re_: pd.DataFrame) -> pd.DataFrame:
    pvol = ['plays','games','team_games','TEPA','yards','comp','att','passing_td','sacked','sack_yds','pass_int','dropbacks']
    rvol = ['plays','games','team_games','TEPA','yards','rushing_td','fumbles','team_rushes','team_rush_yards','team_rush_td','team_pass_att','team_dropbacks']
    cvol = ['plays','games','team_games','TEPA','yards','comp','targets','passing_td','fumbles','team_targets','team_rec_yards','team_rec_td','team_receptions','team_pass_att','team_dropbacks']
    pa2 = combine_for_sos(pa, 'passer_player_name', pvol)
    ru2 = combine_for_sos(ru, 'rusher_player_name', rvol)
    re2 = combine_for_sos(re_, 'receiver_player_name', cvol)
    maps = {
        'QB': ({k:g for k,g in pa2.groupby('name_norm')}, list(pa2.name_norm.dropna().unique())),
        'RB': ({k:g for k,g in ru2.groupby('name_norm')}, list(ru2.name_norm.dropna().unique())),
        'WR': ({k:g for k,g in re2.groupby('name_norm')}, list(re2.name_norm.dropna().unique())),
        'TE': ({k:g for k,g in re2.groupby('name_norm')}, list(re2.name_norm.dropna().unique())),
    }
    rows=[]
    for _, r in master[['season','pfr_id','pfr_name','position']].drop_duplicates().iterrows():
        pos=r.position; year=int(r.season); nn=b.norm_name(r.pfr_name)
        mp,cands=maps[pos]
        grp, mt = s2.lookup_group(nn, mp, cands, year)
        rec={'season':year,'pfr_id':r.pfr_id,'pfr_name':r.pfr_name,'position':pos,'tr_sos_player_match':mt}
        if not grp.empty:
            feat=s2.career_features(grp,'tr',['tr_sos_rating','tr_sos_z','tr_sos_rank_pct'],year)
            rec.update(feat)
            q=grp[(grp.season<year)&(grp.season>=year-6)]
            rec['tr_sos_college_seasons_with_match']=int(pd.to_numeric(q.tr_sos_z, errors='coerce').notna().sum()) if 'tr_sos_z' in q else 0
            rec['tr_sos_college_seasons_total']=int(q.season.nunique())
        rows.append(rec)
    out=pd.DataFrame(rows)
    keep=[
        'season','pfr_id','pfr_name','position','tr_sos_player_match','tr_sos_college_seasons_with_match','tr_sos_college_seasons_total',
        'tr_tr_sos_z_final','tr_tr_sos_z_mean','tr_tr_sos_z_peak',
        'tr_tr_sos_rank_pct_final','tr_tr_sos_rank_pct_mean',
        'tr_tr_sos_rating_final','tr_tr_sos_rating_mean',
    ]
    keep=[c for c in keep if c in out.columns]
    return out[keep]


def add_rb_interactions(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out=d.copy(); added=[]
    z='tr_tr_sos_z_mean'
    if z not in out:
        return out, added
    candidates=[
        'rush_EPAplay_mean','rush_yardsplay_mean','rush_rush_share_mean',
        'rec_target_share_mean','rb_scrim_yards_per_team_play_mean',
    ]
    for c in candidates:
        if c in out.columns:
            name=f'tr_sos_x_{c}'
            out[name]=pd.to_numeric(out[c],errors='coerce')*pd.to_numeric(out[z],errors='coerce')
            added.append(name)
    return out, added


def fit_variant(d: pd.DataFrame, job: dict, primary: list[str], secondary: list[str]):
    d=eval6.add_missing(d, list(dict.fromkeys(primary+secondary)))
    val_reg,val_clf=eval6.walk_forward(d,job,primary,secondary)
    final_reg,final_clf=eval6.final_2023(d,job,primary,secondary)
    return val_reg,val_clf,final_reg,final_clf


def main():
    print('Loading TeamRankings SOS history...')
    sos=load_teamrankings_history()
    print('Loading cfb season summaries...')
    pa=b.load_cfb_table('passing'); ru=b.load_cfb_table('rushing'); re_=b.load_cfb_table('receiving'); team=b.load_cfb_table('team_summaries')
    pa,ru,re_=s2.prep_college(pa,ru,re_,team)
    audits=[]
    pa,a=attach_sos(pa,team,sos,'passing'); audits.append(a)
    ru,a=attach_sos(ru,team,sos,'rushing'); audits.append(a)
    re_,a=attach_sos(re_,team,sos,'receiving'); audits.append(a)
    match_audit=pd.concat(audits,ignore_index=True)
    match_audit.to_csv(OUT/'team_name_match_audit.csv',index=False)

    hist=pd.read_csv(HIST_POOL,low_memory=False); cur=pd.read_csv(CUR_POOL,low_memory=False)
    hist['season']=pd.to_numeric(hist.season,errors='coerce'); cur['season']=pd.to_numeric(cur.season,errors='coerce')
    if 'target_valid' in hist: hist=hist[pd.to_numeric(hist.target_valid,errors='coerce').eq(1)].copy()
    hist=hist[hist.season.le(2023)].copy(); cur=cur[cur.season.eq(2026)].copy()
    master=pd.concat([hist[['season','pfr_id','pfr_name','position']],cur[['season','pfr_id','pfr_name','position']]],ignore_index=True)
    sp=aggregate_sos_profiles(master,pa,ru,re_)
    sp.to_csv(OUT/'prospect_teamrankings_sos_features.csv',index=False)
    h=hist.merge(sp,on=['season','pfr_id','pfr_name','position'],how='left')
    c=cur.merge(sp,on=['season','pfr_id','pfr_name','position'],how='left')

    model_sos=[x for x in ['tr_tr_sos_z_final','tr_tr_sos_z_mean','tr_tr_sos_z_peak','tr_tr_sos_rank_pct_final','tr_tr_sos_rank_pct_mean'] if x in h]
    metrics=[]; coverage=[]; decisions=[]
    for pos in POSITIONS:
        job=joblib.load(MODELS/f'{pos}.joblib')
        dh=h[h.position.eq(pos)].copy(); dc=c[c.position.eq(pos)].copy()
        base_p=list(job.get('features',[])); base_s=list(job.get('features_b') or [])
        variants=[('baseline',base_p,base_s)]
        variants.append(('teamrankings_sos',base_p+model_sos,base_s+model_sos if base_s else []))
        rb_int=[]
        if pos=='RB':
            dh,rb_int=add_rb_interactions(dh); dc,_=add_rb_interactions(dc)
            variants.append(('teamrankings_sos_plus_rb_interactions',base_p+model_sos+rb_int,base_s+model_sos+rb_int if base_s else []))
        for f in model_sos+rb_int:
            coverage.append({'position':pos,'feature':f,'hist_nonmissing':int(pd.to_numeric(dh.get(f),errors='coerce').notna().sum()),'hist_n':len(dh),'current_nonmissing':int(pd.to_numeric(dc.get(f),errors='coerce').notna().sum()),'current_n':len(dc)})
        for variant,pf,sf in variants:
            val_reg,val_clf,fin_reg,fin_clf=fit_variant(dh,job,pf,sf)
            metrics.append(eval6.metric_row(pos,variant,'validation_2019_2022',val_reg,val_clf,0,0))
            metrics.append(eval6.metric_row(pos,variant,'final_2023',fin_reg,fin_clf,0,0))
    m=pd.DataFrame(metrics)
    m.to_csv(OUT/'sos_ablation_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'sos_feature_coverage.csv',index=False)

    val=m[m.split.eq('validation_2019_2022')].copy()
    for pos in POSITIONS:
        z=val[val.position.eq(pos)].set_index('variant')
        b0=z.loc['baseline']
        for v in [x for x in z.index if x!='baseline']:
            r=z.loc[v]
            decisions.append({
                'position':pos,'variant':v,
                'delta_mae':r.reg_mae-b0.reg_mae,
                'delta_rmse':r.reg_rmse-b0.reg_rmse,
                'delta_spearman':r.reg_spearman-b0.reg_spearman,
                'delta_pearson':r.reg_pearson-b0.reg_pearson,
                'delta_hit_auc':r.hit_auc-b0.hit_auc,
                'delta_hit_brier':r.hit_brier-b0.hit_brier,
            })
    pd.DataFrame(decisions).to_csv(OUT/'sos_validation_deltas.csv',index=False)

    maprows=[]
    for s in range(SOS_START_SEASON,SOS_END_SEASON+1):
        maprows.append({'college_season':s,'teamrankings_source_date':f'{s+1}-01-20','draft_year_if_final_college_season':s+1})
    pd.DataFrame(maprows).to_csv(OUT/'sos_season_draft_year_mapping.csv',index=False)

    manifest={
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'experiment':'TeamRankings strength-of-schedule ablation',
        'source':'TeamRankings NCAA College Football Strength of Schedule Rankings & Ratings',
        'source_url_template':TEAMRANKINGS_URL,
        'college_season_to_draft_year_rule':'college season Y uses TeamRankings snapshot dated Y+1-01-20; a final college season Y maps to draft year Y+1',
        'selection_split':'2019-2022 walk-forward validation',
        'confirmation_split':'2023 untouched confirmation',
        'published_frozen_model_changed':False,
        'note':'SOS is a candidate feature family. It is not promoted unless validation improves; existing Stage 3 already contains SportsDataverse team_def_strength_faced context, so TeamRankings may be incremental or redundant.',
        'outputs':['teamrankings_sos_history.csv','team_name_match_audit.csv','prospect_teamrankings_sos_features.csv','sos_feature_coverage.csv','sos_ablation_metrics.csv','sos_validation_deltas.csv','sos_season_draft_year_mapping.csv'],
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
