from __future__ import annotations

import io, json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_pahowdy_context'
CACHE = ROOT / '.cache_pahowdy_context'
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

START_CFB = 2004
LAST_CFB = 2025

FINAL_SNAPSHOT = {
    2004:'2005-01-05', 2005:'2006-01-05', 2006:'2007-01-09',
    2007:'2008-01-08', 2008:'2009-01-09', 2009:'2010-01-08',
    2010:'2011-01-11', 2011:'2012-01-10', 2012:'2013-01-08',
    2013:'2014-01-07', 2014:'2015-01-13', 2015:'2016-01-12',
    2016:'2017-01-10', 2017:'2018-01-09', 2018:'2019-01-08',
    2019:'2020-01-14', 2020:'2021-01-12', 2021:'2022-01-11',
    2022:'2023-01-10', 2023:'2024-01-09', 2024:'2025-01-21',
    2025:'2026-01-20',
}

TR_URLS = {
    'sos': 'https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={date}',
    'plays_pg': 'https://www.teamrankings.com/college-football/stat/plays-per-game?date={date}',
    'opp_plays_pg': 'https://www.teamrankings.com/college-football/stat/opponent-plays-per-game?date={date}',
}

ALIASES={
 'ohiostate':'ohiost','pennstate':'pennst','floridastate':'floridast','michiganstate':'michiganst',
 'kansasstate':'kansasst','oklahomastate':'oklahomast','iowastate':'iowast','arizonastate':'arizonast',
 'mississippistate':'mississippist','coloradostate':'coloradost','fresnostate':'fresnost','sandiegostate':'sandiegost',
 'sanjosestate':'sanjosest','utahstate':'utahst','arkansasstate':'arkansasst','ballstate':'ballst','kentstate':'kentst',
 'newmexicostate':'newmexicost','georgiastate':'georgiast','washingtonstate':'washingtonst','oregonstate':'oregonst',
 'boisestate':'boisest','texasstate':'texasst','appalachianstate':'appstate','southflorida':'sflorida',
 'eastcarolina':'ecarolina','northtexas':'ntexas','westernmichigan':'wmichigan','easternmichigan':'emichigan',
 'centralmichigan':'cmichigan','northernillinois':'nillinois','southalabama':'salabama','westernkentucky':'wkentucky',
 'georgiasouthern':'georgiaso','coastalcarolina':'coastalcar','middletennessee':'middletenn',
 'floridainternational':'floridaintl','louisianamonroe':'ulmonroe','jamesmadison':'jmadison',
 'miamiohio':'miamioh','connecticut':'uconn','massachusetts':'umass','southerncalifornia':'usc',
 'olemiss':'mississippi','louisiana':'lalafayette','texassanantonio':'utsa','texaselpaso':'utep',
 'alabamabirmingham':'uab','centralflorida':'ucf','southernmethodist':'smu','texaschristian':'tcu',
 'brighamyoung':'byu','northcarolinastate':'ncstate','southernmississippi':'southernmiss',
}

S = requests.Session()
S.headers.update({'User-Agent':'Mozilla/5.0 Chrome/131 Safari/537.36','Accept-Language':'en-US,en;q=0.9'})

def norm_text(x):
    if pd.isna(x): return ''
    s=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower().replace('&','and')
    s=re.sub(r'\buniversity\b|\bthe\b',' ',s)
    return re.sub(r'[^a-z0-9]+','',s)

def norm_team(x):
    s=norm_text(x)
    return ALIASES.get(s,s)

def norm_name(x):
    if pd.isna(x): return ''
    s=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower()
    s=re.sub(r'\b(jr|sr|ii|iii|iv|v)\b','',s)
    return re.sub(r'[^a-z0-9]+','',s)

def get(url, timeout=90):
    r=S.get(url,timeout=timeout); r.raise_for_status(); return r

def release_assets(tag):
    fp=CACHE/f'{tag}_assets.json'
    if fp.exists(): return json.loads(fp.read_text())
    assets={a['name']:a['browser_download_url'] for a in get(f'https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/{tag}').json().get('assets',[])}
    fp.write_text(json.dumps(assets)); return assets

def load_cfb(kind):
    assets=release_assets(f'espn_cfb_{kind}')
    frames=[]
    for y in range(START_CFB,LAST_CFB+1):
        name=next((n for n in (f'cfb_{kind}_{y}.csv',f'{kind}_{y}.csv') if n in assets),None)
        if not name: continue
        fp=CACHE/name
        if not fp.exists(): fp.write_bytes(get(assets[name],180).content)
        d=pd.read_csv(fp,low_memory=False); d['season']=pd.to_numeric(d.get('season',y),errors='coerce').fillna(y).astype(int)
        frames.append(d)
    return pd.concat(frames,ignore_index=True,sort=False)

def first_col(d, names): return next((c for c in names if c in d.columns),None)

def numeric(d,c): return pd.to_numeric(d[c],errors='coerce') if c and c in d else pd.Series(0,index=d.index,dtype=float)

def player_team_seasons(kind,d):
    if kind=='passing': namecol=first_col(d,['passer_player_name','player_name','athlete_display_name'])
    elif kind=='rushing': namecol=first_col(d,['rusher_player_name','player_name','athlete_display_name'])
    else: namecol=first_col(d,['receiver_player_name','player_name','athlete_display_name'])
    teamcol=first_col(d,['pos_team','team','team_name','school','team_abbreviation'])
    if not namecol or not teamcol: raise RuntimeError(f'{kind}: could not identify name/team columns')
    x=d.copy(); x['player_name']=x[namecol].astype(str); x['player_norm']=x.player_name.map(norm_name)
    x['college_team']=x[teamcol].astype(str); x['team_key']=x.college_team.map(norm_team)
    x=x[(x.player_norm!='')&(x.team_key!='')].copy()
    x['yards_num']=numeric(x,first_col(x,['yards']))
    if kind=='passing':
        x['pass_yards']=x.yards_num; x['pass_att']=numeric(x,first_col(x,['att','attempts'])); x['rush_yards']=0.; x['rush_att']=0.; x['rec_yards']=0.; x['targets_num']=0.
    elif kind=='rushing':
        x['rush_yards']=x.yards_num; x['rush_att']=numeric(x,first_col(x,['plays','att','attempts'])); x['pass_yards']=0.; x['pass_att']=0.; x['rec_yards']=0.; x['targets_num']=0.
    else:
        x['rec_yards']=x.yards_num; x['targets_num']=numeric(x,first_col(x,['targets'])); x['pass_yards']=0.; x['pass_att']=0.; x['rush_yards']=0.; x['rush_att']=0.
    keep=['season','player_name','player_norm','college_team','team_key','pass_yards','pass_att','rush_yards','rush_att','rec_yards','targets_num']
    return x[keep].groupby(['season','player_norm','college_team','team_key'],as_index=False).agg({
        'player_name':'last','pass_yards':'sum','pass_att':'sum','rush_yards':'sum','rush_att':'sum','rec_yards':'sum','targets_num':'sum'})

def parse_tr(metric, season, date):
    url=TR_URLS[metric].format(date=date)
    fp=CACHE/f'tr_{metric}_{season}_{date}.html'
    text=fp.read_text(errors='ignore') if fp.exists() else get(url,60).text
    if not fp.exists(): fp.write_text(text)
    soup=BeautifulSoup(text,'html.parser')
    best=[]
    for table in soup.find_all('table'):
        headers=[re.sub(r'\s+',' ',h.get_text(' ',strip=True)) for h in table.find_all('th')]
        hs=' | '.join(headers).lower()
        if 'team' not in hs or 'rank' not in hs: continue
        rows=[]
        for tr in table.find_all('tr'):
            cells=tr.find_all('td')
            if len(cells)<3: continue
            vals=[re.sub(r'\s+',' ',c.get_text(' ',strip=True)) for c in cells]
            try: rank=int(re.sub(r'[^0-9]','',vals[0]))
            except: continue
            team=re.sub(r'\s*\([0-9]+-[0-9]+(?:-[0-9]+)?\)\s*$','',vals[1]).strip()
            try: value=float(vals[2])
            except: continue
            rows.append({'college_season':season,'source_date':date,'metric':metric,'rank':rank,'teamrankings_team':team,'team_key':norm_team(team),'value':value,'source_url':url})
        if len(rows)>len(best): best=rows
    if len(best)<70: raise RuntimeError(f'{metric} {season} parsed only {len(best)} teams')
    return pd.DataFrame(best).drop_duplicates(['college_season','team_key'])

def main():
    # TeamRankings full-season tables.
    metric_frames=[]; audit=[]
    for sy,date in FINAL_SNAPSHOT.items():
        for metric in ['sos','plays_pg','opp_plays_pg']:
            try:
                d=parse_tr(metric,sy,date); metric_frames.append(d)
                audit.append({'college_season':sy,'source_date':date,'metric':metric,'teams':len(d),'min':d.value.min(),'max':d.value.max(),'status':'ok'})
                print(sy,metric,len(d))
            except Exception as e:
                audit.append({'college_season':sy,'source_date':date,'metric':metric,'teams':0,'min':np.nan,'max':np.nan,'status':repr(e)})
                print('MISS',sy,metric,e)
    long=pd.concat(metric_frames,ignore_index=True)
    wide=long.pivot_table(index=['college_season','source_date','teamrankings_team','team_key'],columns='metric',values='value',aggfunc='first').reset_index()
    # Because team names can differ trivially between the three pages, merge again by season/key.
    base=long[['college_season','source_date','teamrankings_team','team_key']].drop_duplicates(['college_season','team_key'])
    out=base.copy()
    for metric in ['sos','plays_pg','opp_plays_pg']:
        m=long[long.metric.eq(metric)][['college_season','team_key','value','rank']].rename(columns={'value':metric,'rank':f'{metric}_rank'})
        out=out.merge(m,on=['college_season','team_key'],how='left')
    for metric in ['sos','plays_pg','opp_plays_pg']:
        vals=pd.to_numeric(out[metric],errors='coerce')
        out[f'{metric}_z']=out.groupby('college_season')[metric].transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0)
        out[f'{metric}_pct']=out.groupby('college_season')[metric].rank(pct=True,method='average')
    out['total_game_plays_pg']=out.plays_pg+out.opp_plays_pg
    out['team_pace_factor_vs_median']=out.groupby('college_season').plays_pg.transform('median')/out.plays_pg
    out['game_pace_factor_vs_median']=out.groupby('college_season').total_game_plays_pg.transform('median')/out.total_game_plays_pg
    out.to_csv(OUT/'teamrankings_full_season_sos_pace_2004_2025.csv',index=False)
    long.to_csv(OUT/'teamrankings_full_page_rows_long.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'teamrankings_full_page_audit.csv',index=False)

    # SportsDataverse player/team/season rows, preserving transfers.
    frames=[]
    for kind in ['passing','rushing','receiving']:
        d=load_cfb(kind); frames.append(player_team_seasons(kind,d))
    pts=pd.concat(frames,ignore_index=True)
    pts=pts.groupby(['season','player_norm','college_team','team_key'],as_index=False).agg({
        'player_name':'last','pass_yards':'sum','pass_att':'sum','rush_yards':'sum','rush_att':'sum','rec_yards':'sum','targets_num':'sum'})
    # Match each player-team-season directly to that exact season's TeamRankings row.
    context=out.rename(columns={'college_season':'season'})
    pts=pts.merge(context,on=['season','team_key'],how='left',suffixes=('','_tr'))
    pts['context_matched']=pts.sos.notna()|pts.plays_pg.notna()|pts.opp_plays_pg.notna()
    pts.to_csv(OUT/'player_team_season_sos_pace_2004_2025.csv',index=False)

    coverage=pts.groupby('season').agg(player_team_rows=('player_norm','size'),matched=('context_matched','sum')).reset_index()
    coverage['match_rate']=coverage.matched/coverage.player_team_rows
    coverage.to_csv(OUT/'player_team_season_context_coverage.csv',index=False)
    print(coverage.to_string(index=False))

if __name__=='__main__': main()
