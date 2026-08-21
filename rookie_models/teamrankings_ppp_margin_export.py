from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_pahowdy_context'
OUT.mkdir(exist_ok=True)
URL='https://www.teamrankings.com/college-football/stat/points-per-play-margin?date={date}'
FINAL_SNAPSHOT={
2004:'2005-01-05',2005:'2006-01-05',2006:'2007-01-09',2007:'2008-01-08',2008:'2009-01-09',2009:'2010-01-08',2010:'2011-01-11',2011:'2012-01-10',2012:'2013-01-08',2013:'2014-01-07',2014:'2015-01-13',2015:'2016-01-12',2016:'2017-01-10',2017:'2018-01-09',2018:'2019-01-08',2019:'2020-01-14',2020:'2021-01-12',2021:'2022-01-11',2022:'2023-01-10',2023:'2024-01-09',2024:'2025-01-21',2025:'2026-01-20'}
S=requests.Session();S.headers.update({'User-Agent':'Mozilla/5.0','Accept-Language':'en-US,en;q=0.9'})

def norm(x):
 s=str(x).lower().replace('&',' and ');s=re.sub(r'\([^)]*\)$','',s).strip();return re.sub(r'[^a-z0-9]+','',s)

def parse(season,date):
 r=S.get(URL.format(date=date),timeout=75);r.raise_for_status();soup=BeautifulSoup(r.text,'html.parser')
 rows=[];headers_seen=None
 for table in soup.find_all('table'):
  headers=[re.sub(r'\s+',' ',th.get_text(' ',strip=True)) for th in table.find_all('th')]
  hlow=[h.lower() for h in headers]
  if 'team' not in hlow or 'rank' not in hlow: continue
  headers_seen=headers
  team_i=hlow.index('team');rank_i=hlow.index('rank')
  value_i=headers.index(str(season)) if str(season) in headers else 2
  for tr in table.find_all('tr'):
   cells=[re.sub(r'\s+',' ',td.get_text(' ',strip=True)) for td in tr.find_all('td')]
   if len(cells)<=max(team_i,rank_i,value_i): continue
   try:
    rank=int(re.sub(r'[^0-9]','',cells[rank_i])); val=float(cells[value_i])
   except Exception: continue
   team=re.sub(r'\s*\([0-9]+-[0-9]+(?:-[0-9]+)?\)\s*$','',cells[team_i]).strip()
   rows.append({'college_season':season,'source_date':date,'teamrankings_team':team,'team_key':norm(team),'points_per_play_margin':val,'points_per_play_margin_rank':rank,'source_url':URL.format(date=date)})
  if rows: break
 d=pd.DataFrame(rows).drop_duplicates(['college_season','team_key'])
 if len(d)<80: raise RuntimeError(f'PPP margin {season} parsed only {len(d)} rows; headers={headers_seen}')
 if not d['points_per_play_margin'].between(-2,2).all(): raise RuntimeError(f'PPP margin {season} out of expected range')
 return d

def main():
 frames=[];audit=[]
 for season,date in FINAL_SNAPSHOT.items():
  d=parse(season,date);frames.append(d)
  audit.append({'college_season':season,'source_date':date,'teams':len(d),'min':d.points_per_play_margin.min(),'max':d.points_per_play_margin.max(),'status':'ok'})
  print(season,len(d),d.points_per_play_margin.min(),d.points_per_play_margin.max())
 pd.concat(frames,ignore_index=True).to_csv(OUT/'teamrankings_points_per_play_margin_2004_2025.csv',index=False)
 pd.DataFrame(audit).to_csv(OUT/'teamrankings_points_per_play_margin_audit.csv',index=False)
if __name__=='__main__':main()
