from pathlib import Path
import re, unicodedata
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_stage6_sos'; OUT.mkdir(exist_ok=True)
SOS=OUT/'teamrankings_sos_final_history.csv'
HIST=ROOT/'results_v3'/'prospect_pool_v3.csv'
CUR=ROOT/'results_2026'/'prospect_pool_2026.csv'

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
 'miamiohio':'miamioh','hawaii':'hawaii','connecticut':'uconn','massachusetts':'umass',
 'southerncalifornia':'usc','olemiss':'mississippi','louisiana':'lalafayette','texassanantonio':'utsa',
 'texaselpaso':'utep','alabamabirmingham':'uab','centralflorida':'ucf','southernmethodist':'smu',
 'texaschristian':'tcu','brighamyoung':'byu','northcarolinastate':'ncstate'
}

def norm(x):
 if pd.isna(x): return ''
 s=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower().replace('&','and')
 s=re.sub(r'\buniversity\b|\bthe\b',' ',s)
 s=re.sub(r'[^a-z0-9]+','',s)
 return ALIASES.get(s,s)

def first_school(x):
 if pd.isna(x): return ''
 return str(x).split(';')[0].strip()

def attach(d,sos):
 d=d.copy();d['college_season']=pd.to_numeric(d.season,errors='coerce')-1
 d['final_school']=d.college_name.map(first_school)
 d['school_key']=d.final_school.map(norm)
 rows=[]
 for sy,g in d.groupby('college_season'):
  ss=sos[sos.college_season.eq(int(sy))].copy()
  keys=ss.team_key.tolist();keyset=set(keys)
  for raw,key in g[['final_school','school_key']].drop_duplicates().itertuples(index=False):
   match=key if key in keyset else None;score=100.;method='exact'
   if match is None and key:
    z=process.extractOne(key,keys,scorer=fuzz.ratio,score_cutoff=88)
    if z: match,score,_=z;method='fuzzy'
   rows.append({'college_season':int(sy),'final_school':raw,'school_key':key,'matched_team_key':match,'match_score':score if match else np.nan,'match_method':method if match else 'unmatched'})
 mp=pd.DataFrame(rows).drop_duplicates(['college_season','school_key'])
 d=d.merge(mp[['college_season','school_key','matched_team_key','match_method','match_score']],on=['college_season','school_key'],how='left')
 ss=sos[['college_season','team_key','tr_sos_rating','tr_sos_z','tr_sos_rank_pct']]
 d=d.merge(ss,left_on=['college_season','matched_team_key'],right_on=['college_season','team_key'],how='left')
 return d,mp

def main():
 sos=pd.read_csv(SOS,low_memory=False)
 sos['college_season']=pd.to_numeric(sos.college_season,errors='coerce').astype(int)
 for src,label in [(HIST,'historical'),(CUR,'current')]:
  d=pd.read_csv(src,low_memory=False)
  d['season']=pd.to_numeric(d.season,errors='coerce')
  if label=='historical': d=d[d.season.between(2014,2023)].copy()
  else: d=d[d.season.eq(2026)].copy()
  out,mp=attach(d,sos)
  cols=['season','pfr_id','pfr_name','position','college_name','final_school','college_season','tr_sos_rating','tr_sos_z','tr_sos_rank_pct','match_method','match_score']
  out[cols].to_csv(OUT/f'prospect_final_school_sos_{label}.csv',index=False)
  mp.to_csv(OUT/f'final_school_sos_match_audit_{label}.csv',index=False)
  print(label,len(out),'matched',int(out.tr_sos_z.notna().sum()),'rate',float(out.tr_sos_z.notna().mean()))
if __name__=='__main__':main()
