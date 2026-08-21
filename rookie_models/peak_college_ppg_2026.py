from __future__ import annotations
import io, json, re, unicodedata
from pathlib import Path
import pandas as pd
import requests
from rapidfuzz import process, fuzz

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_pahowdy_context'
OUT.mkdir(exist_ok=True)
S=requests.Session(); S.headers.update({'User-Agent':'rookie-peak-ppg/1.0'})


def get(url,timeout=120):
    r=S.get(url,timeout=timeout); r.raise_for_status(); return r

def norm_name(x):
    if pd.isna(x): return ''
    x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower()
    x=re.sub(r'\b(jr|sr|ii|iii|iv|v)\b','',x)
    return re.sub(r'[^a-z0-9]','',x)

def release_assets(tag):
    j=get(f'https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/{tag}').json()
    return {a['name']:a['browser_download_url'] for a in j.get('assets',[])}

def load_kind(kind, years=range(2020,2026)):
    assets=release_assets(f'espn_cfb_{kind}')
    frames=[]
    for y in years:
        name=next((n for n in (f'cfb_{kind}_{y}.csv',f'{kind}_{y}.csv') if n in assets),None)
        if not name: continue
        d=pd.read_csv(io.BytesIO(get(assets[name],180).content),low_memory=False)
        d['season']=pd.to_numeric(d.get('season',y),errors='coerce').fillna(y).astype(int)
        frames.append(d)
    return pd.concat(frames,ignore_index=True,sort=False)

def num(d,c):
    return pd.to_numeric(d[c],errors='coerce') if c in d else pd.Series(0.0,index=d.index)

def prep(d,namecol):
    d=d.copy(); d['name_norm']=d[namecol].map(norm_name)
    return d

def aggregate(d, cols):
    if d.empty: return pd.DataFrame(columns=['season','name_norm']+cols+['games'])
    ag={c:'sum' for c in cols if c in d.columns}
    if 'games' in d.columns: ag['games']='max'
    z=d.groupby(['season','name_norm'],as_index=False).agg(ag)
    if 'games' not in z: z['games']=0
    return z

def main():
    pool=pd.read_csv(ROOT/'results_2026'/'prospect_pool_2026.csv',low_memory=False)
    pool=pool[['pfr_name','position']].drop_duplicates().copy(); pool['name_norm']=pool.pfr_name.map(norm_name)
    p=prep(load_kind('passing'),'passer_player_name')
    r=prep(load_kind('rushing'),'rusher_player_name')
    c=prep(load_kind('receiving'),'receiver_player_name')
    pa=aggregate(p,['yards','passing_td','pass_int']).rename(columns={'yards':'pass_yards','passing_td':'pass_td','pass_int':'pass_int','games':'pass_games'})
    ra=aggregate(r,['yards','rushing_td']).rename(columns={'yards':'rush_yards','rushing_td':'rush_td','games':'rush_games'})
    ca=aggregate(c,['yards','passing_td','comp']).rename(columns={'yards':'rec_yards','passing_td':'rec_td','comp':'receptions','games':'rec_games'})
    all_names=set(pa.name_norm)|set(ra.name_norm)|set(ca.name_norm)
    rows=[]
    for _,pr in pool.iterrows():
        nn=pr.name_norm
        key=nn if nn in all_names else None
        method='exact'
        if key is None:
            m=process.extractOne(nn,list(all_names),scorer=fuzz.ratio,score_cutoff=91)
            if m: key=m[0]; method='fuzzy'
        if key is None:
            rows.append({'pfr_name':pr.pfr_name,'position':pr.position,'peak_college_season':None,'peak_college_ppg':None,'match_method':'unmatched'}); continue
        seasons=sorted(set(pa.loc[pa.name_norm==key,'season'])|set(ra.loc[ra.name_norm==key,'season'])|set(ca.loc[ca.name_norm==key,'season']))
        cand=[]
        for y in seasons:
            if y>2025: continue
            pp=pa[(pa.name_norm==key)&(pa.season==y)]
            rr=ra[(ra.name_norm==key)&(ra.season==y)]
            cc=ca[(ca.name_norm==key)&(ca.season==y)]
            def one(df,col): return float(df[col].iloc[0]) if (not df.empty and col in df and pd.notna(df[col].iloc[0])) else 0.0
            games=max(one(pp,'pass_games'),one(rr,'rush_games'),one(cc,'rec_games'))
            if games<=0: continue
            # Pahowdy labels college PPG as PPR scoring. Historical QB rows align closely with 5-pt passing TD scoring.
            pts=(one(pp,'pass_yards')*0.04 + one(pp,'pass_td')*5 - one(pp,'pass_int')*2 +
                 one(rr,'rush_yards')*0.1 + one(rr,'rush_td')*6 +
                 one(cc,'rec_yards')*0.1 + one(cc,'rec_td')*6 + one(cc,'receptions'))
            cand.append((pts/games,y,pts,games))
        if not cand:
            rows.append({'pfr_name':pr.pfr_name,'position':pr.position,'peak_college_season':None,'peak_college_ppg':None,'match_method':'matched_no_games'}); continue
        cand.sort(reverse=True)
        ppg,y,pts,games=cand[0]
        rows.append({'pfr_name':pr.pfr_name,'position':pr.position,'peak_college_season':int(y),'peak_college_ppg':round(ppg,3),'match_method':method,'matched_name_norm':key,'peak_points':round(pts,2),'peak_games':int(games)})
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'peak_college_ppg_2026.csv',index=False)
    print(out[['position','match_method']].value_counts(dropna=False))
    print('rows',len(out),'peak years',out.peak_college_season.value_counts(dropna=False).sort_index().to_dict())

if __name__=='__main__': main()
