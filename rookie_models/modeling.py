from __future__ import annotations
import io, json, math, os, re, unicodedata, warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests
from rapidfuzz import process, fuzz
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
MODELS = ROOT / 'trained'
CACHE = ROOT / '.cache'
for p in (OUT, MODELS, CACHE): p.mkdir(parents=True, exist_ok=True)

START_CFB = 2004
LAST_CFB = 2025
TRAIN_DRAFT_START = 2005
TRAIN_DRAFT_END = 2023
PREDICT_DRAFT_YEAR = 2026
DEV_TEST_YEARS = list(range(2013, 2020))
HOLDOUT_YEARS = list(range(2020, 2024))
POSITIONS = ['QB','RB','WR','TE']
HIT_RANK = {'QB':12,'RB':12,'WR':24,'TE':12}

S = requests.Session()
S.headers.update({'User-Agent':'rookie-dynasty-model/1.0'})

def get(url, timeout=90):
    r=S.get(url,timeout=timeout); r.raise_for_status(); return r

def read_csv_url(url, **kw):
    return pd.read_csv(io.BytesIO(get(url).content), low_memory=False, **kw)

def norm_name(x):
    if pd.isna(x): return ''
    x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower()
    x=re.sub(r'\b(jr|sr|ii|iii|iv|v)\b','',x)
    x=re.sub(r'[^a-z0-9]','',x)
    return x

def release_assets(tag):
    fp=CACHE/f'{tag}_assets.json'
    if fp.exists(): return json.loads(fp.read_text())
    j=get(f'https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/{tag}').json()
    assets={a['name']:a['browser_download_url'] for a in j.get('assets',[])}
    fp.write_text(json.dumps(assets))
    return assets

def load_cfb_table(kind):
    tag=f'espn_cfb_{kind}'
    assets=release_assets(tag)
    frames=[]
    for y in range(START_CFB,LAST_CFB+1):
        names=[f'cfb_{kind}_{y}.csv',f'{kind}_{y}.csv']
        name=next((n for n in names if n in assets),None)
        if not name: continue
        fp=CACHE/name
        if not fp.exists(): fp.write_bytes(get(assets[name],timeout=180).content)
        d=pd.read_csv(fp,low_memory=False); d['season']=pd.to_numeric(d.get('season',y),errors='coerce').fillna(y).astype(int)
        frames.append(d)
    if not frames: raise RuntimeError(f'No cfb {kind} data')
    return pd.concat(frames,ignore_index=True,sort=False)

def load_nflverse():
    draft=read_csv_url('https://raw.githubusercontent.com/nflverse/nfldata/master/data/draft_picks.csv')
    players=read_csv_url('https://github.com/nflverse/nflverse-data/releases/download/players/players.csv')
    combine=read_csv_url('https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv')
    weeks=[]
    for y in range(TRAIN_DRAFT_START,2026):
        url=f'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{y}.csv'
        try:
            d=read_csv_url(url); d['season']=y; weeks.append(d)
        except Exception as e: print('weekly miss',y,e)
    weekly=pd.concat(weeks,ignore_index=True,sort=False)
    return draft,players,combine,weekly

def first_existing(df, names):
    return next((c for c in names if c in df.columns),None)

def numeric(df, col):
    return pd.to_numeric(df[col],errors='coerce') if col in df else pd.Series(np.nan,index=df.index)

def prep_cfb(pass_df,rush_df,rec_df):
    for d,nc in [(pass_df,'passer_player_name'),(rush_df,'rusher_player_name'),(rec_df,'receiver_player_name')]:
        d['name_norm']=d[nc].map(norm_name)
        d['player_id']=pd.to_numeric(d.get('player_id'),errors='coerce')
    # team denominators by season/team
    rk=['season','team_id'] if 'team_id' in rush_df and 'team_id' in rec_df else ['season','pos_team']
    rec_team=rec_df.groupby(rk,dropna=False).agg(team_targets=('targets','sum'),team_rec_yards=('yards','sum'),team_rec_td=('passing_td','sum'),team_receptions=('comp','sum')).reset_index()
    rush_team=rush_df.groupby(rk,dropna=False).agg(team_rushes=('plays','sum'),team_rush_yards=('yards','sum'),team_rush_td=('rushing_td','sum')).reset_index()
    pass_team=pass_df.groupby(rk,dropna=False).agg(team_pass_att=('att','sum'),team_dropbacks=('dropbacks','sum')).reset_index()
    rec_df=rec_df.merge(rec_team,on=rk,how='left').merge(pass_team,on=rk,how='left')
    rush_df=rush_df.merge(rush_team,on=rk,how='left').merge(pass_team,on=rk,how='left')
    # receiving metrics
    rec_df['target_share']=numeric(rec_df,'targets')/numeric(rec_df,'team_targets').replace(0,np.nan)
    rec_df['rec_yard_share']=numeric(rec_df,'yards')/numeric(rec_df,'team_rec_yards').replace(0,np.nan)
    rec_df['rec_td_share']=numeric(rec_df,'passing_td')/numeric(rec_df,'team_rec_td').replace(0,np.nan)
    rec_df['rec_per_team_pass_att']=numeric(rec_df,'yards')/numeric(rec_df,'team_pass_att').replace(0,np.nan)
    rec_df['yards_per_target']=numeric(rec_df,'yards')/numeric(rec_df,'targets').replace(0,np.nan)
    rec_df['td_per_target']=numeric(rec_df,'passing_td')/numeric(rec_df,'targets').replace(0,np.nan)
    # rushing metrics
    rush_df['rush_share']=numeric(rush_df,'plays')/numeric(rush_df,'team_rushes').replace(0,np.nan)
    rush_df['rush_yard_share']=numeric(rush_df,'yards')/numeric(rush_df,'team_rush_yards').replace(0,np.nan)
    rush_df['rush_td_share']=numeric(rush_df,'rushing_td')/numeric(rush_df,'team_rush_td').replace(0,np.nan)
    return pass_df,rush_df,rec_df

def combine_name_season(d,namecol,volcols):
    # Aggregate transfers / duplicate team rows for same player-season.
    if d.empty:return d
    keys=['season','name_norm']
    keepcols=[c for c in d.columns if c not in keys]
    rows=[]
    for (season,nn),g in d.groupby(keys,dropna=False):
        r={'season':season,'name_norm':nn,namecol:g[namecol].dropna().iloc[-1] if g[namecol].notna().any() else nn}
        for c in keepcols:
            if c==namecol: continue
            if c in volcols: r[c]=pd.to_numeric(g[c],errors='coerce').sum(min_count=1)
            elif pd.api.types.is_numeric_dtype(g[c]):
                w=pd.to_numeric(g.get('plays',pd.Series(1,index=g.index)),errors='coerce').fillna(0).clip(lower=0)
                v=pd.to_numeric(g[c],errors='coerce')
                r[c]=np.average(v.fillna(v.mean() if v.notna().any() else 0),weights=w+1e-6) if v.notna().any() else np.nan
            else: r[c]=g[c].dropna().iloc[-1] if g[c].notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def fuzzy_lookup(nn, candidates):
    if not nn or not candidates:return None
    m=process.extractOne(nn,candidates,scorer=fuzz.ratio,score_cutoff=93)
    return m[0] if m else None

def career_features(g,prefix,base_metrics,draft_year):
    if g is None or g.empty:return {}
    g=g.sort_values('season').copy(); g=g[g.season < draft_year]
    if g.empty:return {}
    # restrict to six pre-draft years; protects same-name collisions across eras
    g=g[g.season>=draft_year-6]
    if g.empty:return {}
    out={f'{prefix}_seasons':len(g),f'{prefix}_first_season':g.season.min(),f'{prefix}_last_season':g.season.max()}
    out[f'{prefix}_years_before_first']=draft_year-g.season.min()
    out[f'{prefix}_years_before_last']=draft_year-g.season.max()
    for c in base_metrics:
        if c not in g: continue
        v=pd.to_numeric(g[c],errors='coerce')
        if not v.notna().any(): continue
        out[f'{prefix}_{c}_final']=v.iloc[-1]
        out[f'{prefix}_{c}_peak']=v.max()
        out[f'{prefix}_{c}_mean']=v.mean()
        if len(v.dropna())>=2:
            x=np.arange(len(v))[v.notna()]
            out[f'{prefix}_{c}_slope']=np.polyfit(x,v.dropna(),1)[0]
    return out

def build_college_profiles(draft,pass_df,rush_df,rec_df):
    pass_df=combine_name_season(pass_df,'passer_player_name',{'plays','games','team_games','TEPA','yards','comp','att','passing_td','sacked','sack_yds','pass_int','dropbacks'})
    rush_df=combine_name_season(rush_df,'rusher_player_name',{'plays','games','team_games','TEPA','yards','rushing_td','fumbles'})
    rec_df=combine_name_season(rec_df,'receiver_player_name',{'plays','games','team_games','TEPA','yards','comp','targets','passing_td','fumbles','team_targets','team_rec_yards','team_rec_td','team_receptions','team_pass_att','team_dropbacks'})
    pmap={k:g for k,g in pass_df.groupby('name_norm')}; rmap={k:g for k,g in rush_df.groupby('name_norm')}; cmap={k:g for k,g in rec_df.groupby('name_norm')}
    pcand=list(pmap); rcand=list(rmap); ccand=list(cmap)
    pmetrics=['playsgame','EPAplay','EPAgame','yardsplay','yardsgame','success','comppct','passing_td','sacked','pass_int','yardsdropback','dropbacks']
    rmetrics=['playsgame','EPAplay','EPAgame','yardsplay','yardsgame','success','rushing_td','fumbles','rush_share','rush_yard_share','rush_td_share']
    cmetrics=['playsgame','EPAplay','EPAgame','yardsplay','yardsgame','success','catchpct','passing_td','targets','target_share','rec_yard_share','rec_td_share','rec_per_team_pass_att','yards_per_target','td_per_target']
    rows=[]; matched=0
    eligible=draft[(draft.season.between(TRAIN_DRAFT_START,PREDICT_DRAFT_YEAR)) & (draft.category.isin(POSITIONS))].copy()
    eligible['name_norm']=eligible.pfr_name.map(norm_name)
    for _,d in eligible.iterrows():
        nn=d.name_norm; pos=d.category; year=int(d.season); feat={}
        pools=[]
        if pos=='QB': pools=[('pass',pmap,pcand,pmetrics)]
        elif pos=='RB': pools=[('rush',rmap,rcand,rmetrics),('rec',cmap,ccand,cmetrics)]
        else: pools=[('rec',cmap,ccand,cmetrics)]
        anymatch=False
        for pre,mp,cands,mets in pools:
            key=nn if nn in mp else fuzzy_lookup(nn,cands)
            g=mp.get(key) if key else None
            if g is not None:
                # only count as plausible if profile overlaps pre-draft period
                gg=g[(g.season<year)&(g.season>=year-6)]
                if not gg.empty:
                    feat.update(career_features(gg,pre,mets,year)); anymatch=True
        if anymatch:matched+=1
        feat.update({'season':year,'pfr_id':d.get('pfr_id'),'pfr_name':d.pfr_name,'position':pos,'draft_round':d.get('round'),'draft_pick':d.get('pick'),'college_match':int(anymatch)})
        # cross-table RB scrimmage features by season where both exist
        if pos=='RB':
            kr=nn if nn in rmap else fuzzy_lookup(nn,rcand); kc=nn if nn in cmap else fuzzy_lookup(nn,ccand)
            rg=rmap.get(kr,pd.DataFrame()); cg=cmap.get(kc,pd.DataFrame())
            if not rg.empty:
                m=rg[['season','yards','plays','yardsgame','EPAplay','success']].rename(columns={'yards':'rush_yards','plays':'rush_plays','yardsgame':'rush_ypg','EPAplay':'rush_epa','success':'rush_success'})
                if not cg.empty:
                    cc=cg[['season','yards','targets','yardsgame','target_share','rec_yard_share','rec_per_team_pass_att','EPAplay','success']].rename(columns={'yards':'rec_yards','yardsgame':'rec_ypg','EPAplay':'rec_epa','success':'rec_success'})
                    m=m.merge(cc,on='season',how='left')
                m=m[(m.season<year)&(m.season>=year-6)].copy()
                if not m.empty:
                    m['scrim_yards']=m.rush_yards.fillna(0)+m.get('rec_yards',0).fillna(0)
                    m['scrim_ypg']=m.rush_ypg.fillna(0)+m.get('rec_ypg',0).fillna(0)
                    for c in ['scrim_yards','scrim_ypg']:
                        feat[f'rb_{c}_final']=m[c].iloc[-1]; feat[f'rb_{c}_peak']=m[c].max(); feat[f'rb_{c}_mean']=m[c].mean()
        rows.append(feat)
    print('college match',matched,'/',len(eligible),matched/max(1,len(eligible)))
    return pd.DataFrame(rows)

def add_nfl_metadata(profiles,players,combine):
    p=profiles.copy()
    # players mapping by PFR id
    pfrcol=first_existing(players,['pfr_id','pfr_player_id'])
    if pfrcol:
        keep=[c for c in [pfrcol,'gsis_id','birth_date','height','weight','college_name','draft_year','draft_round','draft_pick'] if c in players]
        q=players[keep].drop_duplicates(pfrcol).rename(columns={pfrcol:'pfr_id'})
        p=p.merge(q,on='pfr_id',how='left',suffixes=('','_player'))
    # combine by pfr id, else normalized name/year
    if 'pfr_id' in combine.columns:
        cols=[c for c in ['pfr_id','season','ht','wt','forty','vertical','bench','broad_jump','cone','shuttle'] if c in combine]
        p=p.merge(combine[cols].drop_duplicates('pfr_id'),on='pfr_id',how='left',suffixes=('','_combine'))
    else:
        ncol=first_existing(combine,['player_name','name','pfr_name'])
        if ncol:
            combine=combine.copy();combine['name_norm']=combine[ncol].map(norm_name)
            cols=[c for c in ['name_norm','season','ht','wt','forty','vertical','bench','broad_jump','cone','shuttle'] if c in combine]
            p['name_norm']=p.pfr_name.map(norm_name);p=p.merge(combine[cols],on=['name_norm','season'],how='left')
    # age
    bcol=first_existing(p,['birth_date'])
    if bcol:
        bd=pd.to_datetime(p[bcol],errors='coerce'); draft_day=pd.to_datetime(p.season.astype(str)+'-04-25',errors='coerce')
        p['draft_age']=(draft_day-bd).dt.days/365.25
    # physicals
    for c in ['weight','wt','forty','vertical','bench','broad_jump','cone','shuttle','height']:
        if c in p: p[c]=pd.to_numeric(p[c],errors='coerce')
    w=p['wt'] if 'wt' in p else p.get('weight',pd.Series(np.nan,index=p.index))
    if 'forty' in p: p['speed_score']=w*(200/p.forty.clip(lower=4))**4
    p['log_pick']=np.log(pd.to_numeric(p.draft_pick,errors='coerce').clip(lower=1))
    p['pick_inv_sqrt']=1/np.sqrt(pd.to_numeric(p.draft_pick,errors='coerce').clip(lower=1))
    p['day1']=(pd.to_numeric(p.draft_round,errors='coerce')==1).astype(float)
    p['day2']=(pd.to_numeric(p.draft_round,errors='coerce').isin([2,3])).astype(float)
    return p

def nfl_targets(profiles,weekly):
    p=profiles.copy()
    idcol=first_existing(weekly,['player_id','gsis_id'])
    if not idcol: raise RuntimeError('No NFL player id in weekly stats')
    weekly=weekly.copy();weekly['season']=pd.to_numeric(weekly.season,errors='coerce')
    fpcol=first_existing(weekly,['fantasy_points_ppr','fantasy_points'])
    if not fpcol: raise RuntimeError('No fantasy points in weekly stats')
    weekly['fp']=pd.to_numeric(weekly[fpcol],errors='coerce').fillna(0)
    for c in ['attempts','carries','targets','rushing_attempts','passing_attempts']:
        if c in weekly: weekly[c]=pd.to_numeric(weekly[c],errors='coerce').fillna(0)
    outs=[]
    for _,r in p.iterrows():
        gid=r.get('gsis_id'); pos=r.position; dy=int(r.season)
        w=weekly[(weekly[idcol]==gid)&(weekly.season.between(dy,dy+2))].copy() if pd.notna(gid) else pd.DataFrame()
        peak=np.nan; avg=np.nan; total=0; hit=0; seasons=[]
        if not w.empty:
            if pos=='QB': opp=w.get('attempts',w.get('passing_attempts',0))+w.get('carries',w.get('rushing_attempts',0)); meaningful=opp>=15
            elif pos=='RB': meaningful=(w.get('carries',0)+w.get('targets',0))>=5
            else: meaningful=(w.get('targets',0)+w.get('carries',0))>=4
            w['meaningful']=meaningful; total=w.fp.sum()
            for sy,g in w.groupby('season'):
                mg=g[g.meaningful]
                if len(mg)>=4: seasons.append({'season':int(sy),'ppg':mg.fp.mean(),'games':len(mg),'points':mg.fp.sum()})
            if seasons:
                peak=max(x['ppg'] for x in seasons); avg=sum(x['points'] for x in seasons)/sum(x['games'] for x in seasons)
        outs.append((peak,avg,total))
    p[['peak3_ppg','avg3_ppg','total3_ppr']]=pd.DataFrame(outs,index=p.index)
    # hit labels are positional ranks within each NFL season for qualifying meaningful PPG
    season_ranks={}
    for sy in range(TRAIN_DRAFT_START,2026):
        sw=weekly[weekly.season==sy].copy()
        for pos in POSITIONS:
            ids=p.loc[p.position==pos,'gsis_id'].dropna().unique() if 'gsis_id' in p else []
            z=sw[sw[idcol].isin(ids)].copy()
            if z.empty: continue
            if pos=='QB': opp=z.get('attempts',z.get('passing_attempts',0))+z.get('carries',z.get('rushing_attempts',0)); z['m']=opp>=15
            elif pos=='RB': z['m']=(z.get('carries',0)+z.get('targets',0))>=5
            else: z['m']=(z.get('targets',0)+z.get('carries',0))>=4
            a=z[z.m].groupby(idcol).agg(ppg=('fp','mean'),games=('fp','size')).query('games>=6').sort_values('ppg',ascending=False)
            season_ranks[(sy,pos)]={pid:i+1 for i,pid in enumerate(a.index)}
    hits=[]
    for _,r in p.iterrows():
        ranks=[]
        if pd.notna(r.get('gsis_id')):
            for sy in range(int(r.season),int(r.season)+3):
                rank=season_ranks.get((sy,r.position),{}).get(r.gsis_id)
                if rank:ranks.append(rank)
        hits.append(int(bool(ranks) and min(ranks)<=HIT_RANK[r.position]))
    p['hit3']=hits
    return p

def feature_columns(df,position,feature_set):
    banned={'peak3_ppg','avg3_ppg','total3_ppr','hit3','season','pfr_id','pfr_name','position','gsis_id','name_norm','birth_date','college_name','college_match','first_season','last_season'}
    nums=[c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]
    capital=[c for c in nums if c in ['draft_round','draft_pick','log_pick','pick_inv_sqrt','day1','day2']]
    athletic=[c for c in nums if any(k in c for k in ['forty','vertical','bench','broad_jump','cone','shuttle','speed_score','height','weight','wt','draft_age'])]
    college=[c for c in nums if c not in set(capital+athletic) and not c.startswith('draft_')]
    if feature_set=='capital': return capital
    if feature_set=='capital_athletic': return list(dict.fromkeys(capital+athletic))
    if feature_set=='capital_college': return list(dict.fromkeys(capital+college))
    if feature_set=='college_only': return college
    return list(dict.fromkeys(capital+athletic+college))

def candidates(seed=20260816):
    return {
      'ridge':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('m',Ridge(alpha=30.0))]),
      'ridge10':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('m',Ridge(alpha=10.0))]),
      'extra':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('m',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.75,random_state=seed,n_jobs=-1))]),
      'extra2':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('m',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=7,max_features=1.0,random_state=seed+1,n_jobs=-1))]),
      'rf':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('m',RandomForestRegressor(n_estimators=500,min_samples_leaf=5,max_features=.7,random_state=seed,n_jobs=-1))]),
      'gbr':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('m',GradientBoostingRegressor(n_estimators=200,learning_rate=.025,max_depth=2,min_samples_leaf=5,loss='huber',random_state=seed))]),
      'hist':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('m',HistGradientBoostingRegressor(max_iter=250,learning_rate=.04,max_leaf_nodes=15,l2_regularization=3,min_samples_leaf=12,random_state=seed))]),
    }

def metrics(y,p):
    ok=np.isfinite(y)&np.isfinite(p);y=np.asarray(y)[ok];p=np.asarray(p)[ok]
    if len(y)<4:return {'n':len(y),'mae':np.nan,'rmse':np.nan,'r2':np.nan,'pearson':np.nan,'spearman':np.nan}
    return {'n':len(y),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'r2':r2_score(y,p),'pearson':pearsonr(y,p).statistic,'spearman':spearmanr(y,p).statistic}

def walkforward(df,features,model,years):
    preds=[]
    for y in years:
        tr=df[(df.season<y)&df.peak3_ppg.notna()];te=df[(df.season==y)&df.peak3_ppg.notna()]
        if len(tr)<40 or te.empty:continue
        m=clone(model);m.fit(tr[features],tr.peak3_ppg);q=te[['season','pfr_name','peak3_ppg','hit3']].copy();q['pred']=m.predict(te[features]);preds.append(q)
    return pd.concat(preds,ignore_index=True) if preds else pd.DataFrame()

def score_dev(m):
    # lower is better; prioritizes rank-order signal while retaining point accuracy
    return (m['mae'] if np.isfinite(m['mae']) else 99) - 2.0*(m['spearman'] if np.isfinite(m['spearman']) else -1)

def fit_position(df,pos):
    d=df[(df.position==pos)&(df.season<=TRAIN_DRAFT_END)&df.peak3_ppg.notna()].copy()
    feature_sets=['capital','capital_athletic','college_only','capital_college','full']
    dev_rows=[]; pred_store={}
    for fs in feature_sets:
        feats=feature_columns(d,pos,fs)
        if not feats:continue
        for mn,model in candidates().items():
            oof=walkforward(d,feats,model,DEV_TEST_YEARS)
            if oof.empty:continue
            met=metrics(oof.peak3_ppg,oof.pred);met.update({'position':pos,'feature_set':fs,'model':mn,'n_features':len(feats),'dev_score':score_dev(met)})
            dev_rows.append(met);pred_store[(fs,mn)]=oof
    dev=pd.DataFrame(dev_rows).sort_values(['dev_score','mae'])
    best=dev.iloc[0]; bfs,bmn=best.feature_set,best.model; feats=feature_columns(d,pos,bfs); base=candidates()[bmn]
    # Try a simple two-model blend against the next distinct candidate using only dev OOF; promote only if it wins.
    bestpred=pred_store[(bfs,bmn)].copy(); blend={'weight':1.0,'other':None}
    bestmetric=metrics(bestpred.peak3_ppg,bestpred.pred); bestscore=score_dev(bestmetric)
    for _,row in dev.iloc[1:8].iterrows():
        o=pred_store[(row.feature_set,row.model)]
        z=bestpred.merge(o[['season','pfr_name','pred']],on=['season','pfr_name'],suffixes=('_a','_b'))
        if len(z)<20:continue
        for w in [.25,.5,.75]:
            pr=w*z.pred_a+(1-w)*z.pred_b;mm=metrics(z.peak3_ppg,pr);sc=score_dev(mm)
            if sc<bestscore-0.01:
                bestscore=sc;bestmetric=mm;blend={'weight':w,'other':(row.feature_set,row.model)}
    # untouched holdout 2020-23: fit only on years before each test year
    hold_preds=[]
    for y in HOLDOUT_YEARS:
        tr=d[d.season<y];te=d[d.season==y]
        if te.empty:continue
        ma=clone(base);ma.fit(tr[feats],tr.peak3_ppg);pa=ma.predict(te[feats])
        if blend['other']:
            ofs,omn=blend['other'];of=feature_columns(d,pos,ofs);mb=clone(candidates()[omn]);mb.fit(tr[of],tr.peak3_ppg);pb=mb.predict(te[of]);pr=blend['weight']*pa+(1-blend['weight'])*pb
        else:pr=pa
        q=te[['season','pfr_name','peak3_ppg','hit3']].copy();q['pred']=pr;hold_preds.append(q)
    hold=pd.concat(hold_preds,ignore_index=True); holdmet=metrics(hold.peak3_ppg,hold.pred)
    # capital-only baseline on same holdout
    capfeats=feature_columns(d,pos,'capital');capmodel=candidates()['ridge10'];cap=[]
    for y in HOLDOUT_YEARS:
        tr=d[d.season<y];te=d[d.season==y]
        if te.empty:continue
        mm=clone(capmodel);mm.fit(tr[capfeats],tr.peak3_ppg);q=te[['season','pfr_name','peak3_ppg','hit3']].copy();q['pred']=mm.predict(te[capfeats]);cap.append(q)
    cap=pd.concat(cap,ignore_index=True); capmet=metrics(cap.peak3_ppg,cap.pred)
    # hit probability from selected features, walkforward holdout
    hp=[]
    for y in HOLDOUT_YEARS:
        tr=d[d.season<y];te=d[d.season==y]
        if te.empty or tr.hit3.nunique()<2:continue
        clf=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('m',LogisticRegression(C=.25,max_iter=3000,class_weight='balanced'))])
        clf.fit(tr[feats],tr.hit3);q=te[['season','pfr_name','hit3']].copy();q['prob']=clf.predict_proba(te[feats])[:,1];hp.append(q)
    hp=pd.concat(hp,ignore_index=True); auc=roc_auc_score(hp.hit3,hp.prob) if hp.hit3.nunique()>1 else np.nan;brier=brier_score_loss(hp.hit3,hp.prob)
    # final fit through 2023
    finala=clone(base);finala.fit(d[feats],d.peak3_ppg)
    finalb=None;of=None
    if blend['other']:
        ofs,omn=blend['other'];of=feature_columns(d,pos,ofs);finalb=clone(candidates()[omn]);finalb.fit(d[of],d.peak3_ppg)
    clf=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('m',LogisticRegression(C=.25,max_iter=3000,class_weight='balanced'))]);clf.fit(d[feats],d.hit3)
    jobjob={'position':pos,'features':feats,'model_a':finala,'blend':blend,'features_b':of,'model_b':finalb,'classifier':clf}
    joblib.dump(jobjob,MODELS/f'{pos}.joblib')
    summary={'position':pos,'n_train':len(d),'selected_feature_set':bfs,'selected_model':bmn,'n_features':len(feats),'blend':str(blend),'dev':bestmetric,'holdout':holdmet,'capital_holdout':capmet,'hit_auc':auc,'hit_brier':brier,'holdout_improvement_mae_pct':100*(capmet['mae']-holdmet['mae'])/capmet['mae'] if capmet['mae'] else np.nan,'holdout_spearman_gain':holdmet['spearman']-capmet['spearman']}
    return summary,dev,hold,hp,jobjob

def percentile_rank(hist,vals):
    h=np.asarray(pd.Series(hist).dropna());return [100*np.mean(h<=v) if np.isfinite(v) and len(h) else np.nan for v in vals]

def predict_2026(all_df,jobs):
    out=[]
    for pos,j in jobs.items():
        train=all_df[(all_df.position==pos)&(all_df.season<=TRAIN_DRAFT_END)&all_df.peak3_ppg.notna()]
        cur=all_df[(all_df.position==pos)&(all_df.season==PREDICT_DRAFT_YEAR)].copy()
        if cur.empty:continue
        pa=j['model_a'].predict(cur[j['features']]);pred=pa
        if j['model_b'] is not None:
            pb=j['model_b'].predict(cur[j['features_b']]);w=j['blend']['weight'];pred=w*pa+(1-w)*pb
        cur['pred_peak3_ppg']=pred;cur['hit_probability']=j['classifier'].predict_proba(cur[j['features']])[:,1]
        cur['model_percentile']=percentile_rank(train.peak3_ppg,cur.pred_peak3_ppg)
        # profile percentiles using interpretable pillars
        capital_score=-pd.to_numeric(cur.draft_pick,errors='coerce')
        hist_cap=-pd.to_numeric(train.draft_pick,errors='coerce')
        cur['draft_capital_percentile']=percentile_rank(hist_cap,capital_score)
        prod_candidates=[c for c in j['features'] if any(k in c for k in ['yard_share','target_share','rec_per_team_pass_att','scrim_ypg','yardsgame','playsgame']) and ('peak' in c or 'final' in c)]
        eff_candidates=[c for c in j['features'] if any(k in c for k in ['EPAplay','success','yardsplay','yards_per_target','yardsdropback']) and ('peak' in c or 'final' in c)]
        ath_candidates=[c for c in j['features'] if any(k in c for k in ['speed_score','forty','vertical','broad_jump'])]
        def pillar(cols,invert40=False):
            if not cols:return pd.Series(np.nan,index=cur.index),pd.Series(np.nan,index=train.index)
            def zframe(df):
                z=[]
                for c in cols:
                    a=pd.to_numeric(train[c],errors='coerce');mu=a.mean();sd=a.std() or 1
                    x=(pd.to_numeric(df[c],errors='coerce')-mu)/sd
                    if 'forty' in c:x=-x
                    z.append(x)
                return pd.concat(z,axis=1).mean(axis=1)
            return zframe(cur),zframe(train)
        for label,cols in [('production',prod_candidates),('efficiency',eff_candidates),('athleticism',ath_candidates)]:
            cv,hv=pillar(cols);cur[f'{label}_percentile']=percentile_rank(hv,cv)
        cur['rank']=cur.pred_peak3_ppg.rank(ascending=False,method='first').astype(int)
        keep=['position','rank','pfr_name','draft_round','draft_pick','pred_peak3_ppg','hit_probability','model_percentile','draft_capital_percentile','production_percentile','efficiency_percentile','athleticism_percentile','college_match']
        out.append(cur[keep].sort_values('rank'))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def correlations(df):
    rows=[]
    for pos in POSITIONS:
        d=df[(df.position==pos)&(df.season<=TRAIN_DRAFT_END)&df.peak3_ppg.notna()]
        for c in feature_columns(d,pos,'full'):
            x=pd.to_numeric(d[c],errors='coerce');ok=x.notna()&d.peak3_ppg.notna()
            if ok.sum()<20 or x[ok].nunique()<3:continue
            rows.append({'position':pos,'feature':c,'n':ok.sum(),'pearson_to_peak3_ppg':pearsonr(x[ok],d.loc[ok,'peak3_ppg']).statistic,'spearman_to_peak3_ppg':spearmanr(x[ok],d.loc[ok,'peak3_ppg']).statistic})
    return pd.DataFrame(rows).sort_values(['position','spearman_to_peak3_ppg'],ascending=[True,False])

def write_report(summaries,preds,corrs,devs):
    lines=['# Dynasty Rookie Prospect Models v1.0','',f'Generated {datetime.now(timezone.utc).isoformat()}','',
    'Four position-specific post-NFL-Draft models for PPR dynasty rookie evaluation. College features come from cfbfastR/SportsDataverse season summaries (2004-2025); NFL outcomes, draft capital, player IDs, and combine data come from nflverse.','',
    '## Outcome definition','', 'Primary regression target: best meaningful-game PPR points per game achieved in a prospect’s first three NFL seasons. A meaningful game is >=15 QB pass+rush attempts, >=5 RB carries+targets, or >=4 WR/TE targets+carries. Hit probability is the chance of at least one top-12 QB/RB/TE or top-24 WR meaningful-game PPG season within the first three years.','',
    '## Validation','', 'Model/feature-set selection uses leakage-safe draft-year walk-forward validation on 2013-2019 only. The 2020-2023 draft classes are then held out and touched once for final reporting. Every selected model is also compared with a draft-capital-only baseline on the identical holdout.','']
    for s in summaries:
        lines += [f"## {s['position']}", '', f"Training prospects: {s['n_train']}. Selected: {s['selected_feature_set']} / {s['selected_model']} ({s['n_features']} features), blend {s['blend']}.", '',
        f"Untouched 2020-23 holdout: MAE {s['holdout']['mae']:.3f}, RMSE {s['holdout']['rmse']:.3f}, Pearson {s['holdout']['pearson']:.3f}, Spearman {s['holdout']['spearman']:.3f}, R² {s['holdout']['r2']:.3f}. Hit AUC {s['hit_auc']:.3f}; Brier {s['hit_brier']:.3f}.",
        f"Draft-capital-only holdout: MAE {s['capital_holdout']['mae']:.3f}, Spearman {s['capital_holdout']['spearman']:.3f}. Final model MAE improvement {s['holdout_improvement_mae_pct']:.1f}%; Spearman gain {s['holdout_spearman_gain']:.3f}.", '']
    if not preds.empty:
        lines += ['## 2026 post-draft rankings','']
        for pos in POSITIONS:
            z=preds[preds.position==pos].head(12)
            if z.empty:continue
            lines.append(f'### {pos}')
            lines.append('')
            lines.append('|Rk|Prospect|Pick|Pred peak PPG|Hit %|Model pct|')
            lines.append('|---:|---|---:|---:|---:|---:|')
            for _,r in z.iterrows():lines.append(f"|{int(r['rank'])}|{r['pfr_name']}|{int(r['draft_pick']) if pd.notna(r['draft_pick']) else ''}|{r['pred_peak3_ppg']:.2f}|{100*r['hit_probability']:.1f}|{r['model_percentile']:.1f}|")
            lines.append('')
    lines += ['## Important interpretation','', 'The percentile is relative to the full historical drafted-prospect pool at that position, not only the current class. Draft capital is intentionally included because the model is designed for rookie drafts held after the NFL Draft. The untouched holdout is the key accuracy number; development results are not treated as final evidence.']
    (OUT/'REPORT.md').write_text('\n'.join(lines))

def main():
    print('Loading CFB summaries...')
    pa=load_cfb_table('passing');ru=load_cfb_table('rushing');re=load_cfb_table('receiving');pa,ru,re=prep_cfb(pa,ru,re)
    print('Loading NFLverse...')
    draft,players,combine,weekly=load_nflverse()
    draft['season']=pd.to_numeric(draft.season,errors='coerce');draft=draft[draft.category.isin(POSITIONS)&draft.season.between(TRAIN_DRAFT_START,PREDICT_DRAFT_YEAR)]
    prof=build_college_profiles(draft,pa,ru,re);prof=add_nfl_metadata(prof,players,combine);prof=nfl_targets(prof,weekly)
    prof.to_csv(OUT/'prospect_pool.csv',index=False)
    summaries=[];devs=[];jobs={}
    for pos in POSITIONS:
        print('Fitting',pos)
        s,dev,hold,hp,j=fit_position(prof,pos);summaries.append(s);devs.append(dev);jobs[pos]=j
        dev.to_csv(OUT/f'{pos}_development_grid.csv',index=False);hold.to_csv(OUT/f'{pos}_holdout_predictions.csv',index=False);hp.to_csv(OUT/f'{pos}_holdout_hit_probs.csv',index=False)
    preds=predict_2026(prof,jobs);preds.to_csv(OUT/'rookie_rankings_2026.csv',index=False)
    cor=correlations(prof);cor.to_csv(OUT/'feature_correlations.csv',index=False)
    pd.DataFrame([{**{k:v for k,v in s.items() if k not in ['dev','holdout','capital_holdout']},**{f'holdout_{k}':v for k,v in s['holdout'].items()},**{f'capital_{k}':v for k,v in s['capital_holdout'].items()}} for s in summaries]).to_csv(OUT/'model_summary.csv',index=False)
    write_report(summaries,preds,cor,devs)
    meta={'generated_utc':datetime.now(timezone.utc).isoformat(),'cfb_years':[START_CFB,LAST_CFB],'train_draft_years':[TRAIN_DRAFT_START,TRAIN_DRAFT_END],'development_years':DEV_TEST_YEARS,'untouched_holdout_years':HOLDOUT_YEARS,'prediction_draft_year':PREDICT_DRAFT_YEAR,'summaries':summaries}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2,default=float))
    print(json.dumps(meta,indent=2,default=float))

if __name__=='__main__': main()
