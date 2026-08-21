from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import stage2 as s2

ROOT=Path(__file__).resolve().parent
POOL=ROOT/'results_v4'/'prospect_pool_v4.csv'
OUT=ROOT/'results_stage6_newmodels'/'original_fast'
OUT.mkdir(parents=True,exist_ok=True)
POSITIONS=['QB','RB','WR','TE']
DEV_END=2018
VAL_YEARS=[2019,2020,2021,2022]
CONFIRM=2023
SEED=20260821


def sp(a,b):
    try:
        z=spearmanr(a,b).statistic
        return float(z) if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def candidate_features(df,pos):
    groups=s2.feature_groups(df,pos)
    union=[]; capital=[]
    for name,fs in groups.items():
        for c in fs:
            if c in df.columns and c not in union: union.append(c)
            if 'capital' in name.lower() and c in df.columns and c not in capital: capital.append(c)
    forbidden=('primary_ppg','peak3_ppg','rookie_ppg','avg3_ppg','total3_ppr','hit3','star3','best_rank3','target_valid','prospect_model','outcome_percentile','pred_')
    union=[c for c in union if not any(x in c.lower() for x in forbidden)]
    dev=df[(df.position.eq(pos)) & df.season.le(DEV_END) & df.target_valid.eq(1) & df.primary_ppg.notna()].copy()
    ok=[]
    for c in union:
        x=pd.to_numeric(dev[c],errors='coerce')
        if x.notna().sum()>=max(12,int(.18*len(dev))) and x.nunique(dropna=True)>1: ok.append(c)
    capital=[c for c in capital if c in ok]
    rank=[]
    for c in ok:
        q=dev[[c,'primary_ppg']].copy();q[c]=pd.to_numeric(q[c],errors='coerce');q=q.dropna()
        if len(q)>=12:
            z=abs(sp(q[c],q.primary_ppg));rank.append((c,z if np.isfinite(z) else -1))
    rank=[c for c,_ in sorted(rank,key=lambda x:(-x[1],x[0]))]
    return capital,rank


def feature_sets(df,pos):
    capital,ranked=candidate_features(df,pos)
    non=[c for c in ranked if c not in capital]
    out={}
    if capital: out['capital']=capital
    for n in [12,24,48,96]:
        fs=list(dict.fromkeys(capital+non[:n]))
        if fs: out[f'capital_top{n}']=fs
    return out


def model(k):
    if k=='ridge10': return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=10.0))])
    if k=='ridge50': return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=50.0))])
    if k=='hist': return HistGradientBoostingRegressor(loss='absolute_error',max_iter=250,learning_rate=.04,max_leaf_nodes=9,min_samples_leaf=12,l2_regularization=5,random_state=SEED)
    if k=='xgb1': return XGBRegressor(n_estimators=300,max_depth=1,learning_rate=.035,subsample=.85,colsample_bytree=.8,min_child_weight=5,reg_alpha=.5,reg_lambda=12,objective='reg:squarederror',random_state=SEED,n_jobs=4,verbosity=0)
    if k=='extra': return ExtraTreesRegressor(n_estimators=350,max_depth=5,min_samples_leaf=5,max_features=.7,random_state=SEED,n_jobs=4)
    raise KeyError(k)


def fit_predict(kind,tr,va,fs):
    use=[c for c in fs if pd.to_numeric(tr[c],errors='coerce').notna().any()]
    X=tr[use].apply(pd.to_numeric,errors='coerce');V=va[use].apply(pd.to_numeric,errors='coerce')
    m=model(kind)
    if kind in ('hist','extra'):
        imp=SimpleImputer(strategy='median',add_indicator=True);X=imp.fit_transform(X);V=imp.transform(V)
    m.fit(X,tr.primary_ppg)
    return m.predict(V)


def walk(df,pos,fs,kind,years):
    out=[];d=df[df.position.eq(pos)].copy()
    for yr in years:
        tr=d[(d.season.lt(yr)) & d.target_valid.eq(1) & d.primary_ppg.notna()].copy()
        va=d[d.season.eq(yr) & d.target_valid.eq(1) & d.primary_ppg.notna()].copy()
        p=fit_predict(kind,tr,va,fs)
        out.append(pd.DataFrame({'season':yr,'position':pos,'pfr_name':va.pfr_name.values,'y':va.primary_ppg.values,'pred':p,'kind':kind}))
    return pd.concat(out,ignore_index=True)


def metrics(o):
    ys=[sp(g.y,g.pred) for _,g in o.groupby('season')]
    return {'n':len(o),'mae':mean_absolute_error(o.y,o.pred),'rmse':mean_squared_error(o.y,o.pred)**.5,'spearman':sp(o.y,o.pred),'mean_year_spearman':float(np.nanmean(ys)),'min_year_spearman':float(np.nanmin(ys))}


def main():
    d=pd.read_csv(POOL,low_memory=False)
    d['season']=pd.to_numeric(d.season,errors='coerce');d['target_valid']=pd.to_numeric(d.target_valid,errors='coerce').fillna(0);d['primary_ppg']=pd.to_numeric(d.primary_ppg,errors='coerce')
    kinds=['ridge10','ridge50','hist','xgb1','extra']
    rows=[];oofs=[];selected=[];conf=[];cp=[];future=[];aud=[]
    for pos in POSITIONS:
        sets=feature_sets(d,pos)
        for sn,fs in sets.items(): aud.append({'position':pos,'feature_set':sn,'n_features':len(fs),'features':'|'.join(fs)})
        for sn,fs in sets.items():
            for kind in kinds:
                o=walk(d,pos,fs,kind,VAL_YEARS);o['feature_set']=sn;oofs.append(o)
                r=metrics(o);r.update(position=pos,feature_set=sn,kind=kind,n_features=len(fs));rows.append(r)
        mm=pd.DataFrame([x for x in rows if x['position']==pos]).copy()
        for c,asc in [('mae',True),('rmse',True),('spearman',False),('mean_year_spearman',False),('min_year_spearman',False)]: mm['r_'+c]=mm[c].rank(method='min',ascending=asc,na_option='bottom')
        mm['selection_score']=1.35*mm.r_mae+mm.r_rmse+mm.r_spearman+.55*mm.r_mean_year_spearman+.35*mm.r_min_year_spearman
        w=mm.sort_values(['selection_score','mae','spearman'],ascending=[True,True,False]).iloc[0];selected.append(w.to_dict())
        fs=sets[w.feature_set];co=walk(d,pos,fs,w.kind,[CONFIRM]);co['feature_set']=w.feature_set;cp.append(co)
        z=metrics(co);z.update(position=pos,feature_set=w.feature_set,kind=w.kind,n_features=len(fs));conf.append(z)
        tr=d[(d.position.eq(pos)) & d.season.le(CONFIRM) & d.target_valid.eq(1) & d.primary_ppg.notna()].copy();va=d[(d.position.eq(pos)) & d.season.between(2024,2026)].copy()
        if len(va):
            p=fit_predict(w.kind,tr,va,fs);future.append(pd.DataFrame({'season':va.season.values,'position':pos,'pfr_name':va.pfr_name.values,'draft_pick':pd.to_numeric(va.get('draft_pick'),errors='coerce').values,'original_fast_pred_ppg':p,'feature_set':w.feature_set,'kind':w.kind}))
    pd.DataFrame(rows).to_csv(OUT/'validation_model_search.csv',index=False);pd.concat(oofs,ignore_index=True).to_csv(OUT/'validation_oof_all.csv',index=False);pd.DataFrame(selected).to_csv(OUT/'selected_models.csv',index=False);pd.DataFrame(conf).to_csv(OUT/'confirmation_2023.csv',index=False);pd.concat(cp,ignore_index=True).to_csv(OUT/'confirmation_2023_predictions.csv',index=False);pd.DataFrame(aud).to_csv(OUT/'feature_set_audit.csv',index=False)
    if future: pd.concat(future,ignore_index=True).to_csv(OUT/'predictions_2024_2026.csv',index=False)
    oo=pd.concat(oofs,ignore_index=True);ss=pd.DataFrame(selected);picked=[]
    for _,r in ss.iterrows(): picked.append(oo[(oo.position.eq(r.position)) & oo.kind.eq(r.kind) & oo.feature_set.eq(r.feature_set)].copy())
    pd.concat(picked,ignore_index=True).to_csv(OUT/'selected_validation_oof.csv',index=False)
    print(ss[['position','kind','feature_set','mae','rmse','spearman','mean_year_spearman','min_year_spearman']].to_string(index=False));print(pd.DataFrame(conf).to_string(index=False))

if __name__=='__main__': main()
