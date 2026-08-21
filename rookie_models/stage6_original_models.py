from __future__ import annotations

import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import stage2 as s2

ROOT=Path(__file__).resolve().parent
POOL=ROOT/'results_v4'/'prospect_pool_v4.csv'
OUT=ROOT/'results_stage6_newmodels'/'original_branch'
OUT.mkdir(parents=True,exist_ok=True)
POSITIONS=['QB','RB','WR','TE']
DEV_END=2018
VAL_YEARS=[2019,2020,2021,2022]
CONFIRM=2023
SEED=20260821


def safe_sp(a,b):
    try:
        z=spearmanr(a,b).statistic
        return float(z) if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def candidate_features(df,pos):
    groups=s2.feature_groups(df,pos)
    union=[]
    capital=[]
    for name,fs in groups.items():
        for c in fs:
            if c in df.columns and c not in union:
                union.append(c)
            if 'capital' in name.lower() and c in df.columns and c not in capital:
                capital.append(c)
    # Explicitly block labels/model outputs even if a future group definition changes.
    forbidden=('primary_ppg','peak3_ppg','rookie_ppg','avg3_ppg','total3_ppr','hit3','star3','best_rank3','target_valid','prospect_model','outcome_percentile','pred_')
    union=[c for c in union if not any(x in c.lower() for x in forbidden)]
    capital=[c for c in capital if c in union]
    dev=df[(df.position.eq(pos)) & (df.season.le(DEV_END)) & (df.target_valid.eq(1)) & df.primary_ppg.notna()].copy()
    numeric=[]
    for c in union:
        x=pd.to_numeric(dev[c],errors='coerce')
        if x.notna().sum()>=max(12,int(.18*len(dev))):
            numeric.append(c)
    capital=[c for c in capital if c in numeric]
    rank=[]
    for c in numeric:
        q=dev[[c,'primary_ppg']].copy();q[c]=pd.to_numeric(q[c],errors='coerce');q=q.dropna()
        if len(q)<12 or q[c].nunique()<2: continue
        rank.append((c,abs(safe_sp(q[c],q.primary_ppg))))
    rank=sorted(rank,key=lambda z:(-(-1 if pd.isna(z[1]) else z[1]),z[0]))
    ranked=[c for c,_ in rank]
    return groups,capital,ranked


def feature_sets(df,pos):
    groups,capital,ranked=candidate_features(df,pos)
    noncap=[c for c in ranked if c not in capital]
    sets={}
    for n in [8,16,32,64,96]:
        fs=list(dict.fromkeys(capital+noncap[:n]))
        if fs: sets[f'capital_plus_top{n}']=fs
    sets['all_dev_observed']=list(dict.fromkeys(capital+noncap))
    return sets, groups


def model(kind):
    if kind=='ridge30':
        return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=30.0))])
    if kind=='xgb1':
        return XGBRegressor(n_estimators=350,max_depth=1,learning_rate=.035,subsample=.85,colsample_bytree=.75,min_child_weight=5,reg_alpha=.5,reg_lambda=10,objective='reg:squarederror',random_state=SEED,n_jobs=4,verbosity=0)
    if kind=='xgb2':
        return XGBRegressor(n_estimators=450,max_depth=2,learning_rate=.025,subsample=.85,colsample_bytree=.75,min_child_weight=7,reg_alpha=.5,reg_lambda=16,objective='reg:squarederror',random_state=SEED,n_jobs=4,verbosity=0)
    if kind=='cat3':
        return CatBoostRegressor(iterations=450,depth=3,learning_rate=.03,l2_leaf_reg=8,loss_function='MAE',random_seed=SEED,verbose=False,allow_writing_files=False)
    if kind=='cat4':
        return CatBoostRegressor(iterations=450,depth=4,learning_rate=.025,l2_leaf_reg=16,loss_function='MAE',random_seed=SEED,verbose=False,allow_writing_files=False)
    raise KeyError(kind)


def score_rows(o):
    ys=[]
    for _,g in o.groupby('season'):
        ys.append(safe_sp(g.y,g.pred))
    return {
        'n':len(o),
        'mae':mean_absolute_error(o.y,o.pred),
        'rmse':mean_squared_error(o.y,o.pred)**.5,
        'spearman':safe_sp(o.y,o.pred),
        'pearson':float(pearsonr(o.y,o.pred).statistic) if len(o)>2 else np.nan,
        'mean_year_spearman':float(np.nanmean(ys)),
        'min_year_spearman':float(np.nanmin(ys)),
    }


def walk(df,pos,fs,kind,years):
    out=[]
    d=df[df.position.eq(pos)].copy()
    for yr in years:
        tr=d[(d.season.lt(yr)) & d.target_valid.eq(1) & d.primary_ppg.notna()].copy()
        va=d[d.season.eq(yr) & d.target_valid.eq(1) & d.primary_ppg.notna()].copy()
        use=[c for c in fs if pd.to_numeric(tr[c],errors='coerce').notna().any()]
        X=tr[use].apply(pd.to_numeric,errors='coerce');V=va[use].apply(pd.to_numeric,errors='coerce')
        m=model(kind);m.fit(X,tr.primary_ppg);p=m.predict(V)
        out.append(pd.DataFrame({'season':yr,'position':pos,'pfr_name':va.pfr_name.values,'y':va.primary_ppg.values,'pred':p,'kind':kind,'feature_set':''}))
    return pd.concat(out,ignore_index=True)


def main():
    d=pd.read_csv(POOL,low_memory=False)
    d['season']=pd.to_numeric(d.season,errors='coerce')
    d['target_valid']=pd.to_numeric(d.target_valid,errors='coerce').fillna(0)
    d['primary_ppg']=pd.to_numeric(d.primary_ppg,errors='coerce')
    metrics=[]; all_oof=[]; selected=[]; confirms=[]; future=[]; audits=[]
    kinds=['ridge30','xgb1','xgb2','cat3','cat4']
    for pos in POSITIONS:
        sets,groups=feature_sets(d,pos)
        for name,fs in sets.items():
            audits.append({'position':pos,'feature_set':name,'n_features':len(fs),'features':'|'.join(fs)})
        # Keep search compact: full observed set is allowed only for heavily regularized tree models.
        for sname,fs in sets.items():
            usek=kinds if sname!='all_dev_observed' else ['xgb1','cat3']
            for kind in usek:
                o=walk(d,pos,fs,kind,VAL_YEARS);o['feature_set']=sname
                r=score_rows(o);r.update(position=pos,feature_set=sname,kind=kind,n_features=len(fs));metrics.append(r);all_oof.append(o)
        mm=pd.DataFrame([r for r in metrics if r['position']==pos]).copy()
        for c,asc in [('mae',True),('rmse',True),('spearman',False),('mean_year_spearman',False),('min_year_spearman',False)]:
            mm['r_'+c]=mm[c].rank(method='min',ascending=asc)
        mm['selection_score']=mm['r_mae']+mm['r_rmse']+mm['r_spearman']+.6*mm['r_mean_year_spearman']+.6*mm['r_min_year_spearman']
        win=mm.sort_values(['selection_score','mae','spearman'],ascending=[True,True,False]).iloc[0]
        selected.append(win.to_dict())
        fs=sets[win.feature_set];kind=win.kind
        co=walk(d,pos,fs,kind,[CONFIRM]);cr=score_rows(co);cr.update(position=pos,feature_set=win.feature_set,kind=kind,n_features=len(fs));confirms.append(cr)
        # Final prediction-only scoring after architecture is frozen.
        tr=d[(d.position.eq(pos)) & d.season.le(CONFIRM) & d.target_valid.eq(1) & d.primary_ppg.notna()].copy()
        va=d[(d.position.eq(pos)) & d.season.between(2024,2026)].copy()
        use=[c for c in fs if pd.to_numeric(tr[c],errors='coerce').notna().any()]
        if len(va):
            m=model(kind);m.fit(tr[use].apply(pd.to_numeric,errors='coerce'),tr.primary_ppg)
            p=m.predict(va[use].apply(pd.to_numeric,errors='coerce'))
            future.append(pd.DataFrame({'season':va.season.values,'position':pos,'pfr_name':va.pfr_name.values,'draft_pick':pd.to_numeric(va.get('draft_pick'),errors='coerce').values,'original_branch_pred_ppg':p,'feature_set':win.feature_set,'kind':kind}))
    pd.DataFrame(metrics).to_csv(OUT/'validation_model_search.csv',index=False)
    pd.concat(all_oof,ignore_index=True).to_csv(OUT/'validation_oof_all.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'selected_models.csv',index=False)
    pd.DataFrame(confirms).to_csv(OUT/'confirmation_2023.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'feature_set_audit.csv',index=False)
    if future: pd.concat(future,ignore_index=True).to_csv(OUT/'predictions_2024_2026.csv',index=False)
    # Small selected-OOF files are easy to transfer without exposing any licensed advanced source data.
    oo=pd.concat(all_oof,ignore_index=True)
    ss=pd.DataFrame(selected)
    picked=[]
    for _,r in ss.iterrows():
        picked.append(oo[(oo.position.eq(r.position)) & oo.kind.eq(r.kind) & oo.feature_set.eq(r.feature_set)].copy())
    pd.concat(picked,ignore_index=True).to_csv(OUT/'selected_validation_oof.csv',index=False)
    print('SELECTED')
    print(ss[['position','kind','feature_set','n_features','mae','rmse','spearman','mean_year_spearman','min_year_spearman','selection_score']].to_string(index=False))
    print('2023 CONFIRM')
    print(pd.DataFrame(confirms).to_string(index=False))

if __name__=='__main__':
    main()
