from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import stage2 as s2

ROOT=Path(__file__).resolve().parent
POOL=ROOT/'results_v4'/'prospect_pool_v4.csv'
OUT=ROOT/'results_stage6_newmodels'/'original_classifiers_fast'
OUT.mkdir(parents=True,exist_ok=True)
POSITIONS=['QB','RB','WR','TE'];LABELS=['hit3','star3'];DEV_END=2018;VAL_YEARS=[2019,2020,2021,2022];CONFIRM=2023;SEED=20260821


def auc(y,p):
    try:return float(roc_auc_score(y,p)) if pd.Series(y).nunique()>1 else np.nan
    except:return np.nan

def feature_sets(df,pos,label):
    groups=s2.feature_groups(df,pos);union=[];capital=[]
    for name,fs in groups.items():
        for c in fs:
            if c in df.columns and c not in union:union.append(c)
            if 'capital' in name.lower() and c in df.columns and c not in capital:capital.append(c)
    forbidden=('primary_ppg','peak3_ppg','rookie_ppg','avg3_ppg','total3_ppr','hit3','star3','best_rank3','target_valid','prospect_model','outcome_percentile','pred_')
    union=[c for c in union if not any(x in c.lower() for x in forbidden)]
    dev=df[(df.position.eq(pos))&df.season.le(DEV_END)&df.target_valid.eq(1)&df[label].notna()].copy()
    ok=[]
    for c in union:
        x=pd.to_numeric(dev[c],errors='coerce')
        if x.notna().sum()>=max(12,int(.18*len(dev))) and x.nunique(dropna=True)>1:ok.append(c)
    capital=[c for c in capital if c in ok];rank=[]
    for c in ok:
        x=pd.to_numeric(dev[c],errors='coerce');q=pd.DataFrame({'x':x,'y':dev[label]}).dropna()
        if len(q)>=12 and q.y.nunique()>1:
            a=auc(q.y,q.x);rank.append((c,abs(a-.5) if np.isfinite(a) else -1))
    ranked=[c for c,_ in sorted(rank,key=lambda z:(-z[1],z[0]))];non=[c for c in ranked if c not in capital]
    out={}
    if capital:out['capital']=capital
    for n in [16,32,64]:out[f'capital_top{n}']=list(dict.fromkeys(capital+non[:n]))
    return out

def model(k,pos,label):
    # Strong regularization matters for tiny positive classes, especially TE Star.
    if k=='logit':return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',LogisticRegression(C=.2,max_iter=2000,class_weight='balanced',random_state=SEED))])
    if k=='hist':return HistGradientBoostingClassifier(max_iter=220,learning_rate=.035,max_leaf_nodes=7,min_samples_leaf=12,l2_regularization=8,random_state=SEED)
    if k=='rf':return RandomForestClassifier(n_estimators=350,max_depth=5,min_samples_leaf=5,max_features=.7,class_weight='balanced_subsample',random_state=SEED,n_jobs=4)
    if k=='xgb':return XGBClassifier(n_estimators=250,max_depth=1,learning_rate=.035,subsample=.85,colsample_bytree=.8,min_child_weight=5,reg_alpha=.5,reg_lambda=12,objective='binary:logistic',eval_metric='logloss',random_state=SEED,n_jobs=4,verbosity=0)
    raise KeyError(k)

def fp(kind,tr,va,fs,label,pos):
    use=[c for c in fs if pd.to_numeric(tr[c],errors='coerce').notna().any()]
    X=tr[use].apply(pd.to_numeric,errors='coerce');V=va[use].apply(pd.to_numeric,errors='coerce');m=model(kind,pos,label)
    if kind in ('hist','rf','xgb'):
        imp=SimpleImputer(strategy='median',add_indicator=True);X=imp.fit_transform(X);V=imp.transform(V)
    m.fit(X,tr[label].astype(int));return m.predict_proba(V)[:,1]

def walk(df,pos,label,fs,kind,years):
    out=[];d=df[df.position.eq(pos)].copy()
    for yr in years:
        tr=d[(d.season.lt(yr))&d.target_valid.eq(1)&d[label].notna()].copy();va=d[d.season.eq(yr)&d.target_valid.eq(1)&d[label].notna()].copy()
        if tr[label].nunique()<2 or not len(va):continue
        p=fp(kind,tr,va,fs,label,pos);out.append(pd.DataFrame({'season':yr,'position':pos,'pfr_name':va.pfr_name.values,'label':label,'y':va[label].astype(int).values,'pred':p,'kind':kind}))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def metrics(o):
    ya=[]
    for _,g in o.groupby('season'):ya.append(auc(g.y,g.pred))
    return {'n':len(o),'positives':int(o.y.sum()),'auc':auc(o.y,o.pred),'brier':brier_score_loss(o.y,o.pred),'mean_year_auc':float(np.nanmean(ya)) if np.isfinite(np.array(ya,dtype=float)).any() else np.nan,'min_year_auc':float(np.nanmin(ya)) if np.isfinite(np.array(ya,dtype=float)).any() else np.nan}

def main():
    d=pd.read_csv(POOL,low_memory=False);d['season']=pd.to_numeric(d.season,errors='coerce');d['target_valid']=pd.to_numeric(d.target_valid,errors='coerce').fillna(0)
    for lab in LABELS:d[lab]=pd.to_numeric(d[lab],errors='coerce')
    kinds=['logit','hist','rf','xgb'];rows=[];oofs=[];sel=[];conf=[];cp=[];future=[];aud=[]
    for pos in POSITIONS:
      for lab in LABELS:
        sets=feature_sets(d,pos,lab)
        for sn,fs in sets.items():aud.append({'position':pos,'label':lab,'feature_set':sn,'n_features':len(fs),'features':'|'.join(fs)})
        for sn,fs in sets.items():
          for kind in kinds:
            o=walk(d,pos,lab,fs,kind,VAL_YEARS)
            if o.empty:continue
            o['feature_set']=sn;oofs.append(o);r=metrics(o);r.update(position=pos,label=lab,feature_set=sn,kind=kind,n_features=len(fs));rows.append(r)
        mm=pd.DataFrame([x for x in rows if x['position']==pos and x['label']==lab]).copy()
        if mm.empty:continue
        mm['r_auc']=mm.auc.rank(method='min',ascending=False,na_option='bottom');mm['r_brier']=mm.brier.rank(method='min',ascending=True);mm['r_mean']=mm.mean_year_auc.rank(method='min',ascending=False,na_option='bottom');mm['r_min']=mm.min_year_auc.rank(method='min',ascending=False,na_option='bottom');mm['selection_score']=1.3*mm.r_auc+1.1*mm.r_brier+.5*mm.r_mean+.25*mm.r_min
        w=mm.sort_values(['selection_score','auc','brier'],ascending=[True,False,True]).iloc[0];sel.append(w.to_dict());fs=sets[w.feature_set]
        co=walk(d,pos,lab,fs,w.kind,[CONFIRM]);
        if not co.empty:
            co['feature_set']=w.feature_set;cp.append(co);z=metrics(co);z.update(position=pos,label=lab,feature_set=w.feature_set,kind=w.kind,n_features=len(fs));conf.append(z)
        tr=d[(d.position.eq(pos))&d.season.le(CONFIRM)&d.target_valid.eq(1)&d[lab].notna()].copy();va=d[(d.position.eq(pos))&d.season.between(2024,2026)].copy()
        if len(va) and tr[lab].nunique()>1:
            p=fp(w.kind,tr,va,fs,lab,pos);future.append(pd.DataFrame({'season':va.season.values,'position':pos,'pfr_name':va.pfr_name.values,'label':lab,'original_fast_prob':p,'feature_set':w.feature_set,'kind':w.kind}))
    pd.DataFrame(rows).to_csv(OUT/'validation_search.csv',index=False);pd.DataFrame(sel).to_csv(OUT/'selected_models.csv',index=False);pd.DataFrame(conf).to_csv(OUT/'confirmation_2023.csv',index=False);pd.DataFrame(aud).to_csv(OUT/'feature_set_audit.csv',index=False)
    if oofs:
        oo=pd.concat(oofs,ignore_index=True);oo.to_csv(OUT/'validation_oof_all.csv',index=False);ss=pd.DataFrame(sel);picked=[]
        for _,r in ss.iterrows():picked.append(oo[(oo.position.eq(r.position))&(oo.label.eq(r.label))&(oo.kind.eq(r.kind))&(oo.feature_set.eq(r.feature_set))].copy())
        pd.concat(picked,ignore_index=True).to_csv(OUT/'selected_validation_oof.csv',index=False)
    if cp:pd.concat(cp,ignore_index=True).to_csv(OUT/'confirmation_2023_predictions.csv',index=False)
    if future:pd.concat(future,ignore_index=True).to_csv(OUT/'predictions_2024_2026.csv',index=False)
    print(pd.DataFrame(sel)[['position','label','kind','feature_set','auc','brier','mean_year_auc','min_year_auc']].to_string(index=False));print(pd.DataFrame(conf).to_string(index=False))

if __name__=='__main__':main()
