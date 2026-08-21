from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import stage2 as s2

ROOT = Path(__file__).resolve().parent
POOL = ROOT / 'results_v4' / 'prospect_pool_v4.csv'
OUT = ROOT / 'results_stage6_newmodels' / 'original_classifiers'
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ['QB','RB','WR','TE']
TARGETS = ['hit3','star3']
DEV_END = 2018
VAL_YEARS = [2019,2020,2021,2022]
CONFIRM = 2023
SEED = 20260821


def safe_auc(y,p):
    y=np.asarray(y)
    if len(np.unique(y))<2: return np.nan
    return float(roc_auc_score(y,p))


def base_features(df,pos):
    groups=s2.feature_groups(df,pos)
    union=[]; capital=[]
    for name,fs in groups.items():
        for c in fs:
            if c in df.columns and c not in union: union.append(c)
            if 'capital' in name.lower() and c in df.columns and c not in capital: capital.append(c)
    forbidden=('primary_ppg','peak3_ppg','rookie_ppg','avg3_ppg','total3_ppr','hit3','star3','best_rank3','target_valid','prospect_model','outcome_percentile','pred_')
    union=[c for c in union if not any(x in c.lower() for x in forbidden)]
    capital=[c for c in capital if c in union]
    return union,capital


def feature_sets(df,pos,target):
    union,capital=base_features(df,pos)
    dev=df[(df.position.eq(pos)) & (df.season.le(DEV_END)) & (df.target_valid.eq(1)) & df[target].notna()].copy()
    numeric=[]
    mincov=max(12,int(.18*len(dev)))
    for c in union:
        x=pd.to_numeric(dev[c],errors='coerce')
        if x.notna().sum()>=mincov and x.nunique(dropna=True)>=2:
            numeric.append(c)
    capital=[c for c in capital if c in numeric]
    ranks=[]
    for c in numeric:
        q=dev[[c,target,'season']].copy();q[c]=pd.to_numeric(q[c],errors='coerce');q=q.dropna()
        if len(q)<12 or q[target].nunique()<2: continue
        try: corr=float(spearmanr(q[c],q[target]).statistic)
        except Exception: corr=np.nan
        if not np.isfinite(corr): continue
        signs=[]
        for _,g in q.groupby('season'):
            if len(g)>=5 and g[target].nunique()>1 and g[c].nunique()>1:
                z=spearmanr(g[c],g[target]).statistic
                if np.isfinite(z) and z!=0: signs.append(np.sign(z))
        consistency=abs(np.mean(signs)) if signs else 0.0
        score=abs(corr)*(0.7+0.3*consistency)
        ranks.append((c,score,abs(corr),consistency,len(q)))
    ranks=sorted(ranks,key=lambda z:(-z[1],-z[2],z[0]))
    ranked=[c for c,*_ in ranks]
    noncap=[c for c in ranked if c not in capital]
    sets={}
    for n in [8,16,32,64]:
        fs=list(dict.fromkeys(capital+noncap[:n]))
        if fs: sets[f'capital_plus_top{n}']=fs
    sets['all_dev_observed']=list(dict.fromkeys(capital+noncap))
    audit=pd.DataFrame(ranks,columns=['feature','score','abs_corr','sign_consistency','n'])
    return sets,audit


def model(kind,positive_rate):
    if kind=='logit':
        return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',LogisticRegression(C=.25,max_iter=3000,class_weight='balanced',random_state=SEED))])
    if kind=='hist':
        return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',HistGradientBoostingClassifier(max_iter=220,learning_rate=.035,max_leaf_nodes=7,l2_regularization=5,min_samples_leaf=12,random_state=SEED))])
    if kind=='rf':
        return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',RandomForestClassifier(n_estimators=700,max_depth=4,min_samples_leaf=5,max_features=.65,class_weight='balanced_subsample',random_state=SEED,n_jobs=4))])
    if kind=='xgb':
        scale=(1-positive_rate)/max(positive_rate,1e-4)
        return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',XGBClassifier(n_estimators=350,max_depth=2,learning_rate=.025,subsample=.85,colsample_bytree=.75,min_child_weight=5,reg_alpha=.8,reg_lambda=12,objective='binary:logistic',eval_metric='logloss',scale_pos_weight=min(scale,8.0),random_state=SEED,n_jobs=4,verbosity=0))])
    raise KeyError(kind)


def metrics(o):
    y=o.y.astype(int).to_numpy();p=np.clip(o.pred.to_numpy(),1e-6,1-1e-6)
    yearly=[]
    for _,g in o.groupby('season'):
        yearly.append(safe_auc(g.y,g.pred))
    return {'n':len(o),'positives':int(y.sum()),'auc':safe_auc(y,p),'brier':float(brier_score_loss(y,p)),'logloss':float(log_loss(y,p,labels=[0,1])),'ap':float(average_precision_score(y,p)),'mean_year_auc':float(np.nanmean(yearly)) if np.isfinite(yearly).any() else np.nan,'min_year_auc':float(np.nanmin(yearly)) if np.isfinite(yearly).any() else np.nan}


def walk(df,pos,target,fs,kind,years):
    out=[];d=df[df.position.eq(pos)].copy()
    for yr in years:
        tr=d[(d.season.lt(yr)) & d.target_valid.eq(1) & d[target].notna()].copy()
        va=d[(d.season.eq(yr)) & d.target_valid.eq(1) & d[target].notna()].copy()
        use=[c for c in fs if pd.to_numeric(tr[c],errors='coerce').notna().any()]
        X=tr[use].apply(pd.to_numeric,errors='coerce');V=va[use].apply(pd.to_numeric,errors='coerce')
        pr=float(tr[target].mean())
        m=model(kind,pr);m.fit(X,tr[target].astype(int));p=m.predict_proba(V)[:,1]
        out.append(pd.DataFrame({'season':yr,'position':pos,'target':target,'pfr_name':va.pfr_name.values,'y':va[target].astype(int).values,'pred':p,'kind':kind,'feature_set':''}))
    return pd.concat(out,ignore_index=True)


def main():
    d=pd.read_csv(POOL,low_memory=False)
    d['season']=pd.to_numeric(d.season,errors='coerce');d['target_valid']=pd.to_numeric(d.target_valid,errors='coerce').fillna(0)
    for t in TARGETS:d[t]=pd.to_numeric(d[t],errors='coerce')
    rows=[];oofs=[];selected=[];conf=[];future=[];audits=[]
    kinds=['logit','hist','rf','xgb']
    for pos in POSITIONS:
        for target in TARGETS:
            sets,audit=feature_sets(d,pos,target);audit['position']=pos;audit['target']=target;audits.append(audit)
            local=[]
            for sname,fs in sets.items():
                usek=kinds if sname!='all_dev_observed' else ['logit','xgb']
                for kind in usek:
                    o=walk(d,pos,target,fs,kind,VAL_YEARS);o['feature_set']=sname
                    r=metrics(o);r.update(position=pos,target=target,feature_set=sname,kind=kind,n_features=len(fs));rows.append(r);local.append(r);oofs.append(o)
            mm=pd.DataFrame(local)
            for c,asc in [('auc',False),('brier',True),('logloss',True),('ap',False),('mean_year_auc',False),('min_year_auc',False)]:
                mm['r_'+c]=mm[c].rank(method='min',ascending=asc,na_option='bottom')
            mm['selection_score']=mm.r_auc+mm.r_brier+.75*mm.r_logloss+.35*mm.r_ap+.65*mm.r_mean_year_auc+.65*mm.r_min_year_auc
            win=mm.sort_values(['selection_score','auc','brier'],ascending=[True,False,True]).iloc[0];selected.append(win.to_dict())
            fs=sets[win.feature_set];kind=win.kind
            co=walk(d,pos,target,fs,kind,[CONFIRM]);cr=metrics(co);cr.update(position=pos,target=target,feature_set=win.feature_set,kind=kind,n_features=len(fs));conf.append(cr)
            tr=d[(d.position.eq(pos)) & d.season.le(CONFIRM) & d.target_valid.eq(1) & d[target].notna()].copy();va=d[(d.position.eq(pos)) & d.season.between(2024,2026)].copy()
            use=[c for c in fs if pd.to_numeric(tr[c],errors='coerce').notna().any()]
            if len(va):
                m=model(kind,float(tr[target].mean()));m.fit(tr[use].apply(pd.to_numeric,errors='coerce'),tr[target].astype(int));p=m.predict_proba(va[use].apply(pd.to_numeric,errors='coerce'))[:,1]
                future.append(pd.DataFrame({'season':va.season.values,'position':pos,'target':target,'pfr_name':va.pfr_name.values,'original_branch_probability':p,'feature_set':win.feature_set,'kind':kind}))
    pd.DataFrame(rows).to_csv(OUT/'validation_model_search.csv',index=False)
    oo=pd.concat(oofs,ignore_index=True);oo.to_csv(OUT/'validation_oof_all.csv',index=False)
    ss=pd.DataFrame(selected);ss.to_csv(OUT/'selected_models.csv',index=False)
    pd.DataFrame(conf).to_csv(OUT/'confirmation_2023.csv',index=False)
    pd.concat(audits,ignore_index=True).to_csv(OUT/'dev_feature_ranking.csv',index=False)
    if future:pd.concat(future,ignore_index=True).to_csv(OUT/'predictions_2024_2026.csv',index=False)
    picked=[]
    for _,r in ss.iterrows():
        picked.append(oo[(oo.position.eq(r.position)) & oo.target.eq(r.target) & oo.kind.eq(r.kind) & oo.feature_set.eq(r.feature_set)].copy())
    pd.concat(picked,ignore_index=True).to_csv(OUT/'selected_validation_oof.csv',index=False)
    print('SELECTED')
    print(ss[['position','target','kind','feature_set','auc','brier','mean_year_auc','min_year_auc','selection_score']].to_string(index=False))
    print('2023 CONFIRM')
    print(pd.DataFrame(conf).to_string(index=False))

if __name__=='__main__':main()
