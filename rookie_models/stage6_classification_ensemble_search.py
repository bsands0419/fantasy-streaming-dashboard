from __future__ import annotations

import base64
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score

ROOT=Path(__file__).resolve().parent
R=ROOT/'results_stage6_newmodels'
ADV=R/'advanced_selected_validation_classification_oof.b64'
ORG=R/'original_classifiers'/'selected_validation_oof.csv'
OUT=R/'classification_ensemble'
OUT.mkdir(parents=True,exist_ok=True)
WEIGHTS=np.round(np.arange(0.0,1.0001,.05),2)


def safe_auc(y,p):
    y=np.asarray(y)
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan


def metrics(d):
    y=d.y.astype(int).to_numpy();p=np.clip(d.pred.to_numpy(),1e-6,1-1e-6)
    ya=[safe_auc(g.y,g.pred) for _,g in d.groupby('season')]
    return {
        'n':len(d),'positives':int(y.sum()),'auc':safe_auc(y,p),
        'brier':float(brier_score_loss(y,p)),
        'logloss':float(log_loss(y,p,labels=[0,1])),
        'ap':float(average_precision_score(y,p)),
        'mean_year_auc':float(np.nanmean(ya)) if np.isfinite(ya).any() else np.nan,
        'min_year_auc':float(np.nanmin(ya)) if np.isfinite(ya).any() else np.nan,
    }


def read_adv():
    raw=gzip.decompress(base64.b64decode(ADV.read_text().strip()))
    return pd.read_csv(io.BytesIO(raw))


def norm(d):
    d=d.copy()
    if 'name' in d.columns and 'pfr_name' not in d.columns:d=d.rename(columns={'name':'pfr_name'})
    return d


def main():
    if not ADV.exists() or not ORG.exists():
        raise FileNotFoundError(f'missing inputs advanced={ADV.exists()} original={ORG.exists()}')
    a=norm(read_adv());o=norm(pd.read_csv(ORG))
    a=a[['season','position','target','pfr_name','y','pred']].rename(columns={'y':'y_adv','pred':'pred_adv'})
    o=o[['season','position','target','pfr_name','y','pred']].rename(columns={'y':'y_org','pred':'pred_org'})
    m=a.merge(o,on=['season','position','target','pfr_name'],how='inner',validate='one_to_one')
    if len(m):
        md=np.nanmax(np.abs(pd.to_numeric(m.y_adv,errors='coerce')-pd.to_numeric(m.y_org,errors='coerce')))
        if np.isfinite(md) and md>1e-6:raise ValueError(f'target mismatch {md}')
    m['y']=m.y_adv.astype(int)
    rows=[];sel=[];preds=[]
    for (pos,target),g in m.groupby(['position','target']):
        local=[]
        for w in WEIGHTS:
            q=g.copy();q['pred']=w*q.pred_adv+(1-w)*q.pred_org
            r=metrics(q);r.update(position=pos,target=target,w_advanced=float(w),w_original=float(1-w));rows.append(r);local.append(r)
        mm=pd.DataFrame(local)
        for c,asc in [('auc',False),('brier',True),('logloss',True),('ap',False),('mean_year_auc',False),('min_year_auc',False)]:
            mm['r_'+c]=mm[c].rank(method='min',ascending=asc,na_option='bottom')
        mm['selection_score']=1.2*mm.r_auc+1.2*mm.r_brier+.75*mm.r_logloss+.35*mm.r_ap+.55*mm.r_mean_year_auc+.35*mm.r_min_year_auc
        win=mm.sort_values(['selection_score','auc','brier','logloss'],ascending=[True,False,True,True]).iloc[0]
        sel.append(win.to_dict())
        q=g.copy();w=float(win.w_advanced);q['pred']=w*q.pred_adv+(1-w)*q.pred_org;q['w_advanced']=w;q['w_original']=1-w;preds.append(q[['season','position','target','pfr_name','y','pred','pred_adv','pred_org','w_advanced','w_original']])
    pd.DataFrame(rows).to_csv(OUT/'weight_search_2019_2022.csv',index=False)
    pd.DataFrame(sel).to_csv(OUT/'selected_weights.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/'selected_validation_oof.csv',index=False)
    print(pd.DataFrame(sel)[['position','target','w_advanced','w_original','auc','brier','mean_year_auc','min_year_auc','selection_score']].to_string(index=False))

if __name__=='__main__':main()
