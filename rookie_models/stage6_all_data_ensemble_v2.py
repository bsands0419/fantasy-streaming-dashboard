from __future__ import annotations
from pathlib import Path
import base64
import gzip
import io
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parent
R=ROOT/'results_stage6_newmodels'
# Canonical selected advanced artifacts. Do not substitute convenience/manual bridges.
ADV_VAL=R/'advanced_selected_validation_regression_oof.b64'
ADV_23=R/'advanced_selected_confirmation_2023_regression.csv'
OUT=R/'all_data_ensemble_v2'
OUT.mkdir(parents=True,exist_ok=True)
BRANCHES={'broad':R/'original_branch','fast':R/'original_fast'}
WEIGHTS=np.round(np.arange(0,1.0001,.05),2) # original-data weight


def nk(x): return re.sub(r'[^a-z0-9]+','',str(x).lower().replace('’',"'"))
def sp(a,b):
    try:
        z=spearmanr(a,b).statistic
        return float(z) if np.isfinite(z) else np.nan
    except Exception:return np.nan

def met(d):
    ys=[sp(g.y,g.pred) for _,g in d.groupby('season')]
    return {'n':len(d),'mae':mean_absolute_error(d.y,d.pred),'rmse':mean_squared_error(d.y,d.pred)**.5,'spearman':sp(d.y,d.pred),'mean_year_spearman':float(np.nanmean(ys)),'min_year_spearman':float(np.nanmin(ys))}

def prep_csv(path,pred):
    d=pd.read_csv(path);d['season']=pd.to_numeric(d.season,errors='coerce');d['name_key']=d.pfr_name.map(nk)
    if pred not in d.columns and 'pred' in d.columns:d=d.rename(columns={'pred':pred})
    return d

def prep_adv_oof(path):
    enc=path.read_text().strip();raw=gzip.decompress(base64.b64decode(enc));d=pd.read_csv(io.BytesIO(raw))
    if 'name' in d.columns and 'pfr_name' not in d.columns:d=d.rename(columns={'name':'pfr_name'})
    d['season']=pd.to_numeric(d.season,errors='coerce');d['name_key']=d.pfr_name.map(nk)
    if 'adv_pred' not in d.columns and 'pred' in d.columns:d=d.rename(columns={'pred':'adv_pred'})
    return d

def join(a,o,opred):
    m=a[['season','position','pfr_name','name_key','y','adv_pred']].merge(o[['season','position','name_key','y',opred]],on=['season','position','name_key'],how='inner',suffixes=('','_org'))
    if len(m) and np.nanmax(np.abs(pd.to_numeric(m.y,errors='coerce')-pd.to_numeric(m.y_org,errors='coerce')))>1e-6:raise ValueError('target mismatch')
    return m

def rank_candidates(z):
    z=z.copy()
    for c,asc in [('mae',True),('rmse',True),('spearman',False),('mean_year_spearman',False),('min_year_spearman',False)]:z['r_'+c]=z[c].rank(method='min',ascending=asc,na_option='bottom')
    z['selection_score']=1.35*z.r_mae+z.r_rmse+z.r_spearman+.55*z.r_mean_year_spearman+.35*z.r_min_year_spearman
    return z

def main():
    if not ADV_VAL.exists() or not ADV_23.exists():raise FileNotFoundError('canonical advanced artifacts missing')
    a=prep_adv_oof(ADV_VAL);a23=prep_csv(ADV_23,'adv_pred')
    searches=[]
    for bname,bdir in BRANCHES.items():
        vp=bdir/'selected_validation_oof.csv'
        if not vp.exists():continue
        o=prep_csv(vp,'orig_pred');m=join(a,o,'orig_pred')
        for pos,g in m.groupby('position'):
            for w in WEIGHTS:
                q=g.copy();q['pred']=(1-w)*q.adv_pred+w*q.orig_pred
                r=met(q);r.update(position=pos,original_branch=bname,original_weight=float(w),advanced_weight=float(1-w),matched_n=len(g));searches.append(r)
    if not searches:
        print('WAITING_FOR_ORIGINAL_BRANCH_OUTPUTS');return
    allz=[];wins=[]
    for pos,g in pd.DataFrame(searches).groupby('position'):
        z=rank_candidates(g);allz.append(z);wins.append(z.sort_values(['selection_score','mae','spearman'],ascending=[True,True,False]).iloc[0].to_dict())
    zall=pd.concat(allz,ignore_index=True);sel=pd.DataFrame(wins)
    zall.to_csv(OUT/'validation_branch_weight_search.csv',index=False);sel.to_csv(OUT/'selected_all_data_models.csv',index=False)
    confirms=[];preds=[]
    for _,r in sel.iterrows():
        bdir=BRANCHES[r.original_branch];p23=bdir/'confirmation_2023_predictions.csv'
        if not p23.exists():continue
        o23=prep_csv(p23,'orig_pred');m23=join(a23,o23,'orig_pred');q=m23[m23.position.eq(r.position)].copy();w=float(r.original_weight);q['pred']=(1-w)*q.adv_pred+w*q.orig_pred
        x=met(q);x.update(position=r.position,original_branch=r.original_branch,original_weight=w,advanced_weight=1-w,advanced_only_mae=mean_absolute_error(q.y,q.adv_pred),advanced_only_spearman=sp(q.y,q.adv_pred),original_only_mae=mean_absolute_error(q.y,q.orig_pred),original_only_spearman=sp(q.y,q.orig_pred));confirms.append(x)
        q['original_branch']=r.original_branch;q['original_weight']=w;q['advanced_weight']=1-w;preds.append(q[['season','position','pfr_name','y','adv_pred','orig_pred','pred','original_branch','advanced_weight','original_weight']])
    if confirms:pd.DataFrame(confirms).to_csv(OUT/'confirmation_2023_fixed.csv',index=False)
    if preds:pd.concat(preds,ignore_index=True).to_csv(OUT/'confirmation_2023_fixed_predictions.csv',index=False)
    print('SELECTED');print(sel[['position','original_branch','advanced_weight','original_weight','mae','rmse','spearman','mean_year_spearman','min_year_spearman']].to_string(index=False))
    if confirms:print('2023');print(pd.DataFrame(confirms).to_string(index=False))

if __name__=='__main__':main()
