from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'results_stage6_newmodels'
ORIG=BASE/'original_branch'
ADV_VAL=BASE/'advanced_selected_validation_regression_oof.csv'
ADV_23=BASE/'advanced_selected_2023_regression.csv'
OUT=BASE/'all_data_ensemble'
OUT.mkdir(parents=True,exist_ok=True)
WEIGHTS=np.round(np.arange(0,1.0001,.05),2)


def nn(x):
    s=str(x).lower().replace('’',"'")
    s=re.sub(r'[^a-z0-9]+','',s)
    return s


def sp(a,b):
    try:
        z=spearmanr(a,b).statistic
        return float(z) if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def metrics(d,pcol='pred'):
    vals=[]
    for _,g in d.groupby('season'):
        vals.append(sp(g.y,g[pcol]))
    return {'n':len(d),'mae':mean_absolute_error(d.y,d[pcol]),'rmse':mean_squared_error(d.y,d[pcol])**.5,'spearman':sp(d.y,d[pcol]),'mean_year_spearman':float(np.nanmean(vals)),'min_year_spearman':float(np.nanmin(vals))}


def prep(p,predname):
    d=pd.read_csv(p)
    d['season']=pd.to_numeric(d.season,errors='coerce')
    d['name_key']=d.pfr_name.map(nn)
    d=d.rename(columns={'pred':predname})
    return d


def main():
    need=[ADV_VAL,ADV_23,ORIG/'selected_validation_oof.csv',ORIG/'confirmation_2023_predictions.csv']
    missing=[str(x) for x in need if not x.exists()]
    if missing:
        print('WAITING_FOR_INPUTS')
        print('\n'.join(missing))
        return
    a=prep(ADV_VAL,'adv_pred')
    o=prep(ORIG/'selected_validation_oof.csv','orig_pred')
    m=a[['season','position','pfr_name','name_key','y','adv_pred']].merge(o[['season','position','name_key','y','orig_pred']],on=['season','position','name_key'],how='inner',suffixes=('','_orig'))
    if len(m)==0: raise RuntimeError('No matched validation rows')
    if np.nanmax(np.abs(pd.to_numeric(m.y,errors='coerce')-pd.to_numeric(m.y_orig,errors='coerce')))>1e-6: raise RuntimeError('Outcome mismatch in OOF bridge')
    searches=[];selected=[]
    for pos,g in m.groupby('position'):
        rows=[]
        for w in WEIGHTS:
            q=g.copy();q['pred']=(1-w)*q.adv_pred+w*q.orig_pred
            r=metrics(q);r.update(position=pos,original_weight=float(w),advanced_weight=float(1-w));rows.append(r)
        z=pd.DataFrame(rows)
        for c,asc in [('mae',True),('rmse',True),('spearman',False),('mean_year_spearman',False),('min_year_spearman',False)]:
            z['r_'+c]=z[c].rank(method='min',ascending=asc)
        z['selection_score']=z.r_mae+z.r_rmse+z.r_spearman+.6*z.r_mean_year_spearman+.6*z.r_min_year_spearman
        win=z.sort_values(['selection_score','mae','spearman'],ascending=[True,True,False]).iloc[0]
        selected.append(win.to_dict());searches.append(z)
    sr=pd.DataFrame(selected)
    pd.concat(searches,ignore_index=True).to_csv(OUT/'validation_blend_weight_search.csv',index=False)
    sr.to_csv(OUT/'selected_blend_weights.csv',index=False)
    a23=prep(ADV_23,'adv_pred')
    o23=prep(ORIG/'confirmation_2023_predictions.csv','orig_pred')
    c=a23[['season','position','pfr_name','name_key','y','adv_pred']].merge(o23[['season','position','name_key','y','orig_pred']],on=['season','position','name_key'],how='inner',suffixes=('','_orig'))
    out=[];preds=[]
    for pos,g in c.groupby('position'):
        w=float(sr.loc[sr.position.eq(pos),'original_weight'].iloc[0])
        q=g.copy();q['pred']=(1-w)*q.adv_pred+w*q.orig_pred;q['original_weight']=w;q['advanced_weight']=1-w
        r=metrics(q);r.update(position=pos,original_weight=w,advanced_weight=1-w,adv_mae=mean_absolute_error(q.y,q.adv_pred),adv_spearman=sp(q.y,q.adv_pred),orig_mae=mean_absolute_error(q.y,q.orig_pred),orig_spearman=sp(q.y,q.orig_pred));out.append(r)
        preds.append(q[['season','position','pfr_name','y','adv_pred','orig_pred','pred','advanced_weight','original_weight']])
    pd.DataFrame(out).to_csv(OUT/'confirmation_2023_ensemble.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/'confirmation_2023_ensemble_predictions.csv',index=False)
    print('SELECTED BLENDS')
    print(sr[['position','advanced_weight','original_weight','mae','rmse','spearman','mean_year_spearman','min_year_spearman']].to_string(index=False))
    print('2023 FIXED CONFIRMATION')
    print(pd.DataFrame(out).to_string(index=False))

if __name__=='__main__':
    main()
