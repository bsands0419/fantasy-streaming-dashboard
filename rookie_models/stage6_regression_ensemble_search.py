from __future__ import annotations

import base64
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results_stage6_newmodels'
ADV_OOF = R / 'advanced_selected_validation_regression_oof.b64'
ADV_2023 = R / 'advanced_selected_confirmation_2023_regression.csv'
ORG_OOF = R / 'original_branch' / 'selected_validation_oof.csv'
ORG_2023 = R / 'original_branch' / 'confirmation_2023_predictions.csv'
OUT = R / 'regression_ensemble'
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ['QB','RB','WR','TE']
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)  # advanced-model weight


def sp(y, p):
    try:
        z = spearmanr(y, p).statistic
        return float(z) if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def metrics(d: pd.DataFrame) -> dict:
    ys = [sp(g.y, g.pred) for _, g in d.groupby('season')]
    return {
        'n': len(d),
        'mae': float(mean_absolute_error(d.y, d.pred)),
        'rmse': float(mean_squared_error(d.y, d.pred) ** 0.5),
        'spearman': sp(d.y, d.pred),
        'mean_year_spearman': float(np.nanmean(ys)),
        'min_year_spearman': float(np.nanmin(ys)),
    }


def read_adv_oof() -> pd.DataFrame:
    enc = ADV_OOF.read_text().strip()
    raw = gzip.decompress(base64.b64decode(enc))
    return pd.read_csv(io.BytesIO(raw))


def normalize_name_col(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if 'name' in d.columns and 'pfr_name' not in d.columns:
        d = d.rename(columns={'name':'pfr_name'})
    return d


def matched(a: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    a = normalize_name_col(a)
    o = normalize_name_col(o)
    keep_a = ['season','position','pfr_name','y','pred']
    keep_o = ['season','position','pfr_name','y','pred']
    a = a[keep_a].rename(columns={'y':'y_adv','pred':'pred_adv'})
    o = o[keep_o].rename(columns={'y':'y_org','pred':'pred_org'})
    m = a.merge(o, on=['season','position','pfr_name'], how='inner', validate='one_to_one')
    if len(m):
        diff = np.nanmax(np.abs(pd.to_numeric(m.y_adv, errors='coerce') - pd.to_numeric(m.y_org, errors='coerce')))
        if np.isfinite(diff) and diff > 1e-6:
            raise ValueError(f'target mismatch max={diff}')
    m['y'] = pd.to_numeric(m.y_adv, errors='coerce')
    return m


def search_position(m: pd.DataFrame, pos: str):
    g = m[m.position.eq(pos)].copy()
    rows=[]; preds=[]
    for w in WEIGHTS:
        q=g.copy()
        q['pred'] = w*q.pred_adv + (1-w)*q.pred_org
        r=metrics(q);r.update(position=pos,w_advanced=float(w),w_original=float(1-w));rows.append(r)
        z=q[['season','position','pfr_name','y']].copy();z['pred']=q.pred;z['w_advanced']=w;preds.append(z)
    mm=pd.DataFrame(rows)
    # Multi-objective temporal selection. Error is primary; ranking/stability prevent a fragile MAE-only choice.
    for c,asc in [('mae',True),('rmse',True),('spearman',False),('mean_year_spearman',False),('min_year_spearman',False)]:
        mm['r_'+c]=mm[c].rank(method='min',ascending=asc,na_option='bottom')
    mm['selection_score'] = 1.35*mm.r_mae + 1.0*mm.r_rmse + 1.0*mm.r_spearman + .55*mm.r_mean_year_spearman + .35*mm.r_min_year_spearman
    win=mm.sort_values(['selection_score','mae','spearman','rmse'],ascending=[True,True,False,True]).iloc[0]
    return mm, pd.concat(preds,ignore_index=True), win


def main():
    required=[ADV_OOF, ADV_2023, ORG_OOF, ORG_2023]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('missing ensemble inputs: ' + ', '.join(missing))

    avo=read_adv_oof(); ovo=pd.read_csv(ORG_OOF)
    mv=matched(avo,ovo)
    all_search=[]; selected=[]; selected_oof=[]
    for pos in POSITIONS:
        mm, pp, win=search_position(mv,pos)
        all_search.append(mm);selected.append(win.to_dict())
        selected_oof.append(pp[(pp.position.eq(pos)) & np.isclose(pp.w_advanced,float(win.w_advanced))].copy())

    search=pd.concat(all_search,ignore_index=True)
    sel=pd.DataFrame(selected)
    oof=pd.concat(selected_oof,ignore_index=True)
    search.to_csv(OUT/'weight_search_2019_2022.csv',index=False)
    sel.to_csv(OUT/'selected_weights.csv',index=False)
    oof.to_csv(OUT/'selected_validation_oof.csv',index=False)

    # One-shot confirmation using the already-frozen weight chosen above.
    a23=pd.read_csv(ADV_2023);o23=pd.read_csv(ORG_2023)
    m23=matched(a23,o23)
    cm=[]; cp=[]
    for _,r in sel.iterrows():
        pos=r.position;w=float(r.w_advanced)
        q=m23[m23.position.eq(pos)].copy();q['pred']=w*q.pred_adv+(1-w)*q.pred_org
        z=metrics(q);z.update(position=pos,w_advanced=w,w_original=1-w);cm.append(z)
        q['w_advanced']=w;q['w_original']=1-w;cp.append(q[['season','position','pfr_name','y','pred','pred_adv','pred_org','w_advanced','w_original']])
    pd.DataFrame(cm).to_csv(OUT/'confirmation_2023_metrics.csv',index=False)
    pd.concat(cp,ignore_index=True).to_csv(OUT/'confirmation_2023_predictions.csv',index=False)
    print('SELECTED WEIGHTS')
    print(sel[['position','w_advanced','w_original','mae','rmse','spearman','mean_year_spearman','min_year_spearman','selection_score']].to_string(index=False))
    print('2023 CONFIRMATION')
    print(pd.DataFrame(cm).to_string(index=False))


if __name__=='__main__':
    main()
