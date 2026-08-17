from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parent
IN=ROOT/'results_v3'
OUT=ROOT/'results_v4_audit'
OUT.mkdir(parents=True,exist_ok=True)
POSITIONS=['QB','RB','WR','TE']
RNG=np.random.default_rng(20260817)


def met(df):
    y=pd.to_numeric(df['primary_ppg'],errors='coerce').to_numpy(float)
    p=pd.to_numeric(df['pred'],errors='coerce').to_numpy(float)
    ok=np.isfinite(y)&np.isfinite(p); y=y[ok]; p=p[ok]
    if len(y)<4:return {'n':len(y),'mae':np.nan,'rmse':np.nan,'pearson':np.nan,'spearman':np.nan}
    return {'n':len(y),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**0.5,'pearson':pearsonr(y,p).statistic,'spearman':spearmanr(y,p).statistic}


def bootstrap(df,n=3000):
    d=df.reset_index(drop=True); vals=[]
    for _ in range(n):
        ix=RNG.integers(0,len(d),len(d)); m=met(d.iloc[ix]); vals.append(m)
    z=pd.DataFrame(vals)
    out={}
    for c in ['mae','rmse','pearson','spearman']:
        out[c+'_lo']=z[c].quantile(.025); out[c+'_med']=z[c].quantile(.5); out[c+'_hi']=z[c].quantile(.975)
    return out


def yearly(df):
    rows=[]
    for y,g in df.groupby('season'):
        m=met(g);m['season']=int(y);rows.append(m)
    return pd.DataFrame(rows)


def calibration(df):
    y=pd.to_numeric(df.primary_ppg,errors='coerce');p=pd.to_numeric(df.pred,errors='coerce')
    ok=y.notna()&p.notna(); y=y[ok].to_numpy();p=p[ok].to_numpy()
    if len(y)<5:return {'intercept':np.nan,'slope':np.nan,'mean_bias':np.nan}
    slope,intercept=np.polyfit(p,y,1)
    return {'intercept':intercept,'slope':slope,'mean_bias':float(np.mean(p-y))}


def topk_capture(df,k):
    vals=[]
    for y,g in df.groupby('season'):
        n=min(k,len(g));
        if n==0:continue
        predset=set(g.nlargest(n,'pred').pfr_name)
        actualset=set(g.nlargest(n,'primary_ppg').pfr_name)
        vals.append(len(predset&actualset)/n)
    return float(np.mean(vals)) if vals else np.nan


def main():
    if not IN.exists(): raise SystemExit('results_v3 not present yet')
    summary=[]; yearly_all=[]
    for pos in POSITIONS:
        files=[IN/f'{pos}_validation_predictions.csv',IN/f'{pos}_final_2023_predictions.csv']
        if not all(x.exists() for x in files): raise SystemExit(f'missing stage3 outputs for {pos}')
        d=pd.concat([pd.read_csv(x) for x in files],ignore_index=True)
        m=met(d); ci=bootstrap(d); cal=calibration(d)
        yr=yearly(d);yr['position']=pos;yearly_all.append(yr)
        summary.append({'position':pos,**m,**ci,**cal,'top5_capture':topk_capture(d,5),'top10_capture':topk_capture(d,10),'yearly_spearman_positive_rate':float((yr.spearman>0).mean()),'yearly_mae_sd':float(yr.mae.std())})
    s=pd.DataFrame(summary);s.to_csv(OUT/'robustness_summary.csv',index=False)
    pd.concat(yearly_all,ignore_index=True).to_csv(OUT/'yearly_robustness.csv',index=False)
    # promotion guardrails: require positive rank signal, tolerable calibration, and non-pathological bootstrap floor.
    dec=[]
    for _,r in s.iterrows():
        promote=bool((r.spearman>0.10) and (r.spearman_lo>-0.05) and (0.45<r.slope<1.55) and (r.yearly_spearman_positive_rate>=0.60))
        dec.append({'position':r.position,'robustness_pass':promote,'reason':'pass' if promote else 'fails one or more stability/calibration guardrails'})
    pd.DataFrame(dec).to_csv(OUT/'promotion_guardrails.csv',index=False)
    report=['# Stage 4 robustness audit','', 'This stage does not tune the model. It audits frozen Stage 3 predictions using bootstrap uncertainty, calibration, year-to-year stability, and top-k capture.','']
    for _,r in s.iterrows():
        report += [f"## {r.position}",f"MAE {r.mae:.3f}; Spearman {r.spearman:.3f} (95% bootstrap {r.spearman_lo:.3f} to {r.spearman_hi:.3f}); Pearson {r.pearson:.3f}; calibration slope {r.slope:.3f}; yearly positive-Spearman rate {100*r.yearly_spearman_positive_rate:.1f}%; top-5 capture {100*r.top5_capture:.1f}%; top-10 capture {100*r.top10_capture:.1f}%.",'']
    (OUT/'REPORT.md').write_text('\n'.join(report))
    (OUT/'manifest.json').write_text(json.dumps({'stage':'4-audit','bootstrap_reps':3000,'seed':20260817,'purpose':'robustness audit only; no model tuning'},indent=2))

if __name__=='__main__':main()
