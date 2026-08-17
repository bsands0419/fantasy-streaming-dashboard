from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import modeling as b
import stage2 as s2
import stage3 as s3

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results_v4"
OUT.mkdir(parents=True,exist_ok=True)
POS=["QB","RB","WR","TE"]; DEV=list(range(2013,2019)); VAL=list(range(2019,2023)); TEST=2023
SEED=20260816

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);ok=np.isfinite(y)&np.isfinite(p);y=y[ok];p=p[ok]
    if len(y)<4:return dict(n=len(y),mae=np.nan,rmse=np.nan,r2=np.nan,pearson=np.nan,spearman=np.nan)
    return dict(n=len(y),mae=mean_absolute_error(y,p),rmse=mean_squared_error(y,p)**.5,
                r2=r2_score(y,p),pearson=pearsonr(y,p).statistic,spearman=spearmanr(y,p).statistic)

def rank_metrics(z):
    a=[]
    for _,g in z.groupby("season"):
        if len(g)<4:continue
        sp=spearmanr(g.primary_ppg,g.pred).statistic
        nd=ndcg_score([g.primary_ppg.to_numpy()],[g.pred.to_numpy()])
        n=min(5,len(g)); chosen=g.nlargest(n,"pred").primary_ppg.mean(); oracle=g.nlargest(n,"primary_ppg").primary_ppg.mean()
        a.append((sp,nd,chosen,oracle-chosen))
    if not a:return dict(yearly_spearman=np.nan,ndcg=np.nan,top5_actual=np.nan,top5_regret=np.nan)
    x=np.asarray(a,float)
    return dict(yearly_spearman=np.nanmean(x[:,0]),ndcg=np.nanmean(x[:,1]),top5_actual=np.nanmean(x[:,2]),top5_regret=np.nanmean(x[:,3]))

def score(m,r,sd):
    def v(x,d=0):return d if not np.isfinite(x) else x
    return .35*v(m["spearman"])+.20*v(r["yearly_spearman"])+.20*v(r["ndcg"])-.15*v(m["mae"],9)/sd-.10*v(r["top5_regret"],9)/sd

def add_role(p,w):
    p=p.copy();w=w.copy();idc=b.first_existing(w,["player_id","gsis_id"])
    for c in ["attempts","passing_attempts","carries","rushing_attempts","targets"]:
        if c in w:w[c]=pd.to_numeric(w[c],errors="coerce").fillna(0)
    w["season"]=pd.to_numeric(w.season,errors="coerce"); vals=[]
    for _,r in p.iterrows():
        z=w[(w[idc]==r.get("gsis_id"))&w.season.between(int(r.season),int(r.season)+2)] if idc and pd.notna(r.get("gsis_id")) else w.iloc[:0]
        if z.empty: vals.append(0);continue
        if r.position=="QB":
            q=(z["attempts"] if "attempts" in z else z.get("passing_attempts",0))+(z["carries"] if "carries" in z else z.get("rushing_attempts",0)); vals.append(int((q>=15).sum()))
        elif r.position=="RB":
            q=(z["carries"] if "carries" in z else z.get("rushing_attempts",0))+z.get("targets",0); vals.append(int((q>=5).sum()))
        else:
            q=z.get("targets",0)+(z["carries"] if "carries" in z else z.get("rushing_attempts",0)); vals.append(int((q>=4).sum()))
    p["meaningful_games3"]=vals;p["role3"]=(p.meaningful_games3>=6).astype(int);return p

def add_classpct(p):
    p=p.copy()
    cols=[c for c in p if c.startswith(("pass_","rush_","rec_","rb_","qb_")) and pd.api.types.is_numeric_dtype(p[c])]
    for c in cols:p["classpct_"+c]=p.groupby(["season","position"])[c].rank(pct=True)
    return p

def groups(p,pos):
    g=s3.groups_v3(p,pos);pct=[c for c in p if c.startswith("classpct_") and pd.api.types.is_numeric_dtype(p[c])]
    g["full_landing_scout_classpct"]=list(dict.fromkeys(g["full_landing_scout"]+pct))
    return g

def capmodel():
    return Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("s",StandardScaler()),("m",Ridge(alpha=10))])
def direct():
    return {
      "ridge":Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("s",StandardScaler()),("m",Ridge(alpha=20))]),
      "extra":Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("m",ExtraTreesRegressor(n_estimators=550,min_samples_leaf=6,max_features=.7,n_jobs=-1,random_state=SEED))]),
      "xgb":Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("m",XGBRegressor(n_estimators=500,max_depth=2,learning_rate=.02,min_child_weight=10,subsample=.85,colsample_bytree=.7,reg_alpha=.75,reg_lambda=12,n_jobs=2,random_state=SEED))])
    }
def residual():
    return {
      "ridgeR":Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("s",StandardScaler()),("m",Ridge(alpha=35))]),
      "xgbR":Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("m",XGBRegressor(n_estimators=425,max_depth=2,learning_rate=.02,min_child_weight=10,subsample=.85,colsample_bytree=.7,reg_alpha=1,reg_lambda=14,n_jobs=2,random_state=SEED+1))])
    }
def conditional():
    return {
      "ridgeH":Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("s",StandardScaler()),("m",Ridge(alpha=25))]),
      "xgbH":Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("m",XGBRegressor(n_estimators=425,max_depth=2,learning_rate=.02,min_child_weight=8,subsample=.85,colsample_bytree=.7,reg_alpha=.75,reg_lambda=12,n_jobs=2,random_state=SEED+2))])
    }
def roleclf():
    return Pipeline([("i",SimpleImputer(strategy="median",add_indicator=True)),("s",StandardScaler()),("m",LogisticRegression(C=.2,max_iter=5000,class_weight="balanced"))])

def fold(tr,te,arch,feats,mn,cap):
    if arch=="direct":
        m=clone(direct()[mn]);m.fit(tr[feats],tr.primary_ppg);return m.predict(te[feats])
    if arch=="residual":
        cm=clone(capmodel());cm.fit(tr[cap],tr.primary_ppg);res=tr.primary_ppg.to_numpy()-cm.predict(tr[cap])
        rm=clone(residual()[mn]);rm.fit(tr[feats],res);return cm.predict(te[cap])+rm.predict(te[feats])
    cl=roleclf()
    if tr.role3.nunique()>1:cl.fit(tr[feats],tr.role3);pr=cl.predict_proba(te[feats])[:,1]
    else:pr=np.repeat(tr.role3.mean(),len(te))
    rt=tr[tr.role3==1];non=float(tr.loc[tr.role3==0,"primary_ppg"].mean()) if (tr.role3==0).any() else 0
    if len(rt)<25:cond=np.repeat(rt.primary_ppg.mean() if len(rt) else tr.primary_ppg.mean(),len(te))
    else:
        cm=clone(conditional()[mn]);cm.fit(rt[feats],rt.primary_ppg);cond=cm.predict(te[feats])
    return pr*np.maximum(cond,0)+(1-pr)*max(non,0)

def wf(d,years,arch,feats,mn,cap):
    out=[]
    for y in years:
        tr=d[d.season<y];te=d[d.season==y]
        if len(tr)<40 or te.empty:continue
        q=te[["season","pfr_name","primary_ppg"]].copy();q["pred"]=fold(tr,te,arch,feats,mn,cap);out.append(q)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def evalpos(p,pos):
    d=p[(p.position==pos)&(p.season<=2023)&(p.target_valid==1)].copy();g=groups(d,pos);cap=g["capital"]
    sd=float(d.loc[d.season<2019,"primary_ppg"].std()) or 1
    specs=[]
    for fs in ["capital_production","capital_prod_context","full_landing_scout","full_landing_scout_classpct"]:
        if fs not in g or not g[fs]:continue
        specs += [("direct",fs,m) for m in direct()]+[("residual",fs,m) for m in residual()]+[("hurdle",fs,m) for m in conditional()]
    dev=[]
    for arch,fs,mn in specs:
        z=wf(d,DEV,arch,g[fs],mn,cap)
        if z.empty:continue
        m=metrics(z.primary_ppg,z.pred);r=rank_metrics(z);dev.append(dict(architecture=arch,feature_set=fs,model=mn,score=score(m,r,sd),**m,**r))
    dev=pd.DataFrame(dev).sort_values("score",ascending=False)
    val=[];stores={}
    for _,x in dev.head(12).iterrows():
        k=(x.architecture,x.feature_set,x.model);z=wf(d,VAL,k[0],g[k[1]],k[2],cap)
        if z.empty:continue
        m=metrics(z.primary_ppg,z.pred);r=rank_metrics(z);val.append(dict(architecture=k[0],feature_set=k[1],model=k[2],score=score(m,r,sd),**m,**r));stores[k]=z
    val=pd.DataFrame(val).sort_values("score",ascending=False);best=val.iloc[0];ka=(best.architecture,best.feature_set,best.model);za=stores[ka]
    bm=metrics(za.primary_ppg,za.pred);br=rank_metrics(za);bs=score(bm,br,sd);blend=dict(weight_a=1.0,other=None)
    for _,x in val.head(6).iterrows():
        kb=(x.architecture,x.feature_set,x.model)
        if kb==ka or kb[0]==ka[0]:continue
        z=za.merge(stores[kb][["season","pfr_name","pred"]],on=["season","pfr_name"],suffixes=("_a","_b"))
        for wa in [.35,.5,.65]:
            pp=wa*z.pred_a+(1-wa)*z.pred_b;m=metrics(z.primary_ppg,pp);r=rank_metrics(z.assign(pred=pp));sc=score(m,r,sd)
            if sc>bs+.003:bs=sc;bm=m;br=r;blend=dict(weight_a=wa,other=kb)
    cz=wf(d,VAL,"direct",cap,"ridge",cap);cm=metrics(cz.primary_ppg,cz.pred);cr=rank_metrics(cz)
    tr=d[d.season<TEST];te=d[d.season==TEST]
    fp=fold(tr,te,ka[0],g[ka[1]],ka[2],cap)
    if blend["other"] is not None:
        kb=tuple(blend["other"]);fp=blend["weight_a"]*fp+(1-blend["weight_a"])*fold(tr,te,kb[0],g[kb[1]],kb[2],cap)
    f=te[["season","pfr_name","primary_ppg"]].copy();f["pred"]=fp;fm=metrics(f.primary_ppg,f.pred);fr=rank_metrics(f)
    c=clone(capmodel());c.fit(tr[cap],tr.primary_ppg);cf=te[["season","pfr_name","primary_ppg"]].copy();cf["pred"]=c.predict(te[cap]);cfm=metrics(cf.primary_ppg,cf.pred);cfr=rank_metrics(cf)
    return dict(position=pos,n=len(d),selected=dict(architecture=ka[0],feature_set=ka[1],model=ka[2]),blend=blend,
                validation={**bm,**br},capital_validation={**cm,**cr},final_2023={**fm,**fr},capital_2023={**cfm,**cfr}),dev,val,f

def main():
    pa=b.load_cfb_table("passing");ru=b.load_cfb_table("rushing");re=b.load_cfb_table("receiving");team=b.load_cfb_table("team_summaries")
    pa,ru,re=s2.prep_college(pa,ru,re,team)
    draft,players,combine,weekly=b.load_nflverse();draft["season"]=pd.to_numeric(draft.season,errors="coerce")
    draft=draft[draft.category.isin(POS)&draft.season.between(b.TRAIN_DRAFT_START,2026)].copy()
    p=s2.build_profiles(draft,pa,ru,re);p=s2.add_nfl_meta(p,players,combine);p=s2.add_nfl_targets(p,weekly)
    p=s3.add_landing_features(p,draft,weekly);p=s3.add_scouting_surprise(p);p=add_role(p,weekly);p=add_classpct(p)
    p.to_csv(OUT/"prospect_pool_v4.csv",index=False)
    sums=[]
    for pos in POS:
        print("stage4",pos);s,d,v,f=evalpos(p,pos);sums.append(s);d.to_csv(OUT/f"{pos}_development.csv",index=False);v.to_csv(OUT/f"{pos}_validation.csv",index=False);f.to_csv(OUT/f"{pos}_2023_test.csv",index=False)
    rows=[]
    for s in sums:
        rows.append(dict(position=s["position"],architecture=s["selected"]["architecture"],feature_set=s["selected"]["feature_set"],model=s["selected"]["model"],blend=str(s["blend"]),
                         **{"val_"+k:v for k,v in s["validation"].items()},**{"capval_"+k:v for k,v in s["capital_validation"].items()},
                         **{"test_"+k:v for k,v in s["final_2023"].items()},**{"captest_"+k:v for k,v in s["capital_2023"].items()}))
    pd.DataFrame(rows).to_csv(OUT/"model_summary_v4.csv",index=False)
    manifest=dict(generated_utc=datetime.now(timezone.utc).isoformat(),stage=4,development_years=DEV,validation_years=VAL,final_test_year=TEST,
                  precommit="Written before reading 2023 Stage-3/Stage-4 results; tests direct vs residual vs hurdle and within-class percentiles.",summaries=sums)
    (OUT/"manifest_v4.json").write_text(json.dumps(manifest,indent=2,default=float))
    print(json.dumps(manifest,indent=2,default=float))
if __name__=="__main__":main()
