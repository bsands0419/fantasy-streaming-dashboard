"""Chronological, reproducible K/DST research. Never fits on evaluation seasons."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_k,'2')
import sys, json, warnings, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend/runtime'))
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, PoissonRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.base import clone, RegressorMixin, BaseEstimator
from catboost import CatBoostRegressor as _CatBoostRegressor
class CatBoostRegressor(_CatBoostRegressor, RegressorMixin, BaseEstimator):
    """scikit-learn 1.8 estimator-tag compatibility for CatBoost 1.2.8."""
    pass
from championship_models_v4_2 import ChampionshipDSTModelV42, ChampionshipKickerModelV42
from championship_feature_factory import DST_V4_FEATURES
from championship_feature_factory_v4_2 import K_V42_FEATURES
from frozen_models import GenericDSTScoring, GenericKickerScoring
import joblib
warnings.filterwarnings('ignore',category=UserWarning)
OUT=ROOT/'research/streaming/results'
CACHE=ROOT/'research/streaming/cache'

def load_data():
    return (pd.read_pickle(ROOT/'backend/runtime/team_history.pkl.gz').sort_values(['season','week','game_id','team']).reset_index(drop=True),
            pd.read_pickle(ROOT/'backend/runtime/kicker_history.pkl.gz').sort_values(['season','week','game_id','team','player_id']).reset_index(drop=True))

def pa_expectation(m, X, rounded=False):
    """Exact mean of the existing clipped Gaussian PA simulation, without MC noise."""
    out=np.zeros(len(X))
    for dist,w in [(m.ridge,1-m.bayes_weight),(m.bayes,m.bayes_weight)]:
        mu=dist.predict(X); sd=dist.residual_sd_
        boundaries=np.array([0,6,13,20,27,34],float)+(0.5 if rounded else 0)
        cdf=norm.cdf((boundaries[None,:]-mu[:,None])/sd)
        out+=w*(-4+cdf@np.array([3,3,3,1,1,3]))
    return out

def baseline_predict(model, X, pos):
    if pos=='K':
        fam=model._component_families(X)
        comp=model._analytic_component_score(fam,GenericKickerScoring())
        return sum(w*p for w,p in zip(model.PRESET_WEIGHTS,[*comp,model._direct_prediction(X,GenericKickerScoring())]))
    s=GenericDSTScoring()
    weights=dict(def_sacks=s.sack,def_interceptions=s.interception,fumble_recovery_opp=s.fumble_recovery,
                 def_safeties=s.safety,blocks=s.blocked_kick,def_tds=s.defensive_td,special_teams_tds=s.special_teams_td)
    p=sum(w*model.models[t].predict(X) for t,w in weights.items())+pa_expectation(model.pa_model,X)
    cat=model.direct_cat.predict(model.direct_imp.transform(X[model.direct_features]))
    return model.core_weight*p+(1-model.core_weight)*cat

def augment(t,k):
    """New variables computed strictly from previous completed weeks."""
    t=t.copy();k=k.copy()
    # Opponent outcome histories: prior points/attempts conceded, not future outcomes.
    kk=k.groupby(['game_id','team'],as_index=False).agg(k_points=('k_bucket_points','sum'),long_att=('fg50_att','sum'),pat_made=('pat_made','sum'))
    raw=t.merge(kk,on=['game_id','team'],how='left',validate='one_to_one')
    opp=raw[['game_id','team','fg_att','pat_att','k_points','plays_off','long_att']].rename(columns={'team':'opponent_team',**{c:'allowed_'+c for c in ['fg_att','pat_att','k_points','plays_off','long_att']}})
    raw=raw.merge(opp,on=['game_id','opponent_team'],how='left',validate='many_to_one')
    raw['sack_conversion']=raw.def_sacks/(raw.def_qb_hits+1)
    raw['recovery_luck']=raw.fumble_recovery_opp-.5*raw.def_fumbles_forced
    raw['fg_scoring_share']=raw.fg_att/(raw.fg_att+raw.pat_att).replace(0,np.nan)
    raw['points_vs_market']=raw.points_allowed_sleeper-raw.implied_opp_total
    raws=['allowed_fg_att','allowed_pat_att','allowed_k_points','allowed_plays_off','allowed_long_att','sack_conversion','recovery_luck','fg_scoring_share','points_vs_market','k_points','long_att','fg_att','pat_att','plays_off','off_punt_rate','off_fg_share_scoring','off_td_per_play','off_epa_play','off_sack_rate_allowed','off_int_rate','def_sacks','def_qb_hits','def_interceptions','def_fumbles_forced','fumble_recovery_opp','def_tds','special_teams_tds']
    raw=raw.sort_values(['team','season','week'])
    new={}
    for c in raws:
        for span in [6,16]:
            name=f'new_{c}_ewm{span}'
            new[name]=raw.groupby('team')[c].transform(lambda s:s.shift().ewm(span=span,adjust=False).mean())
    state=pd.concat([raw[['game_id','team','opponent_team']],pd.DataFrame(new,index=raw.index)],axis=1)
    own=list(new)
    opponent=state[['game_id','team']+own].rename(columns={'team':'opponent_team',**{c:'opp_'+c for c in own}})
    state=state.merge(opponent,on=['game_id','opponent_team'],how='left',validate='many_to_one')
    for d in [t,k]:
        joined=d[['game_id','team']].merge(state.drop(columns='opponent_team'),on=['game_id','team'],how='left',validate='many_to_one')
        for c in joined.columns:
            if c not in ['game_id','team']:d[c]=joined[c].to_numpy()
        d['new_outdoor_wind']=d.wind.fillna(0)*(1-d.roof_dome)
        d['new_wind15']=np.maximum(d.new_outdoor_wind-15,0)
        d['new_cold']=np.maximum(40-d.temp.fillna(60),0)*(1-d.roof_dome)
        d['new_expected_plays']=(d.new_plays_off_ewm16+d.opp_new_allowed_plays_off_ewm16)/2
        d['new_expected_opp_plays']=(d.opp_new_plays_off_ewm16+d.new_allowed_plays_off_ewm16)/2
        d['new_fg_matchup']=(d.new_fg_att_ewm16+d.opp_new_allowed_fg_att_ewm16)/2
        d['new_long_matchup']=(d.new_long_att_ewm16+d.opp_new_allowed_long_att_ewm16)/2
        d['new_fg_market']=d.new_fg_scoring_share_ewm16*d.implied_team_total
        d['new_spread_sq']=d.team_spread**2
    return t,k

def metrics(frame,pred):
    y=frame.y.to_numpy(float);p=np.asarray(pred,float)
    z=frame[['season','week','team','y']].copy();z['p']=p
    rho=[];top1=[];top3=[];top5=[]
    for _,g in z.groupby(['season','week']):
        rho.append(spearmanr(g.y,g.p).statistic if g.p.nunique()>1 else 0)
        g=g.sort_values(['p','team'],ascending=[False,True]);top1.append(g.y.iloc[0]);top3.append(g.y.head(3).mean());top5.append(g.y.head(5).mean())
    return dict(n=len(y),mae=float(np.mean(abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))),pearson=float(np.corrcoef(y,p)[0,1]) if np.std(p)>0 else 0,
                weekly_spearman=float(np.nanmean(rho)),top1=float(np.mean(top1)),top3=float(np.mean(top3)),top5=float(np.mean(top5)))

def candidates():
    def linear(m):return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),m)
    for a in [10,100,1000]:yield f'ridge{a}',linear(Ridge(alpha=a))
    yield 'bayes',linear(BayesianRidge())
    for leaf in [16,40]:yield f'extra{leaf}',make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=300,min_samples_leaf=leaf,max_features=.7,n_jobs=2,random_state=42))
    for depth in [2,4,6]:
        yield f'cat{depth}',make_pipeline(SimpleImputer(strategy='median'),CatBoostRegressor(iterations=400,depth=depth,learning_rate=.035,l2_leaf_reg=20,loss_function='RMSE',thread_count=2,verbose=False,allow_writing_files=False,random_seed=42))
    yield 'hist',HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=20,learning_rate=.035,random_state=42)

def run(years):
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    t,k=load_data();t,k=augment(t,k)
    allpred=[];allmetrics=[]
    for pos,d,features,target,cls in [('DST',t,DST_V4_FEATURES,'dst_fantasy',ChampionshipDSTModelV42),('K',k,K_V42_FEATURES,'k_bucket_points',ChampionshipKickerModelV42)]:
        d['y']=d[target]
        extended=list(features)+[c for c in d if c.startswith(('new_','opp_new_'))]
        pd.Series(extended).to_csv(OUT/f'{pos}_feature_names.csv',index=False,header=['feature'])
        for year in years:
            tr=d[d.season<year];te=d[d.season==year]
            path=CACHE/f'baseline_{pos}_{year}.joblib'
            if path.exists(): base=joblib.load(path)
            else:
                base=cls().fit(tr);joblib.dump(base,path,compress=3)
            out=te[['season','week','game_id','team','y']].copy();out['position']=pos
            if pos=='K':out['player_id']=te.player_id
            out['v42']=baseline_predict(base,te,pos)
            out['market']=linear_market(tr,te,pos)
            for name,model in candidates():
                for fs,cols in [('base',features),('extended',extended)]:
                    model.fit(tr[cols],tr.y);out[name+'_'+fs]=model.predict(te[cols])
            for name in ['ridge100','cat4','extra40']:
                model=dict(candidates())[name];rtr=tr[tr.season>=year-3]
                model.fit(rtr[extended],rtr.y);out[name+'_recent']=model.predict(te[extended])
            out.to_csv(OUT/f'{pos}_{year}_predictions.csv',index=False)
            predcols=[c for c in out if c not in ['season','week','game_id','team','y','position','player_id']]
            for name in predcols:allmetrics.append(dict(position=pos,year=year,model=name,**metrics(out,out[name])))
            allpred.append(out)
            pd.DataFrame(allmetrics).to_csv(OUT/'yearly_metrics.csv',index=False)
            print(pos,year,'baseline',metrics(out,out.v42),flush=True)
    pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('*_20??_predictions.csv'))]).to_csv(OUT/'all_predictions.csv',index=False)
    summarize()

def linear_market(tr,te,pos):
    cols=['implied_opp_total','team_spread'] if pos=='DST' else ['implied_team_total','total_line','roof_dome']
    m=make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=100)).fit(tr[cols],tr.y)
    return m.predict(te[cols])

def summarize():
    d=pd.read_csv(OUT/'all_predictions.csv');rows=[]
    exclude=['season','week','game_id','team','y','position','player_id']
    for pos,p in d.groupby('position'):
        for period,sub in [('development_2021_2023',p[p.season<=2023]),('confirmation_2024_2025',p[p.season>=2024]),('all',p)]:
            if sub.empty:continue
            for c in p.columns:
                if c not in exclude and sub[c].notna().all():rows.append(dict(position=pos,period=period,model=c,**metrics(sub,sub[c])))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    print(pd.DataFrame(rows).query("period=='development_2021_2023'").sort_values(['position','rmse']).groupby('position').head(8).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--years',nargs='+',type=int,default=[2021,2022,2023])
    run(parser.parse_args().years)
