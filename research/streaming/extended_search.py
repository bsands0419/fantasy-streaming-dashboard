"""Second development stage: play-by-play features and event shrinkage."""
from evaluate import *
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def enrich():
    t,k=load_data();t,k=augment(t,k)
    pbp=pd.read_parquet(CACHE/'pbp_features.parquet');pbp['team']=pbp.team.replace({'LAR':'LA'})
    result=[]
    for pos,d in [('DST',t),('K',k)]:
        d=d.merge(pbp,on=['game_id','team'],how='left',validate='many_to_one')
        if d.loc[d.season>=2021,'pbp_drives_ewm16'].isna().any():raise ValueError('Missing PBP match')
        d['y']=d.dst_fantasy if pos=='DST' else d.k_bucket_points
        result.append(d)
    return result

def pbp_columns(pos):
    if pos=='K':
        own=['drives','range_drives','redzone_drives','range_stall_rate','redzone_td_rate','fg_per_range','fourth_go_rate','neutral_pace','neutral_pass_rate','no_huddle_rate','deep_fg_att','fg_att']
        opp=['allowed_drives','allowed_range_drives','allowed_redzone_drives','allowed_range_stall_rate','allowed_redzone_td_rate','allowed_fg_per_range','allowed_fg_att','allowed_deep_fg_att']
    else:
        own=['allowed_pressure_rate','allowed_sacks_per_pressure','allowed_dropbacks','allowed_interceptions','allowed_range_drives','allowed_redzone_td_rate','neutral_pace']
        opp=['pressure_rate','sacks_per_pressure','dropbacks','neutral_pass_rate','neutral_pace','no_huddle_rate','fourth_go_rate','interceptions','range_drives','redzone_td_rate']
    return [f'pbp_{c}_ewm{s}' for c in own for s in [8,16]]+[f'opp_pbp_{c}_ewm{s}' for c in opp for s in [8,16]]

def linear(m):return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),m)

def stage_candidates():
    for a in [100,1000]:yield f'pbp_ridge{a}',linear(Ridge(alpha=a))
    yield 'pbp_bayes',linear(BayesianRidge())
    for depth in [2,4]:
        yield f'pbp_cat{depth}',make_pipeline(SimpleImputer(strategy='median'),CatBoostRegressor(iterations=400,depth=depth,learning_rate=.035,l2_leaf_reg=20,loss_function='RMSE',thread_count=2,verbose=False,allow_writing_files=False,random_seed=42))
    yield 'pbp_extra',make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=300,min_samples_leaf=30,max_features=.7,n_jobs=2,random_state=42))
    yield 'pbp_xgb',XGBRegressor(n_estimators=300,max_depth=2,learning_rate=.025,min_child_weight=40,reg_lambda=30,subsample=.85,colsample_bytree=.8,n_jobs=2,random_state=42)
    yield 'pbp_lgbm',LGBMRegressor(n_estimators=250,num_leaves=7,learning_rate=.025,min_child_samples=70,reg_lambda=30,verbosity=-1,n_jobs=2,random_state=42)

def event_predict(tr,te,cols,pos):
    if pos=='DST':
        mapping={'def_sacks':1,'def_interceptions':2,'fumble_recovery_opp':2}
        pred=np.zeros(len(te))
        for c,w in mapping.items():
            m=linear(PoissonRegressor(alpha=1,max_iter=300)).fit(tr[cols],tr[c]);pred+=w*m.predict(te[cols])
        for c,w in {'def_tds':6,'special_teams_tds':6,'blocks':2,'def_safeties':2}.items():pred+=w*tr[c].mean()
        def band(x):return np.select([x<=0,x<=6,x<=13,x<=20,x<=27,x<=34],[10,7,4,1,0,-1],default=-4)
        m=linear(Ridge(alpha=1000)).fit(tr[cols],band(tr.points_allowed_sleeper));pred+=m.predict(te[cols])
    else:
        mapping={'fg_made_0_19':3,'fg_made_20_29':3,'fg_made_30_39':3,'fg_made_40_49':4,'fg_made_50_59':5,'fg_made_60_':5,'pat_made':1}
        pred=np.zeros(len(te))
        for c,w in mapping.items():
            m=linear(PoissonRegressor(alpha=1,max_iter=300)).fit(tr[cols],tr[c]);pred+=w*m.predict(te[cols])
    return pred

def run_extended(years):
    t,k=enrich()
    for pos,d,features in [('DST',t,DST_V4_FEATURES),('K',k,K_V42_FEATURES)]:
        cols=list(features)+pbp_columns(pos)
        for year in years:
            tr=d[d.season<year];te=d[d.season==year]
            path=OUT/f'{pos}_{year}_predictions.csv';out=pd.read_csv(path)
            assert list(out.game_id)==list(te.game_id)
            base=joblib.load(CACHE/f'baseline_{pos}_{year}.joblib')
            if pos=='DST':
                for fraction in [.5,1.0]:
                    delta=sum(w*(tr[c].mean()-base.models[c].predict(te)) for c,w in {'def_tds':6,'special_teams_tds':6,'blocks':2,'def_safeties':2}.items())
                    out[f'rare_shrink{fraction}']=out.v42+fraction*base.core_weight*delta
                out['rounded_pa']=out.v42+base.core_weight*(pa_expectation(base.pa_model,te,True)-pa_expectation(base.pa_model,te))
            for name,model in stage_candidates():
                model.fit(tr[cols],tr.y);out[name]=model.predict(te[cols])
            # This alternative avoids Gaussian regressions for nonnegative event counts.
            out['pbp_poisson_events']=event_predict(tr,te,cols,pos)
            out.to_csv(path,index=False)
            print(pos,year,'extended complete',flush=True)
    pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('*_20??_predictions.csv'))]).to_csv(OUT/'all_predictions.csv',index=False)
    summarize()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--years',nargs='+',type=int,default=[2021,2022,2023]);run_extended(parser.parse_args().years)
