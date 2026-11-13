"""Paired audit of stale versus repaired live D/ST inputs, with fixed model weights."""
from evaluate import *
from championship_feature_factory_v4_2 import build_v42_team_features
sys.path.insert(0,str(ROOT/'backend'))
from live_inputs import restore_adjusted_rates

def replay():
    t,k=load_data();t['gameday']=pd.to_datetime(t.gameday)
    s=pd.read_csv(CACHE/'games.csv',low_memory=False);s['gameday']=pd.to_datetime(s.gameday)
    for c in ['home_team','away_team']:s[c]=s[c].replace({'OAK':'LV','LAR':'LA','SD':'LAC'})
    stale_raw=['off_big20_rate','off_big40_rate','def_allowed_big20_rate','def_allowed_big40_rate','adj_def_sack_game','adj_def_int_game','adj_def_points_game','adj_off_sack_game','adj_off_int_game','adj_off_points_game']
    rows=[]
    for year in [2024,2025]:
        model=joblib.load(CACHE/f'baseline_DST_{year}.joblib')
        for week in sorted(t[t.season.eq(year)].week.unique()):
            games=s[s.season.eq(year)&s.week.eq(week)&s.game_type.eq('REG')];cutoff=games.gameday.min()
            good=t[t.gameday<cutoff].copy();bad=good.copy();bad.loc[bad.season.eq(year),stale_raw]=np.nan
            a=build_v42_team_features(bad,s,year,int(week));b=build_v42_team_features(good,s,year,int(week))
            assert list(a.team)==list(b.team)
            r=b[['season','week','game_id','team']].copy();r['stale']=baseline_predict(model,a,'DST');r['restored']=baseline_predict(model,b,'DST')
            r=r.merge(t[['game_id','team','dst_fantasy']].rename(columns={'dst_fantasy':'y'}),on=['game_id','team'],validate='one_to_one')
            rows.append(r);print(year,week,flush=True)
    p=pd.concat(rows);p.to_csv(OUT/'live_input_replay_predictions.csv',index=False)
    summary=[]
    for year in [2024,2025,'all']:
        x=p if year=='all' else p[p.season.eq(year)]
        for name in ['stale','restored']:summary.append({'period':str(year),'inputs':name,**metrics(x,x[name])})
    pd.DataFrame(summary).to_csv(OUT/'live_input_replay_metrics.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':replay()
