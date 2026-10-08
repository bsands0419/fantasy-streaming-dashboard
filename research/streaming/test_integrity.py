"""Meaningful leakage, scoring, weather, and runtime-parity regression tests."""
from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from live_inputs import kickoff_utc,apply_weather,restore_adjusted_rates,restore_big_plays,install
from evaluate import load_data,augment,baseline_predict,CACHE,joblib,GenericDSTScoring,ChampionshipDSTModelV42

class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.t,cls.k=load_data()

    def test_current_and_future_outcomes_cannot_change_prior_new_features(self):
        t,k=augment(self.t,self.k)
        changed=self.t.copy();changed_k=self.k.copy();mask=changed.season.ge(2024);kmask=changed_k.season.ge(2024)
        for c in ['def_sacks','def_tds','fg_att','off_epa_play','plays_off']:changed.loc[mask,c]=999
        changed_k.loc[kmask,'k_bucket_points']=999
        tt,kk=augment(changed,changed_k)
        # Includes week 1 of the first altered season, whose own outcomes must be excluded.
        for a,b in [(t,tt),(k,kk)]:
            rows=a.season.lt(2024)|(a.season.eq(2024)&a.week.eq(1));cols=[c for c in a if c.startswith(('new_','opp_new_'))]
            pd.testing.assert_frame_equal(a.loc[rows,cols],b.loc[rows,cols])

    def test_adjusted_rate_reconstruction_matches_frozen_definitions(self):
        expected=self.t[self.t.season.eq(2024)].set_index(['game_id','team'])
        repaired=restore_adjusted_rates(self.t,2024);repaired=repaired[repaired.season.eq(2024)].set_index(['game_id','team']).reindex(expected.index)
        cols=['adj_def_sack_game','adj_def_int_game','adj_def_points_game','adj_off_sack_game','adj_off_int_game','adj_off_points_game']
        np.testing.assert_allclose(repaired[cols],expected[cols],atol=2e-7,equal_nan=True)

    def test_adjusted_rates_preserve_frozen_seasons(self):
        repaired=restore_adjusted_rates(self.t,2024).set_index(['game_id','team'])
        before=self.t[self.t.season.ne(2024)].set_index(['game_id','team'])
        cols=[c for c in before if c.startswith('adj_')]
        np.testing.assert_array_equal(repaired.loc[before.index,cols],before[cols])

    def test_kickoff_timezone_domestic_and_international(self):
        self.assertEqual(str(kickoff_utc({'gameday':'2026-10-18','gametime':'09:30'})),'2026-10-18 13:30:00+00:00')
        self.assertEqual(str(kickoff_utc({'gameday':'2026-09-10','gametime':'20:20'})),'2026-09-11 00:20:00+00:00')
        self.assertEqual(str(kickoff_utc({'gameday':'2026-12-13','gametime':'16:25'})),'2026-12-13 21:25:00+00:00')
        with self.assertRaises(ValueError):kickoff_utc({'gameday':'2026-10-18','gametime':None})

    def test_weather_uses_forecast_nearest_actual_kickoff(self):
        from types import SimpleNamespace
        day=(pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=1)).date().isoformat()
        row={'gameday':day,'gametime':'20:20','season':2026,'week':1,'game_type':'REG','roof':'outdoors','game_id':'test'}
        stamp=kickoff_utc(row).floor('h');hours=pd.date_range(stamp-pd.Timedelta(hours=4),periods=9,freq='h')
        def get(url,params,timeout):
            self.assertEqual(params['timezone'],'UTC')
            return SimpleNamespace(json=lambda:{'hourly':{'time':[t.strftime('%Y-%m-%dT%H:%M') for t in hours],
                'temperature_2m':list(range(9)),'wind_speed_10m':[10]*9}})
        runtime=SimpleNamespace(OPEN_METEO='test',get=get,weather_location=lambda row:(40,-74))
        out,meta=apply_weather(runtime,pd.DataFrame([row]),2026,1,[])
        self.assertEqual(out.temp.iloc[0],4)
        self.assertEqual(meta['games_updated'],1)

    def test_scoring_target_agrees_with_components(self):
        sim={c:self.t[c].to_numpy(float) for c in ['def_sacks','def_interceptions','fumble_recovery_opp','def_fumbles_forced','def_qb_hits','def_tackles_for_loss','def_pass_defended','def_safeties','blocks','def_tds','special_teams_tds','points_allowed_sleeper','yards_allowed']}
        np.testing.assert_allclose(GenericDSTScoring().score(sim),self.t.dst_fantasy,atol=1e-6)

    def test_deterministic_baseline_agrees_with_large_simulation(self):
        path=CACHE/'baseline_DST_2021.joblib'
        model=joblib.load(path) if path.exists() else ChampionshipDSTModelV42().fit(self.t[self.t.season<2021])
        x=self.t[self.t.season.eq(2021)].head(2)
        exact=baseline_predict(model,x,'DST');monte=model.project(x,GenericDSTScoring(),n_sims=100000,seed=990).proj_mean
        np.testing.assert_allclose(exact,monte,atol=.07)

    def test_explosive_play_rates_and_defensive_mirroring(self):
        if not (CACHE/'pbp_2024.parquet').exists():
            self.skipTest('Run pbp_features.py to download the pinned-source historical PBP audit input')
        p=pd.read_parquet(CACHE/'pbp_2024.parquet',columns=['game_id','posteam','season_type','qb_dropback','rush_attempt','yards_gained'])
        repaired=restore_big_plays(self.t,p,2024).set_index(['game_id','team'])
        old=self.t[self.t.season.eq(2024)].set_index(['game_id','team']);new=repaired.loc[old.index]
        # One 20+ play differs in the refreshed upstream file; all 40+ counts agree.
        self.assertLessEqual((abs(new.off_big20_rate-old.off_big20_rate)>1e-6).sum(),1)
        np.testing.assert_allclose(new.off_big40_rate,old.off_big40_rate,atol=1e-6)
        for idx,row in new.iterrows():
            other=repaired.loc[(idx[0],row.opponent_team)]
            self.assertAlmostEqual(row.def_allowed_big20_rate,other.off_big20_rate)

    def test_midweek_kicker_refresh_excludes_same_week_results(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import championship_feature_factory as factory
        seen=[]
        def capture(kh,*args):seen.extend(kh.gameday.tolist());return kh
        runtime=SimpleNamespace(append_current_history=lambda *args:None)
        sched=pd.DataFrame({'season':[2024],'week':[5],'game_type':['REG'],'gameday':['2024-10-03']})
        kh=pd.DataFrame({'gameday':['2024-09-29','2024-10-03','2024-10-06']})
        with patch.object(factory,'build_future_kicker_features',capture):
            install(runtime)
            factory.build_future_kicker_features(kh,None,sched,None,2024,5)
        self.assertEqual(seen,['2024-09-29'])

if __name__=='__main__':unittest.main(verbosity=2)
