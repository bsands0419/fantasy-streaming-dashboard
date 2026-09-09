"""Repair live input parity without modifying the frozen v4.2 model artifact."""
from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from io import BytesIO
import numpy as np
import pandas as pd

def kickoff_utc(row):
    """nflverse gametime is Eastern time, including international fixtures."""
    day=pd.Timestamp(row['gameday']).date()
    clock=str(row.get('gametime',''))
    if not clock or clock in {'nan','None','NaT'}:raise ValueError('Kickoff time unavailable')
    return pd.Timestamp(f'{day.isoformat()} {clock}',tz=ZoneInfo('America/New_York')).tz_convert('UTC')

def apply_weather(runtime,schedule,season,week,warnings):
    schedule=schedule.copy();count=0
    target=schedule[schedule.season.eq(season)&schedule.week.eq(week)&schedule.game_type.eq('REG')]
    now=pd.Timestamp.now(tz='UTC')
    for idx,row in target.iterrows():
        if str(row.get('roof','')).lower() in {'dome','closed'}:continue
        try:stamp=kickoff_utc(row)
        except (ValueError,TypeError):
            warnings.append(f"Weather skipped for {row.get('game_id')}: missing kickoff time.");continue
        if not 0 <= (stamp-now).total_seconds() <= 16*86400:continue
        location=runtime.weather_location(row)
        if not location:continue
        try:
            data=runtime.get(runtime.OPEN_METEO,params={'latitude':location[0],'longitude':location[1],
              'hourly':'temperature_2m,wind_speed_10m','temperature_unit':'fahrenheit','wind_speed_unit':'mph',
              'timezone':'UTC','forecast_days':16},timeout=15).json()['hourly']
            times=pd.to_datetime(data['time'],utc=True)
            j=int(np.argmin(abs(times-stamp)))
            if abs((times[j]-stamp).total_seconds())>3600:raise ValueError('Forecast does not cover kickoff')
            temp=float(data['temperature_2m'][j]);wind=float(data['wind_speed_10m'][j])
            if not np.isfinite([temp,wind]).all():raise ValueError('Missing weather value')
            schedule.loc[idx,['temp','wind']]=[temp,wind];count+=1
        except Exception as exc:warnings.append(f"Weather unavailable for {row.get('game_id')} ({type(exc).__name__}).")
    return schedule,{'source':'open-meteo','games_updated':count,'time_basis':'kickoff_utc'}

def restore_adjusted_rates(history,season):
    """Recreate the exact six opponent-adjusted historical rate definitions."""
    h=history.copy().sort_values(['team','gameday','game_id'])
    raw=['off_sack_rate_allowed','off_int_rate','off_points_per_play','def_sack_rate','def_int_rate','def_points_allowed_per_play']
    lag=h[['game_id','team']].copy()
    for c in raw:lag[c]=h.groupby('team')[c].transform(lambda s:s.shift().ewm(span=10,adjust=False,min_periods=1).mean())
    lag=lag.rename(columns={'team':'opponent_team',**{c:'_prior_opp_'+c for c in raw}})
    h=h.merge(lag,on=['game_id','opponent_team'],how='left',validate='many_to_one')
    mask=h.season.eq(season)
    for target,own,opp in [
        ('adj_def_sack_game','def_sack_rate','off_sack_rate_allowed'),
        ('adj_def_int_game','def_int_rate','off_int_rate'),
        ('adj_def_points_game','def_points_allowed_per_play','off_points_per_play'),
        ('adj_off_sack_game','off_sack_rate_allowed','def_sack_rate'),
        ('adj_off_int_game','off_int_rate','def_int_rate'),
        ('adj_off_points_game','off_points_per_play','def_points_allowed_per_play')]:
        h.loc[mask,target]=h.loc[mask,own]-h.loc[mask,'_prior_opp_'+opp]
    return h.drop(columns=['_prior_opp_'+c for c in raw])

def restore_big_plays(history,pbp,season):
    h=history.copy()
    p=pbp[pbp['season_type'].eq('REG') & (pbp.qb_dropback.eq(1)|pbp.rush_attempt.eq(1))].copy()
    p['team']=p.posteam.replace({'LAR':'LA','OAK':'LV','SD':'LAC'})
    counts=p.groupby(['game_id','team']).yards_gained.agg(_big20=lambda s:s.ge(20).sum(),_big40=lambda s:s.ge(40).sum()).reset_index()
    h=h.merge(counts,on=['game_id','team'],how='left',validate='many_to_one')
    current=h.season.eq(season)
    for n in [20,40]:h.loc[current,f'off_big{n}_rate']=h.loc[current,f'_big{n}']/h.loc[current,'plays_off'].replace(0,np.nan)
    mirror=h[['game_id','team','off_big20_rate','off_big40_rate']].rename(columns={'team':'opponent_team','off_big20_rate':'_oppbig20','off_big40_rate':'_oppbig40'})
    h=h.merge(mirror,on=['game_id','opponent_team'],how='left',validate='many_to_one');current=h.season.eq(season)
    for n in [20,40]:h.loc[current,f'def_allowed_big{n}_rate']=h.loc[current,f'_oppbig{n}']
    return h.drop(columns=['_big20','_big40','_oppbig20','_oppbig40'])

def install(runtime):
    # The base kicker builder previously read the entire supplied player history.
    # Freeze every kicker field at the beginning of the requested week, including
    # when a refresh occurs after Thursday's game has already been played.
    import championship_feature_factory as factory
    original_kicker_builder=factory.build_future_kicker_features
    def cutoff_kicker_builder(kicker_history,team_history,schedule,current_kickers,season,week):
        games=schedule[schedule.season.eq(season)&schedule.week.eq(week)&schedule.game_type.eq('REG')]
        if len(games):
            cutoff=pd.to_datetime(games.gameday).min()
            kicker_history=kicker_history[pd.to_datetime(kicker_history.gameday)<cutoff].copy()
        return original_kicker_builder(kicker_history,team_history,schedule,current_kickers,season,week)
    factory.build_future_kicker_features=cutoff_kicker_builder
    original_append=runtime.append_current_history
    runtime.apply_weather=lambda s,y,w,warnings:apply_weather(runtime,s,y,w,warnings)
    def append(team,kicker,schedule,season,warnings):
        th,kh,meta=original_append(team,kicker,schedule,season,warnings)
        if th.season.eq(season).any():
            th=restore_adjusted_rates(th,season)
            meta['opponent_adjusted_rates']='updated_from_prior_games'
            try:
                url=f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'
                content=runtime.get(url,timeout=60).content
                pbp=pd.read_parquet(BytesIO(content),columns=['game_id','posteam','season_type','qb_dropback','rush_attempt','yards_gained'])
                th=restore_big_plays(th,pbp,season)
                missing=int(th.loc[th.season.eq(season),'off_big20_rate'].isna().sum())
                meta['big_play_features']={'source':'nflverse_pbp','missing_team_games':missing}
                if missing:warnings.append(f'Play-by-play is missing for {missing} completed team games; explosive-play inputs are incomplete.')
            except Exception as exc:
                meta['big_play_features']={'source':'unavailable'}
                warnings.append(f'Explosive-play refresh failed ({type(exc).__name__}); those inputs may still reflect older games.')
        return th,kh,meta
    runtime.append_current_history=append
