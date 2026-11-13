"""Download official nflverse play-by-play and aggregate prior-game opportunities."""
from pathlib import Path
import urllib.request, concurrent.futures, hashlib, json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'
COLS=['game_id','season','week','season_type','posteam','defteam','play_id','fixed_drive','yardline_100','down','ydstogo','play_type',
      'game_seconds_remaining','score_differential','pass_attempt','rush_attempt','qb_dropback','qb_kneel','qb_spike','sack','qb_hit',
      'interception','fumble_lost','field_goal_attempt','field_goal_result','kick_distance','touchdown','no_huddle','shotgun','epa']

def fetch(year):
    url=f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet'
    p=CACHE/f'pbp_{year}.parquet'
    if not p.exists():
        with urllib.request.urlopen(url,timeout=120) as r:p.write_bytes(r.read())
    z=pd.read_parquet(p,columns=COLS);z=z[z.season_type.eq('REG') & z.posteam.notna() & z.defteam.notna()].copy()
    z=z.sort_values(['game_id','play_id'])
    for c in ['posteam','defteam']:z[c]=z[c].replace({'LAR':'LA','OAK':'LV','SD':'LAC'})
    z['scrimmage']=z.play_type.isin(['pass','run']) & ~z.qb_kneel.eq(1)&~z.qb_spike.eq(1)
    z['neutral']=z.scrimmage & z.score_differential.abs().le(8) & z.game_seconds_remaining.gt(120)
    delta=-z.groupby(['game_id','posteam','fixed_drive']).game_seconds_remaining.diff()
    z['neutral_seconds']=delta.where(z.neutral & delta.between(1,60))
    rows=[]
    for (game,team),g in z.groupby(['game_id','posteam']):
        plays=g[g.scrimmage];neutral=g[g.neutral];db=g[g.qb_dropback.eq(1)]
        drives=g[g.scrimmage | g.field_goal_attempt.eq(1)].groupby('fixed_drive').agg(min_yardline=('yardline_100','min'),fg=('field_goal_attempt','max'),td=('touchdown','max'))
        reachable=drives[drives.min_yardline.le(35)];rz=drives[drives.min_yardline.le(20)]
        fourth=g[g.down.eq(4) & g.yardline_100.between(15,40)]
        rows.append(dict(game_id=game,team=team,season=int(g.season.iloc[0]),week=int(g.week.iloc[0]),
          drives=len(drives),range_drives=len(reachable),redzone_drives=len(rz),
          range_stall_rate=(1-reachable.td.mean()) if len(reachable) else np.nan,
          redzone_td_rate=rz.td.mean(),fg_per_range=reachable.fg.mean(),
          fourth_go_rate=fourth.scrimmage.mean(),neutral_pace=neutral.neutral_seconds.mean(),
          neutral_pass_rate=neutral.qb_dropback.mean(),no_huddle_rate=plays.no_huddle.mean(),
          pressure_rate=db.qb_hit.mean(),sacks_per_pressure=g.sack.sum()/max(g.qb_hit.sum(),1),
          deep_fg_att=((g.field_goal_attempt==1)&g.kick_distance.ge(50)).sum(),
          fg_att=g.field_goal_attempt.sum(),fg_made=g.field_goal_result.eq('made').sum(),
          dropbacks=g.qb_dropback.sum(),rushes=g.rush_attempt.sum(),interceptions=g.interception.sum()))
    out=pd.DataFrame(rows)
    # Defense perspective must come from the opposing offensive plays in the same game.
    pairs=z[['game_id','posteam','defteam']].drop_duplicates().rename(columns={'posteam':'team','defteam':'opponent_team'})
    out=out.merge(pairs,on=['game_id','team'],validate='one_to_one')
    print(year,len(out),flush=True)
    return out,dict(season=year,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def build():
    CACHE.mkdir(exist_ok=True,parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,range(2018,2026)))
    raw=pd.concat([x[0] for x in results],ignore_index=True).sort_values(['team','season','week'])
    cols=[c for c in raw if c not in ['game_id','team','season','week','opponent_team']]
    opp=raw[['game_id','team']+cols].rename(columns={'team':'opponent_team',**{c:'allowed_'+c for c in cols}})
    raw=raw.merge(opp,on=['game_id','opponent_team'],validate='one_to_one').sort_values(['team','season','week'])
    state=raw[['game_id','team']].copy()
    for c in cols+['allowed_'+c for c in cols]:
        for span in [8,16]:state['pbp_'+c+f'_ewm{span}']=raw.groupby('team')[c].transform(lambda s:s.shift().ewm(span=span,adjust=False).mean())
    opponent=state.rename(columns={'team':'opponent_team',**{c:'opp_'+c for c in state if c.startswith('pbp_')}})
    state=state.merge(raw[['game_id','team','opponent_team']],on=['game_id','team'],validate='one_to_one').merge(opponent,on=['game_id','opponent_team'],validate='one_to_one').drop(columns='opponent_team')
    raw.to_parquet(CACHE/'pbp_game_raw.parquet',index=False)
    state.to_parquet(CACHE/'pbp_features.parquet',index=False)
    (ROOT/'results/pbp_sources.json').write_text(json.dumps([x[1] for x in results],indent=2)+'\n')
    print('saved',state.shape,flush=True)

if __name__=='__main__':build()
