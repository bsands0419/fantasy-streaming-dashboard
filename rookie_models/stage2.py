from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import modeling as b

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_v2'
MODELS = ROOT / 'trained_v2'
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

POSITIONS = ['QB', 'RB', 'WR', 'TE']
PREDICT_YEAR = 2026
TRAIN_END = 2023
DEV_YEARS = list(range(2013, 2019))
VALID_YEARS = list(range(2019, 2023))
FINAL_TEST_YEAR = 2023
HIT_RANK = {'QB': 12, 'RB': 12, 'WR': 24, 'TE': 12}
STAR_RANK = {'QB': 6, 'RB': 6, 'WR': 12, 'TE': 6}
SEED = 20260816


def nser(df, c, default=np.nan):
    if c in df.columns:
        return pd.to_numeric(df[c], errors='coerce')
    return pd.Series(default, index=df.index, dtype=float)


def sched_games(season: int) -> int:
    return 16 if int(season) <= 2020 else 17


def add_team_context(d: pd.DataFrame, team: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if d.empty or team.empty or 'team_id' not in d.columns or 'team_id' not in team.columns:
        return d
    d['_team_key'] = d['team_id'].astype(str)
    t = team.copy()
    t['_team_key'] = t['team_id'].astype(str)
    wanted = [
        'season', '_team_key', 'adj_off_epa', 'def_strength_faced', 'net_adj_epa',
        'EPAplay_off', 'EPAplay_off_pass', 'EPAplay_off_rush',
        'success_off', 'success_off_pass', 'success_off_rush',
        'explosive_off', 'explosive_off_pass', 'explosive_off_rush',
        'playsgame_off', 'passrate_off', 'rushrate_off', 'yardsplay_off',
        'yardsplay_off_pass', 'yardsplay_off_rush'
    ]
    keep = [c for c in wanted if c in t.columns]
    if len(keep) <= 2:
        return d.drop(columns=['_team_key'], errors='ignore')
    t = t[keep].drop_duplicates(['season', '_team_key'])
    ren = {c: f'team_{c}' for c in keep if c not in ['season', '_team_key']}
    t = t.rename(columns=ren)
    d = d.merge(t, on=['season', '_team_key'], how='left')
    return d.drop(columns=['_team_key'], errors='ignore')


def prep_college(pa, ru, re, team):
    pa, ru, re = b.prep_cfb(pa, ru, re)
    pa = add_team_context(pa, team)
    ru = add_team_context(ru, team)
    re = add_team_context(re, team)

    for d in (pa, ru, re):
        if 'fbs_class' in d.columns:
            d['power_conf'] = d['fbs_class'].astype(str).str.upper().isin(['P4', 'P5']).astype(float)
        else:
            d['power_conf'] = np.nan

    pa['sack_rate'] = nser(pa, 'sacked') / nser(pa, 'dropbacks').replace(0, np.nan)
    pa['int_rate'] = nser(pa, 'pass_int') / nser(pa, 'att').replace(0, np.nan)
    pa['pass_td_rate'] = nser(pa, 'passing_td') / nser(pa, 'att').replace(0, np.nan)
    pa['comp_per_game'] = nser(pa, 'comp') / nser(pa, 'games').replace(0, np.nan)

    ru['rush_td_per_game'] = nser(ru, 'rushing_td') / nser(ru, 'games').replace(0, np.nan)
    ru['rush_td_rate'] = nser(ru, 'rushing_td') / nser(ru, 'plays').replace(0, np.nan)

    re['rec_td_per_game'] = nser(re, 'passing_td') / nser(re, 'games').replace(0, np.nan)
    re['rec_per_game'] = nser(re, 'comp') / nser(re, 'games').replace(0, np.nan)
    re['targets_per_game'] = nser(re, 'targets') / nser(re, 'games').replace(0, np.nan)
    re['reception_share'] = nser(re, 'comp') / nser(re, 'team_receptions').replace(0, np.nan)
    re['receptions_per_team_pass_att'] = nser(re, 'comp') / nser(re, 'team_pass_att').replace(0, np.nan)

    return pa, ru, re


def combine_player_season(d, namecol, volume_cols):
    return b.combine_name_season(d, namecol, set(volume_cols))


def career_features(g, prefix, metrics, draft_year):
    if g is None or g.empty:
        return {}
    g = g[(g['season'] < draft_year) & (g['season'] >= draft_year - 6)].sort_values('season').copy()
    if g.empty:
        return {}
    out = {
        f'{prefix}_seasons': float(g['season'].nunique()),
        f'{prefix}_years_before_first': float(draft_year - g['season'].min()),
        f'{prefix}_years_before_last': float(draft_year - g['season'].max()),
    }
    for c in metrics:
        if c not in g.columns:
            continue
        v = pd.to_numeric(g[c], errors='coerce')
        if not v.notna().any():
            continue
        vv = v.dropna()
        out[f'{prefix}_{c}_final'] = float(vv.iloc[-1])
        out[f'{prefix}_{c}_peak'] = float(vv.max())
        out[f'{prefix}_{c}_mean'] = float(vv.mean())
        if len(vv) >= 2:
            x = np.arange(len(g))[v.notna().to_numpy()]
            out[f'{prefix}_{c}_slope'] = float(np.polyfit(x, vv.to_numpy(), 1)[0])
            early = g.loc[v.notna() & (g['season'] < g['season'].max()), c]
            if len(early):
                out[f'{prefix}_{c}_early_peak'] = float(pd.to_numeric(early, errors='coerce').max())
        # Experience-adjusted peak rewards production farther from draft year without hard-coded breakout labels.
        years_before = (draft_year - g.loc[v.notna(), 'season']).astype(float)
        exp_adj = vv.to_numpy() * (1.0 + 0.10 * np.maximum(years_before.to_numpy() - 1.0, 0.0))
        out[f'{prefix}_{c}_expadj_peak'] = float(np.nanmax(exp_adj))
    return out


def lookup_group(nn, mp, candidates, draft_year):
    if nn in mp:
        g = mp[nn]
        if not g[(g.season < draft_year) & (g.season >= draft_year - 6)].empty:
            return g, 'exact'
    key = b.fuzzy_lookup(nn, candidates)
    if key and key in mp:
        g = mp[key]
        if not g[(g.season < draft_year) & (g.season >= draft_year - 6)].empty:
            return g, 'fuzzy'
    return pd.DataFrame(), 'none'


def build_profiles(draft, pa, ru, re):
    team_context = [c for c in pa.columns if c.startswith('team_')]
    pvol = ['plays','games','team_games','TEPA','yards','comp','att','passing_td','sacked','sack_yds','pass_int','dropbacks']
    rvol = ['plays','games','team_games','TEPA','yards','rushing_td','fumbles','team_rushes','team_rush_yards','team_rush_td','team_pass_att','team_dropbacks']
    cvol = ['plays','games','team_games','TEPA','yards','comp','targets','passing_td','fumbles','team_targets','team_rec_yards','team_rec_td','team_receptions','team_pass_att','team_dropbacks']
    pa2 = combine_player_season(pa, 'passer_player_name', pvol)
    ru2 = combine_player_season(ru, 'rusher_player_name', rvol)
    re2 = combine_player_season(re, 'receiver_player_name', cvol)
    pmap = {k:g for k,g in pa2.groupby('name_norm')}
    rmap = {k:g for k,g in ru2.groupby('name_norm')}
    cmap = {k:g for k,g in re2.groupby('name_norm')}
    pc, rc, cc = list(pmap), list(rmap), list(cmap)

    pm = ['playsgame','EPAplay','EPAgame','yardsplay','yardsgame','success','comppct','passing_td','sacked','pass_int','yardsdropback','dropbacks','sack_rate','int_rate','pass_td_rate','comp_per_game','power_conf'] + team_context
    rm = ['playsgame','EPAplay','EPAgame','yardsplay','yardsgame','success','rushing_td','fumbles','rush_share','rush_yard_share','rush_td_share','rush_td_per_game','rush_td_rate','power_conf'] + [c for c in ru2.columns if c.startswith('team_')]
    cm = ['playsgame','EPAplay','EPAgame','yardsplay','yardsgame','success','catchpct','passing_td','targets','target_share','rec_yard_share','rec_td_share','rec_per_team_pass_att','yards_per_target','td_per_target','rec_td_per_game','rec_per_game','targets_per_game','reception_share','receptions_per_team_pass_att','power_conf'] + [c for c in re2.columns if c.startswith('team_')]

    eligible = draft[draft['season'].between(b.TRAIN_DRAFT_START, PREDICT_YEAR) & draft['category'].isin(POSITIONS)].copy()
    eligible['name_norm'] = eligible['pfr_name'].map(b.norm_name)
    rows = []
    for _, d in eligible.iterrows():
        nn = d['name_norm']; pos = d['category']; year = int(d['season'])
        feat = {'season':year, 'pfr_id':d.get('pfr_id'), 'pfr_name':d['pfr_name'], 'position':pos,
                'draft_round':d.get('round'), 'draft_pick':d.get('pick')}
        match_types = []

        pg, pmatch = lookup_group(nn, pmap, pc, year)
        rg, rmatch = lookup_group(nn, rmap, rc, year)
        cg, cmatch = lookup_group(nn, cmap, cc, year)

        if pos == 'QB':
            if not pg.empty:
                feat.update(career_features(pg, 'pass', pm, year)); match_types.append(pmatch)
            if not rg.empty:
                feat.update(career_features(rg, 'rush', rm, year)); match_types.append(rmatch)
                q = rg[(rg.season < year) & (rg.season >= year - 6)].copy()
                if not q.empty:
                    q['qb_rush_fantasy_pg'] = nser(q, 'yardsgame').fillna(0) / 10 + (nser(q, 'rushing_td') / nser(q, 'games').replace(0,np.nan)).fillna(0) * 6
                    feat.update(career_features(q, 'qb', ['qb_rush_fantasy_pg','rush_share','yardsgame','EPAplay','success'], year))
        elif pos == 'RB':
            if not rg.empty:
                feat.update(career_features(rg, 'rush', rm, year)); match_types.append(rmatch)
            if not cg.empty:
                feat.update(career_features(cg, 'rec', cm, year)); match_types.append(cmatch)
            seasons = sorted(set(rg['season'].tolist() if not rg.empty else []) | set(cg['season'].tolist() if not cg.empty else []))
            combo = []
            for sy in seasons:
                if sy >= year or sy < year - 6:
                    continue
                rr = rg[rg.season == sy].iloc[-1] if not rg.empty and (rg.season == sy).any() else None
                cr = cg[cg.season == sy].iloc[-1] if not cg.empty and (cg.season == sy).any() else None
                rush_y = float(pd.to_numeric(pd.Series([rr.get('yards',0) if rr is not None else 0]),errors='coerce').fillna(0).iloc[0])
                rec_y = float(pd.to_numeric(pd.Series([cr.get('yards',0) if cr is not None else 0]),errors='coerce').fillna(0).iloc[0])
                rushes_team = float(pd.to_numeric(pd.Series([rr.get('team_rushes',np.nan) if rr is not None else np.nan]),errors='coerce').iloc[0])
                pass_team = np.nan
                if cr is not None: pass_team = pd.to_numeric(pd.Series([cr.get('team_pass_att',np.nan)]),errors='coerce').iloc[0]
                if not np.isfinite(pass_team) and rr is not None: pass_team = pd.to_numeric(pd.Series([rr.get('team_pass_att',np.nan)]),errors='coerce').iloc[0]
                team_plays = (rushes_team if np.isfinite(rushes_team) else 0) + (pass_team if np.isfinite(pass_team) else 0)
                games = np.nan
                if rr is not None: games = pd.to_numeric(pd.Series([rr.get('games',np.nan)]),errors='coerce').iloc[0]
                if not np.isfinite(games) and cr is not None: games = pd.to_numeric(pd.Series([cr.get('games',np.nan)]),errors='coerce').iloc[0]
                combo.append({'season':sy,
                              'scrim_yards':rush_y+rec_y,
                              'scrim_ypg':(rush_y+rec_y)/games if np.isfinite(games) and games>0 else np.nan,
                              'scrim_yards_per_team_play':(rush_y+rec_y)/team_plays if team_plays>0 else np.nan,
                              'rec_yards_per_team_pass':rec_y/pass_team if np.isfinite(pass_team) and pass_team>0 else np.nan})
            if combo:
                feat.update(career_features(pd.DataFrame(combo), 'rb', ['scrim_yards','scrim_ypg','scrim_yards_per_team_play','rec_yards_per_team_pass'], year))
        else:
            if not cg.empty:
                feat.update(career_features(cg, 'rec', cm, year)); match_types.append(cmatch)

        feat['college_match'] = float(bool(match_types))
        feat['fuzzy_match'] = float(any(x == 'fuzzy' for x in match_types))
        rows.append(feat)
    return pd.DataFrame(rows)


def parse_height_inches(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int,float,np.integer,np.floating)):
        v=float(x)
        if v > 55: return v
        if 5 <= v <= 8: return v*12
    s=str(x).strip().lower().replace('"','')
    m=re.match(r'^(\d+)\s*[-\']\s*(\d+)', s)
    if m: return float(m.group(1))*12+float(m.group(2))
    try:
        v=float(s)
        return v if v>55 else v*12
    except Exception:
        return np.nan


def add_nfl_meta(prof, players, combine):
    p = b.add_nfl_metadata(prof, players, combine)
    hsource = 'ht' if 'ht' in p.columns else ('height' if 'height' in p.columns else None)
    p['height_inches'] = p[hsource].map(parse_height_inches) if hsource else np.nan
    if 'wt' in p.columns:
        p['weight_lbs'] = pd.to_numeric(p['wt'], errors='coerce')
    elif 'weight' in p.columns:
        p['weight_lbs'] = pd.to_numeric(p['weight'], errors='coerce')
    else:
        p['weight_lbs'] = np.nan
    p['bmi'] = 703 * p['weight_lbs'] / (p['height_inches']**2)
    if 'forty' in p.columns:
        f=pd.to_numeric(p['forty'],errors='coerce')
        p['speed_score_v2'] = p['weight_lbs'] * (200 / f.clip(lower=4.0))**4
    if 'vertical' in p.columns and 'broad_jump' in p.columns:
        p['burst_proxy'] = pd.to_numeric(p['vertical'],errors='coerce') + pd.to_numeric(p['broad_jump'],errors='coerce')/12.0
    p['young_for_position'] = -pd.to_numeric(p.get('draft_age'), errors='coerce')
    p['nfl_meta_match'] = ((p.get('gsis_id', pd.Series(index=p.index,dtype=object)).notna()) |
                           (p.get('birth_date', pd.Series(index=p.index,dtype=object)).notna())).astype(float)
    return p


def build_targets(p, weekly):
    p=p.copy(); w=weekly.copy()
    w['season']=pd.to_numeric(w['season'],errors='coerce')
    if 'season_type' in w.columns:
        w=w[w['season_type'].astype(str).str.upper().eq('REG')].copy()
    idcol=b.first_existing(w,['player_id','gsis_id'])
    fpcol=b.first_existing(w,['fantasy_points_ppr','fantasy_points'])
    if not idcol or not fpcol: raise RuntimeError('NFL weekly id/fantasy fields missing')
    w['fp']=pd.to_numeric(w[fpcol],errors='coerce').fillna(0)
    if 'position' in w.columns:
        w['position_std']=w['position'].astype(str).str.upper().replace({'HB':'RB','FB':'RB'})
    else:
        w['position_std']=''

    # League-wide positional season ranks, not ranks only among our drafted cohort.
    ss=w[w.position_std.isin(POSITIONS)].groupby(['season','position_std',idcol],dropna=False).agg(total=('fp','sum')).reset_index()
    ss['ppg_teamgame']=ss.apply(lambda r: r['total']/sched_games(int(r['season'])),axis=1)
    ss['rank']=ss.groupby(['season','position_std'])['ppg_teamgame'].rank(ascending=False,method='min')
    rankmap={(int(r.season),r.position_std,r[idcol]):int(r['rank']) for _,r in ss.iterrows()}

    records=[]
    for _,r in p.iterrows():
        dy=int(r['season']); pos=r['position']; gid=r.get('gsis_id')
        valid_meta=bool(r.get('nfl_meta_match',0))
        vals=[]; active_vals=[]; totals=[]; ranks=[]
        for sy in range(dy,dy+3):
            z=w[(w[idcol]==gid)&(w.season==sy)] if pd.notna(gid) else w.iloc[0:0]
            total=float(z.fp.sum()) if not z.empty else 0.0
            vals.append(total/sched_games(sy)); totals.append(total)
            active_vals.append(total/len(z) if len(z) else 0.0)
            if pd.notna(gid):
                rk=rankmap.get((sy,pos,gid))
                if rk is not None: ranks.append(rk)
        top2=sorted(vals,reverse=True)[:2]
        records.append({
            'primary_ppg':float(np.mean(top2)) if top2 else 0.0,
            'peak3_ppg':float(max(vals)) if vals else 0.0,
            'avg3_ppg':float(sum(totals)/sum(sched_games(sy) for sy in range(dy,dy+3))),
            'rookie_ppg':float(vals[0]),
            'active_peak_ppg':float(max(active_vals)) if active_vals else 0.0,
            'total3_ppr':float(sum(totals)),
            'hit3':int(bool(ranks) and min(ranks)<=HIT_RANK[pos]),
            'star3':int(bool(ranks) and min(ranks)<=STAR_RANK[pos]),
            'target_valid':int(valid_meta),
            'best_rank3':float(min(ranks)) if ranks else np.nan,
        })
    return pd.concat([p.reset_index(drop=True),pd.DataFrame(records)],axis=1)


def feature_groups(df,pos):
    banned={'primary_ppg','peak3_ppg','avg3_ppg','rookie_ppg','active_peak_ppg','total3_ppr','hit3','star3','target_valid','best_rank3',
            'season','pfr_id','pfr_name','position','gsis_id','name_norm','birth_date','college_name','college_match','fuzzy_match','nfl_meta_match'}
    nums=[c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]
    capital=[c for c in nums if c in ['draft_round','draft_pick','log_pick','pick_inv_sqrt','day1','day2']]
    athletic=[c for c in nums if any(k in c.lower() for k in ['forty','vertical','bench','broad','cone','shuttle','speed_score','height_inches','weight_lbs','bmi','burst_proxy'])]
    age=[c for c in nums if c in ['draft_age','young_for_position'] or 'years_before' in c or c.endswith('_seasons')]
    context=[c for c in nums if 'team_' in c or c.endswith('power_conf')]
    production=[c for c in nums if c not in set(capital+athletic+age+context) and not c.startswith('draft_')]

    if pos=='QB': keys=['yardsdropback','EPAplay','success','comppct','sack_rate','int_rate','pass_td_rate','qb_rush_fantasy_pg','rush_yardsgame','rush_share']
    elif pos=='RB': keys=['scrim_yards_per_team_play','scrim_ypg','rec_yards_per_team_pass','rec_per_team_pass_att','target_share','rush_share','rush_yard_share','EPAplay','success']
    else: keys=['rec_per_team_pass_att','target_share','rec_yard_share','reception_share','EPAplay','success','yards_per_target','yardsgame','targets_per_game']
    coreprod=[c for c in production if any(k in c for k in keys)]
    core=list(dict.fromkeys(capital+coreprod+age+([c for c in athletic if pos in ['RB','TE']])))
    return {
        'capital':capital,
        'college_only':production+age+context,
        'capital_core':core,
        'capital_production':list(dict.fromkeys(capital+production+age)),
        'capital_prod_context':list(dict.fromkeys(capital+production+age+context)),
        'full':list(dict.fromkeys(capital+production+age+context+athletic)),
    }


def model_candidates():
    return {
      'ridge3':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=3.0))]),
      'ridge10':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=10.0))]),
      'ridge30':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=30.0))]),
      'elastic':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',ElasticNet(alpha=.04,l1_ratio=.15,max_iter=10000,random_state=SEED))]),
      'extra4':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',ExtraTreesRegressor(n_estimators=350,min_samples_leaf=4,max_features=.75,random_state=SEED,n_jobs=-1))]),
      'extra7':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',ExtraTreesRegressor(n_estimators=350,min_samples_leaf=7,max_features=1.0,random_state=SEED+1,n_jobs=-1))]),
      'rf':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',RandomForestRegressor(n_estimators=350,min_samples_leaf=5,max_features=.7,random_state=SEED,n_jobs=-1))]),
      'gbr':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',GradientBoostingRegressor(n_estimators=220,learning_rate=.025,max_depth=2,min_samples_leaf=6,loss='huber',random_state=SEED))]),
      'hist':Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',HistGradientBoostingRegressor(max_iter=260,learning_rate=.035,max_leaf_nodes=15,l2_regularization=4,min_samples_leaf=14,random_state=SEED))]),
    }


def reg_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);ok=np.isfinite(y)&np.isfinite(p);y=y[ok];p=p[ok]
    if len(y)<4:return {'n':len(y),'mae':np.nan,'rmse':np.nan,'r2':np.nan,'pearson':np.nan,'spearman':np.nan}
    return {'n':len(y),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'r2':r2_score(y,p),
            'pearson':pearsonr(y,p).statistic,'spearman':spearmanr(y,p).statistic}


def wf_predict(d,features,model,years,target='primary_ppg'):
    out=[]
    for y in years:
        tr=d[(d.season<y)&d[target].notna()];te=d[(d.season==y)&d[target].notna()]
        if len(tr)<35 or te.empty:continue
        m=clone(model);m.fit(tr[features],tr[target])
        q=te[['season','pfr_name',target,'hit3']].copy();q['pred']=m.predict(te[features]);out.append(q)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def selection_score(m,y_sd):
    if not np.isfinite(m['spearman']):return -999
    nmae=m['mae']/y_sd if y_sd and np.isfinite(y_sd) else m['mae']
    return m['spearman'] - 0.18*nmae


def fit_selected_target(d,features,model,target,through_year,features_b=None,model_b=None,w=1.0):
    tr=d[(d.season<=through_year)&d[target].notna()]
    ma=clone(model);ma.fit(tr[features],tr[target])
    mb=None
    if model_b is not None:
        mb=clone(model_b);mb.fit(tr[features_b],tr[target])
    return ma,mb


def predict_pair(ma, X, features, mb=None, features_b=None, w=1.0):
    pa=ma.predict(X[features])
    if mb is None:return pa
    pb=mb.predict(X[features_b]);return w*pa+(1-w)*pb


def classifier_validation(d,features,years):
    rows=[]
    for y in years:
        tr=d[d.season<y];te=d[d.season==y]
        if te.empty or tr.hit3.nunique()<2:continue
        clf=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',LogisticRegression(C=.25,max_iter=4000,class_weight='balanced'))])
        clf.fit(tr[features],tr.hit3);q=te[['season','pfr_name','hit3']].copy();q['prob']=clf.predict_proba(te[features])[:,1];rows.append(q)
    z=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if z.empty or z.hit3.nunique()<2:return z,{'auc':np.nan,'brier':np.nan,'n':len(z)}
    return z,{'auc':roc_auc_score(z.hit3,z.prob),'brier':brier_score_loss(z.hit3,z.prob),'n':len(z)}


def evaluate_position(all_df,pos):
    d=all_df[(all_df.position==pos)&(all_df.season<=TRAIN_END)&(all_df.target_valid==1)].copy()
    groups=feature_groups(d,pos); models=model_candidates(); ysd=float(d.loc[d.season<2019,'primary_ppg'].std()) or 1.0
    dev_rows=[]; devpred={}
    for fs,feats in groups.items():
        if not feats:continue
        for mn,m in models.items():
            z=wf_predict(d,feats,m,DEV_YEARS)
            if z.empty:continue
            mm=reg_metrics(z.primary_ppg,z.pred);mm.update({'feature_set':fs,'model':mn,'n_features':len(feats),'score':selection_score(mm,ysd)})
            dev_rows.append(mm);devpred[(fs,mn)]=z
    dev=pd.DataFrame(dev_rows).sort_values('score',ascending=False)
    shortlist=dev.head(10)
    val_rows=[]; valpred={}
    for _,r in shortlist.iterrows():
        fs,mn=r.feature_set,r.model;feats=groups[fs];z=wf_predict(d,feats,models[mn],VALID_YEARS)
        if z.empty:continue
        mm=reg_metrics(z.primary_ppg,z.pred);mm.update({'feature_set':fs,'model':mn,'n_features':len(feats),'score':selection_score(mm,ysd)})
        val_rows.append(mm);valpred[(fs,mn)]=z
    val=pd.DataFrame(val_rows).sort_values('score',ascending=False)
    best=val.iloc[0]; afs,amn=best.feature_set,best.model; afeats=groups[afs]; am=models[amn]
    best_val=valpred[(afs,amn)].copy();best_met=reg_metrics(best_val.primary_ppg,best_val.pred);best_score=selection_score(best_met,ysd)
    blend={'weight':1.0,'other':None}
    for _,r in val.head(6).iloc[1:].iterrows():
        bfs,bmn=r.feature_set,r.model;z=best_val.merge(valpred[(bfs,bmn)][['season','pfr_name','pred']],on=['season','pfr_name'],suffixes=('_a','_b'))
        for w in [.2,.35,.5,.65,.8]:
            pr=w*z.pred_a+(1-w)*z.pred_b;mm=reg_metrics(z.primary_ppg,pr);sc=selection_score(mm,ysd)
            if sc>best_score+0.002:
                best_score=sc;best_met=mm;blend={'weight':w,'other':(bfs,bmn)}
    bfeats=bmodel=None
    if blend['other']:
        bfs,bmn=blend['other'];bfeats=groups[bfs];bmodel=models[bmn]

    # Validation predictions for the promoted blend.
    if bmodel is not None:
        z=best_val.merge(valpred[blend['other']][['season','pfr_name','pred']],on=['season','pfr_name'],suffixes=('_a','_b'))
        z['pred']=blend['weight']*z.pred_a+(1-blend['weight'])*z.pred_b
        val_final=z[['season','pfr_name','primary_ppg','hit3','pred']]
    else: val_final=best_val
    valmet=reg_metrics(val_final.primary_ppg,val_final.pred)

    # Draft-capital-only benchmark over identical validation years.
    cap=groups['capital'];capz=wf_predict(d,cap,models['ridge10'],VALID_YEARS);capmet=reg_metrics(capz.primary_ppg,capz.pred)

    # Freeze architecture before touching 2023 final test.
    tr=d[d.season<FINAL_TEST_YEAR];te=d[d.season==FINAL_TEST_YEAR]
    ma=clone(am);ma.fit(tr[afeats],tr.primary_ppg);fp=ma.predict(te[afeats])
    if bmodel is not None:
        mb=clone(bmodel);mb.fit(tr[bfeats],tr.primary_ppg);fp=blend['weight']*fp+(1-blend['weight'])*mb.predict(te[bfeats])
    final=te[['season','pfr_name','primary_ppg','hit3']].copy();final['pred']=fp;finalmet=reg_metrics(final.primary_ppg,final.pred)
    cm=clone(models['ridge10']);cm.fit(tr[cap],tr.primary_ppg);cf=te[['season','pfr_name','primary_ppg']].copy();cf['pred']=cm.predict(te[cap]);capfinal=reg_metrics(cf.primary_ppg,cf.pred)

    hv,hmet=classifier_validation(d,afeats,VALID_YEARS)
    # One-shot 2023 hit-prob check.
    htr=d[d.season<FINAL_TEST_YEAR]
    hclf=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',LogisticRegression(C=.25,max_iter=4000,class_weight='balanced'))])
    hclf.fit(htr[afeats],htr.hit3);hf=te[['season','pfr_name','hit3']].copy();hf['prob']=hclf.predict_proba(te[afeats])[:,1]
    hfmet={'auc':roc_auc_score(hf.hit3,hf.prob) if hf.hit3.nunique()>1 else np.nan,'brier':brier_score_loss(hf.hit3,hf.prob) if len(hf) else np.nan,'n':len(hf)}

    # Final fits through 2023. Use the frozen primary architecture for secondary regression targets.
    fitted={}
    for target in ['primary_ppg','peak3_ppg','rookie_ppg','avg3_ppg','total3_ppr']:
        fitted[target]=fit_selected_target(d,afeats,am,target,TRAIN_END,bfeats,bmodel,blend['weight'])
    clf=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',LogisticRegression(C=.25,max_iter=4000,class_weight='balanced'))])
    clf.fit(d[afeats],d.hit3)
    job={'position':pos,'features':afeats,'model_name':amn,'model_template':am,'features_b':bfeats,'model_b_template':bmodel,'blend':blend,'fitted':fitted,'classifier':clf}
    joblib.dump(job,MODELS/f'{pos}.joblib')

    summary={'position':pos,'n_train':len(d),'selected_feature_set':afs,'selected_model':amn,'n_features':len(afeats),'blend':str(blend),
             'validation':valmet,'capital_validation':capmet,'final_2023':finalmet,'capital_final_2023':capfinal,
             'hit_validation':hmet,'hit_final_2023':hfmet,
             'validation_mae_improvement_pct':100*(capmet['mae']-valmet['mae'])/capmet['mae'] if capmet['mae'] else np.nan,
             'validation_spearman_gain':valmet['spearman']-capmet['spearman'],
             'final_mae_improvement_pct':100*(capfinal['mae']-finalmet['mae'])/capfinal['mae'] if capfinal['mae'] else np.nan,
             'final_spearman_gain':finalmet['spearman']-capfinal['spearman']}
    return summary,dev,val,val_final,final,hv,hf,job


def pct(hist, vals):
    h=np.asarray(pd.Series(hist).dropna(),float)
    return [100*np.mean(h<=v) if np.isfinite(v) and len(h) else np.nan for v in vals]


def composite_percentile(train,cur,cols,lower_better=None):
    lower_better=lower_better or []
    valid=[c for c in cols if c in train.columns and c in cur.columns and pd.to_numeric(train[c],errors='coerce').notna().sum()>=20]
    if not valid:return pd.Series(np.nan,index=cur.index)
    tz=[];cz=[]
    for c in valid:
        h=pd.to_numeric(train[c],errors='coerce');mu=h.mean();sd=h.std()
        if not np.isfinite(sd) or sd==0:continue
        a=(h-mu)/sd;v=(pd.to_numeric(cur[c],errors='coerce')-mu)/sd
        if any(k in c for k in lower_better):a=-a;v=-v
        tz.append(a.rename(c));cz.append(v.rename(c))
    if not tz:return pd.Series(np.nan,index=cur.index)
    hs=pd.concat(tz,axis=1).mean(axis=1);cs=pd.concat(cz,axis=1).mean(axis=1)
    return pd.Series(pct(hs,cs),index=cur.index)


def current_rankings(all_df,jobs):
    out=[]
    for pos,j in jobs.items():
        train=all_df[(all_df.position==pos)&(all_df.season<=TRAIN_END)&(all_df.target_valid==1)].copy()
        cur=all_df[(all_df.position==pos)&(all_df.season==PREDICT_YEAR)].copy()
        if cur.empty:continue
        for target,label in [('primary_ppg','pred_best2of3_ppg'),('peak3_ppg','pred_peak3_ppg'),('rookie_ppg','pred_rookie_ppg'),('avg3_ppg','pred_avg3_ppg'),('total3_ppr','pred_total3_ppr')]:
            ma,mb=j['fitted'][target]
            cur[label]=predict_pair(ma,cur,j['features'],mb,j['features_b'],j['blend']['weight'])
        cur['hit_probability']=j['classifier'].predict_proba(cur[j['features']])[:,1]
        cur['model_percentile']=pct(train.primary_ppg,cur.pred_best2of3_ppg)
        cur['draft_capital_percentile']=pct(-pd.to_numeric(train.draft_pick,errors='coerce'),-pd.to_numeric(cur.draft_pick,errors='coerce'))
        if pos=='QB':
            prod=[c for c in train if any(k in c for k in ['pass_yardsdropback','pass_yardsgame','qb_qb_rush_fantasy_pg','rush_yardsgame']) and ('peak' in c or 'expadj' in c)]
            eff=[c for c in train if any(k in c for k in ['pass_EPAplay','pass_success','pass_comppct','pass_sack_rate','pass_int_rate']) and ('peak' in c or 'mean' in c)]
        elif pos=='RB':
            prod=[c for c in train if any(k in c for k in ['rb_scrim_yards_per_team_play','rb_scrim_ypg','rec_rec_per_team_pass_att','rush_rush_share']) and ('peak' in c or 'expadj' in c)]
            eff=[c for c in train if any(k in c for k in ['rush_EPAplay','rush_success','rec_EPAplay','rec_success','rec_yards_per_target']) and ('peak' in c or 'mean' in c)]
        else:
            prod=[c for c in train if any(k in c for k in ['rec_rec_per_team_pass_att','rec_target_share','rec_rec_yard_share','rec_yardsgame']) and ('peak' in c or 'expadj' in c)]
            eff=[c for c in train if any(k in c for k in ['rec_EPAplay','rec_success','rec_yards_per_target','rec_catchpct']) and ('peak' in c or 'mean' in c)]
        ath=[c for c in train if any(k in c for k in ['speed_score_v2','forty','vertical','broad_jump','bmi','burst_proxy'])]
        cur['production_percentile']=composite_percentile(train,cur,prod)
        cur['efficiency_percentile']=composite_percentile(train,cur,eff,lower_better=['sack_rate','int_rate'])
        cur['athleticism_percentile']=composite_percentile(train,cur,ath,lower_better=['forty'])
        cur['rank']=cur.pred_best2of3_ppg.rank(ascending=False,method='first').astype(int)
        keep=['position','rank','pfr_name','draft_round','draft_pick','pred_best2of3_ppg','pred_peak3_ppg','pred_rookie_ppg','pred_avg3_ppg','pred_total3_ppr','hit_probability','model_percentile','draft_capital_percentile','production_percentile','efficiency_percentile','athleticism_percentile','college_match','fuzzy_match']
        out.append(cur[keep].sort_values('rank'))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def feature_correlations(df):
    rows=[]
    targets=['primary_ppg','peak3_ppg','rookie_ppg','avg3_ppg','total3_ppr']
    for pos in POSITIONS:
        d=df[(df.position==pos)&(df.season<=TRAIN_END)&(df.target_valid==1)].copy()
        all_feats=feature_groups(d,pos)['full']
        for c in all_feats:
            x=pd.to_numeric(d[c],errors='coerce')
            for target in targets:
                y=pd.to_numeric(d[target],errors='coerce');ok=x.notna()&y.notna()
                if ok.sum()<25 or x[ok].nunique()<3:continue
                rows.append({'position':pos,'feature':c,'target':target,'n':int(ok.sum()),
                             'pearson':pearsonr(x[ok],y[ok]).statistic,'spearman':spearmanr(x[ok],y[ok]).statistic})
    return pd.DataFrame(rows)


def write_report(summaries,rankings,corr):
    L=['# Dynasty Rookie Prospect Models v2.0','',f'Generated {datetime.now(timezone.utc).isoformat()}','',
       'Post-NFL-Draft, position-specific PPR dynasty models for QB, RB, WR, and TE. College data are sourced from cfbfastR/SportsDataverse beginning in 2004; NFL outcomes/draft capital/combine inputs come from nflverse.','',
       '## Primary outcome','',
       'The primary target is the average of a prospect’s two best NFL regular-season PPR point rates over his first three seasons, where each season is fantasy points divided by the full NFL team schedule (16 games through 2020, 17 thereafter). This intentionally assigns zero value to weeks missed or roles not earned and avoids survivorship bias from conditioning only on games with meaningful usage. Secondary outputs predict rookie-year rate, three-year peak rate, three-year average rate, and total three-year PPR points.','',
       '## Validation design','',
       'Model families and feature sets are screened only on earlier walk-forward draft classes. Final architecture is chosen on 2019-2022 walk-forward validation. The 2023 draft class is then evaluated once as a frozen final test and is not used to choose features, algorithms, or blend weights. Draft-capital-only Ridge is the baseline on identical rows.','']
    for s in summaries:
        v=s['validation'];cv=s['capital_validation'];f=s['final_2023'];cf=s['capital_final_2023'];hv=s['hit_validation'];hf=s['hit_final_2023']
        L += [f"## {s['position']}",'',f"Training pool: {s['n_train']} drafted prospects. Selected {s['selected_feature_set']} + {s['selected_model']} ({s['n_features']} features); blend {s['blend']}.",'',
              f"2019-22 validation: MAE {v['mae']:.3f}, RMSE {v['rmse']:.3f}, Pearson {v['pearson']:.3f}, Spearman {v['spearman']:.3f}, R² {v['r2']:.3f}. Draft-capital-only: MAE {cv['mae']:.3f}, Spearman {cv['spearman']:.3f}. Model MAE improvement {s['validation_mae_improvement_pct']:.1f}%, Spearman gain {s['validation_spearman_gain']:.3f}.",
              f"Frozen 2023 test: MAE {f['mae']:.3f}, RMSE {f['rmse']:.3f}, Pearson {f['pearson']:.3f}, Spearman {f['spearman']:.3f}. Draft-capital-only: MAE {cf['mae']:.3f}, Spearman {cf['spearman']:.3f}. Model MAE improvement {s['final_mae_improvement_pct']:.1f}%, Spearman gain {s['final_spearman_gain']:.3f}.",
              f"Hit-probability calibration: 2019-22 AUC {hv['auc']:.3f}, Brier {hv['brier']:.3f}; frozen 2023 AUC {hf['auc']:.3f}, Brier {hf['brier']:.3f}.",'']
    if not rankings.empty:
        L += ['## 2026 post-draft rankings','']
        for pos in POSITIONS:
            z=rankings[rankings.position==pos].head(15)
            if z.empty:continue
            L += [f'### {pos}','','|Rk|Prospect|Pick|Best-2/3 PPG|Peak PPG|Rookie PPG|Hit %|Hist pct|','|---:|---|---:|---:|---:|---:|---:|---:|']
            for _,r in z.iterrows():
                pick='' if pd.isna(r.draft_pick) else int(r.draft_pick)
                L.append(f"|{int(r['rank'])}|{r.pfr_name}|{pick}|{r.pred_best2of3_ppg:.2f}|{r.pred_peak3_ppg:.2f}|{r.pred_rookie_ppg:.2f}|{100*r.hit_probability:.1f}|{r.model_percentile:.1f}|")
            L.append('')
    L += ['## Interpretation','',
          'Historical percentile compares the forecasted primary outcome with the realized primary-outcome distribution of all drafted historical prospects in that position. Production, efficiency, athleticism, and draft-capital percentiles are separate profile diagnostics. A model is not promoted simply because it is more complex; the draft-capital-only benchmark remains the hurdle.']
    (OUT/'REPORT.md').write_text('\n'.join(L))


def main():
    print('Loading cfbfastR/SportsDataverse season summaries...')
    pa=b.load_cfb_table('passing');ru=b.load_cfb_table('rushing');re=b.load_cfb_table('receiving');team=b.load_cfb_table('team_summaries')
    pa,ru,re=prep_college(pa,ru,re,team)
    print('Loading nflverse...')
    draft,players,combine,weekly=b.load_nflverse()
    draft['season']=pd.to_numeric(draft['season'],errors='coerce')
    draft=draft[draft.category.isin(POSITIONS)&draft.season.between(b.TRAIN_DRAFT_START,PREDICT_YEAR)].copy()
    prof=build_profiles(draft,pa,ru,re);prof=add_nfl_meta(prof,players,combine);prof=build_targets(prof,weekly)
    prof.to_csv(OUT/'prospect_pool_v2.csv',index=False)
    match=prof.groupby('position').agg(n=('pfr_name','size'),college_match=('college_match','mean'),fuzzy_rate=('fuzzy_match','mean'),nfl_meta_match=('nfl_meta_match','mean')).reset_index();match.to_csv(OUT/'match_audit.csv',index=False)

    summaries=[];jobs={}
    for pos in POSITIONS:
        print('Stage2 fitting',pos)
        s,dev,val,valp,final,hv,hf,j=evaluate_position(prof,pos);summaries.append(s);jobs[pos]=j
        dev.to_csv(OUT/f'{pos}_dev_grid.csv',index=False);val.to_csv(OUT/f'{pos}_validation_grid.csv',index=False)
        valp.to_csv(OUT/f'{pos}_validation_predictions.csv',index=False);final.to_csv(OUT/f'{pos}_final_2023_predictions.csv',index=False)
        hv.to_csv(OUT/f'{pos}_validation_hit_probs.csv',index=False);hf.to_csv(OUT/f'{pos}_final_2023_hit_probs.csv',index=False)

    rankings=current_rankings(prof,jobs);rankings.to_csv(OUT/'rookie_rankings_2026.csv',index=False)
    corr=feature_correlations(prof);corr.to_csv(OUT/'feature_correlations.csv',index=False)
    flat=[]
    for s in summaries:
        row={k:v for k,v in s.items() if not isinstance(v,dict)}
        for block in ['validation','capital_validation','final_2023','capital_final_2023','hit_validation','hit_final_2023']:
            for k,v in s[block].items():row[f'{block}_{k}']=v
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUT/'model_summary.csv',index=False)
    write_report(summaries,rankings,corr)
    meta={'generated_utc':datetime.now(timezone.utc).isoformat(),'cfb_start':b.START_CFB,'cfb_end':b.LAST_CFB,'development_years':DEV_YEARS,'validation_years':VALID_YEARS,'frozen_final_test_year':FINAL_TEST_YEAR,'prediction_year':PREDICT_YEAR,'summaries':summaries}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2,default=float))
    print(json.dumps(meta,indent=2,default=float))


if __name__=='__main__':
    main()
