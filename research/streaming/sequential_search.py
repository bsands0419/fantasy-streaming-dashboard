"""Sequential calibration of genuinely out-of-year v4.2 forecast errors.

2021 supplies calibration warmup; 2022-23 select configurations; 2024-25
provide retrospective confirmation. No current-week outcomes enter a fit.
The QB proxy is the last-used primary QB, never the realized current starter.
"""
from evaluate import *
from collections import defaultdict
from itertools import product
import hashlib

RESULT = OUT / 'sequential'
QB_COLS = ['db', 'sack', 'interception', 'qb_hit', 'epa', 'qb_scramble']


def qb_games():
    path = CACHE / 'sequential_qb_games.parquet'
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    cols = ['game_id', 'season', 'week', 'season_type', 'posteam',
            'passer_player_id', 'rusher_player_id', 'qb_dropback', 'qb_scramble',
            'qb_kneel', 'qb_spike', 'sack', 'interception', 'qb_hit', 'epa']
    for year in range(2018, 2026):
        p = pd.read_parquet(CACHE / f'pbp_{year}.parquet', columns=cols)
        p = p[p.season_type.eq('REG') & p.qb_dropback.eq(1)
              & ~p.qb_kneel.eq(1) & ~p.qb_spike.eq(1)].copy()
        p['qb'] = p.passer_player_id.where(~p.qb_scramble.eq(1), p.rusher_player_id)
        p['team'] = p.posteam.replace({'LAR': 'LA', 'OAK': 'LV', 'SD': 'LAC'})
        p['db'] = 1.
        frames.append(p.dropna(subset=['team', 'qb']).groupby(
            ['season', 'week', 'game_id', 'team', 'qb'], as_index=False)[QB_COLS].sum())
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def qb_histories(raw, schedule):
    """Emit all rows in a slate before updating any player or team state."""
    stats = defaultdict(lambda: np.zeros(len(QB_COLS)))
    last = {}
    rows = []
    batches = {key: g for key, g in raw.groupby(['season', 'week'])}
    for key, slate in schedule.sort_values(['season', 'week']).groupby(['season', 'week']):
        for r in slate.itertuples():
            row = {'game_id': r.game_id, 'team': r.team}
            for prefix, team in [('qb_', r.team), ('opp_qb_', r.opponent_team)]:
                q, share = last.get(team, (None, 0.))
                a = stats[q]
                row[prefix + 'prior_dropbacks'] = a[0]
                row[prefix + 'previous_share'] = share
                for i, c in enumerate(QB_COLS[1:], 1):
                    row[prefix + c + '_rate'] = a[i] / a[0] if a[0] else np.nan
            rows.append(row)
        g = batches.get(key)
        if g is not None:
            # Decayed player history keeps earlier experience without career domination.
            for r in g.itertuples():
                stats[r.qb] = .9 * stats[r.qb] + np.array([getattr(r, c) for c in QB_COLS])
            for team, z in g.groupby('team'):
                top = z.sort_values(['db', 'qb'], ascending=[False, True]).iloc[0]
                last[team] = (top.qb, float(top.db / z.db.sum()))
    return pd.DataFrame(rows)


def data():
    t, k = load_data()
    qb = qb_histories(qb_games(), t[['season', 'week', 'game_id', 'team', 'opponent_team']])
    t, k = augment(t, k)
    pbp = pd.read_parquet(CACHE / 'pbp_features.parquet')
    pbp['team'] = pbp.team.replace({'LAR': 'LA'})
    output = {}
    for pos, d in [('DST', t), ('K', k)]:
        ps = []
        for year in range(2021, 2026):
            path = OUT / (f'{pos}_{year}_predictions.csv' if year < 2024
                          else f'confirmation_{pos}_{year}.csv')
            ps.append(pd.read_csv(path))
        p = pd.concat(ps, ignore_index=True)
        keys = ['season', 'week', 'game_id', 'team'] + (['player_id'] if pos == 'K' else [])
        d = d.merge(p[keys + ['v42', 'y']], on=keys, validate='one_to_one')
        d = d.merge(qb, on=['game_id', 'team'], validate='many_to_one')
        d = d.merge(pbp, on=['game_id', 'team'], validate='many_to_one')
        d['clock'] = (d.season - 2021) * 18 + d.week
        d['residual'] = d.y - d.v42
        d['forecast_squared'] = d.v42 ** 2
        d['outdoor_wind'] = d.wind.fillna(0) * (1 - d.roof_dome)
        output[pos] = d.sort_values(['season', 'week', 'game_id', 'team']).reset_index(drop=True)
    return output


def feature_sets(pos):
    basic = ['v42', 'implied_team_total', 'implied_opp_total', 'is_home',
             'roof_dome', 'outdoor_wind', 'rest_diff']
    qb = [prefix + c for prefix in ['qb_', 'opp_qb_']
          for c in ['prior_dropbacks', 'previous_share', 'sack_rate',
                    'interception_rate', 'qb_hit_rate', 'epa_rate', 'qb_scramble_rate']]
    if pos == 'DST':
        compact = ['def_qbhit_rate_ewm10', 'opp_off_sack_rate_allowed_ewm10',
                   'opp_off_int_rate_ewm10', 'opp_plays_off_ewm4',
                   'pbp_allowed_pressure_rate_ewm16', 'opp_pbp_neutral_pass_rate_ewm16']
    else:
        compact = ['pbp_range_drives_ewm16', 'pbp_range_stall_rate_ewm16',
                   'pbp_fourth_go_rate_ewm16', 'opp_pbp_allowed_fg_att_ewm16',
                   'pbp_deep_fg_att_ewm16', 'fg_att_ewm16', 'pat_att_ewm16']
    return {'slope': ['v42'], 'market': basic, 'qb': basic + qb,
            'compact': basic + compact, 'combined': basic + compact + qb}


def configs(pos):
    yield {'name': 'v42', 'method': 'baseline'}
    for half, prior in product([18, 54], [64, 256]):
        yield {'name': f'offset_h{half}_p{prior}', 'method': 'offset', 'half': half, 'prior': prior}
    for group, half, prior in product(['team', 'opponent_team'], [18, 54], [16, 64]):
        yield {'name': f'group_{group}_h{half}_p{prior}', 'method': 'group',
               'group': group, 'half': half, 'prior': prior}
    for fs, half, alpha in product(feature_sets(pos), [18, 54], [100, 1000]):
        yield {'name': f'ridge_{fs}_h{half}_a{alpha}', 'method': 'ridge',
               'fs': fs, 'half': half, 'alpha': alpha}


def forecast(tr, te, cfg, pos):
    base = te.v42.to_numpy()
    if cfg['method'] == 'baseline':
        return base
    assert tr.clock.max() < te.clock.min()
    w = 2. ** (-(te.clock.min() - tr.clock.to_numpy()) / cfg['half'])
    r = tr.residual.to_numpy()
    if cfg['method'] == 'offset':
        delta = np.sum(w * r) / (np.sum(w) + cfg['prior'])
    elif cfg['method'] == 'group':
        c = cfg['group']
        delta = np.array([np.sum(w[tr[c].eq(v)] * r[tr[c].eq(v)]) /
                          (np.sum(w[tr[c].eq(v)]) + cfg['prior']) for v in te[c]])
    else:
        cols = feature_sets(pos)[cfg['fs']]
        imp = SimpleImputer(strategy='median', add_indicator=True)
        x = imp.fit_transform(tr[cols])
        sc = StandardScaler().fit(x, sample_weight=w)
        x = sc.transform(x)
        # Regularize the intercept too, so tiny recent residual samples stay conservative.
        x = np.column_stack([np.ones(len(x)), x])
        xt = np.column_stack([np.ones(len(te)), sc.transform(imp.transform(te[cols]))])
        m = Ridge(alpha=cfg['alpha'], fit_intercept=False).fit(x, r, sample_weight=w)
        delta = m.predict(xt)
    delta = np.clip(delta, -3, 3)
    if cfg.get('center'):
        delta = (delta - np.mean(delta)) * cfg['weight']
    return base + delta


def predictions(d, pos, cfgs, years):
    frames = []
    for (year, week), te in d[d.season.isin(years)].groupby(['season', 'week']):
        tr = d[d.clock < te.clock.min()]
        p = te[['season', 'week', 'game_id', 'team', 'y', 'v42']].copy()
        if pos == 'K':
            p['player_id'] = te.player_id
        for cfg in cfgs:
            p[cfg['name']] = forecast(tr, te, cfg, pos)
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def development():
    RESULT.mkdir(parents=True, exist_ok=True)
    all_data = data()
    lock = {'development': [2022, 2023], 'warmup': [2021], 'confirmation': [2024, 2025],
            'selection': 'Lowest RMSE with MAE <= baseline and weekly Spearman >= baseline; top3 loss <= .1.',
            'promotion': 'Pooled confirmation RMSE and MAE improve; weekly Spearman does not decline; top3 loss <= .1; RMSE improves in both individual seasons.',
            'caveat': 'Retrospective confirmation, already examined in earlier research. No pristine holdout remains.',
            'selected': {}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rows = []
    for pos, d in all_data.items():
        cs = list(configs(pos))
        p = predictions(d, pos, cs, [2022, 2023])
        # Follow-up development experiment: isolate within-slate ranking changes
        # from the global bias shift. Confirmation outcomes remain unopened here.
        extra = []
        for cfg in cs:
            if cfg['method'] not in ['group', 'ridge']:
                continue
            correction = p[cfg['name']] - p.v42
            centered = correction - correction.groupby([p.season, p.week]).transform('mean')
            for weight in [.5, 1.]:
                c = {**cfg, 'name': cfg['name'] + f'_centered{weight}',
                     'center': True, 'weight': weight}
                p[c['name']] = p.v42 + weight * centered
                extra.append(c)
        cs += extra
        p.to_csv(RESULT / f'{pos}_development_predictions.csv', index=False)
        base = metrics(p, p.v42)
        eligible = []
        for cfg in cs:
            m = metrics(p, p[cfg['name']])
            ok = (m['mae'] <= base['mae'] and m['weekly_spearman'] >= base['weekly_spearman']
                  and m['top3'] >= base['top3'] - .1)
            rows.append({'position': pos, 'model': cfg['name'], 'eligible': ok, **m})
            if ok:
                eligible.append((m['rmse'], cfg, m))
        _, cfg, m = min(eligible, key=lambda v: v[0])
        lock['selected'][pos] = {'config': cfg, 'metrics': m, 'baseline': base}
        print(pos, cfg, m, flush=True)
    pd.DataFrame(rows).to_csv(RESULT / 'development_metrics.csv', index=False)
    (RESULT / 'selection_lock.json').write_text(json.dumps(lock, indent=2) + '\n')


def confirmation():
    from confirm import bootstrap
    lock = json.loads((RESULT / 'selection_lock.json').read_text())
    assert lock['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows, decisions = [], {}
    for pos, d in data().items():
        cfg = lock['selected'][pos]['config']
        p = predictions(d, pos, [cfg], lock['confirmation'])
        p['candidate'] = p[cfg['name']]
        p.to_csv(RESULT / f'{pos}_confirmation_predictions.csv', index=False)
        bm, cm = metrics(p, p.v42), metrics(p, p.candidate)
        yearly = []
        for period, g in [('2024', p[p.season.eq(2024)]), ('2025', p[p.season.eq(2025)]), ('2024-2025', p)]:
            b, c = metrics(g, g.v42), metrics(g, g.candidate)
            if period != '2024-2025':
                yearly.append(c['rmse'] < b['rmse'])
            for name, m in [('v42', b), ('candidate', c)]:
                rows.append({'position': pos, 'period': period, 'model': name, **m})
        promote = (cm['rmse'] < bm['rmse'] and cm['mae'] < bm['mae']
                   and cm['weekly_spearman'] >= bm['weekly_spearman']
                   and cm['top3'] >= bm['top3'] - .1 and all(yearly))
        decisions[pos] = {'config': cfg, 'promote': bool(promote), 'baseline': bm,
                          'candidate': cm, 'bootstrap': bootstrap(p)}
        print(pos, json.dumps(decisions[pos]), flush=True)
    pd.DataFrame(rows).to_csv(RESULT / 'confirmation_metrics.csv', index=False)
    (RESULT / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['development', 'confirmation'])
    {'development': development, 'confirmation': confirmation}[p.parse_args().stage]()
