"""Train slate-relative objectives on prior seasons; select on 2021-23 only."""
from evaluate import *
from catboost import CatBoostRanker
from sequential_search import qb_games, qb_histories
from extended_search import pbp_columns
import hashlib

RESULT = OUT / 'ranking'


def ranking_data():
    t, k = load_data()
    qb = qb_histories(qb_games(), t[['season', 'week', 'game_id', 'team', 'opponent_team']])
    pbp = pd.read_parquet(CACHE / 'pbp_features.parquet')
    pbp['team'] = pbp.team.replace({'LAR': 'LA'})
    out = {}
    for pos, d in [('DST', t), ('K', k)]:
        d = d.merge(qb, on=['game_id', 'team'], validate='many_to_one')
        d = d.merge(pbp, on=['game_id', 'team'], validate='many_to_one')
        d['y'] = d.dst_fantasy if pos == 'DST' else d.k_bucket_points
        d['slate'] = d.season * 100 + d.week
        d['relevance'] = d.groupby('slate').y.rank(method='average', pct=True)
        out[pos] = d.sort_values(['season', 'week', 'game_id', 'team']).reset_index(drop=True)
    return out


def specifications():
    return [{'name': f'{objective}_{fs}', 'objective': objective, 'features': fs}
            for objective in ['QueryRMSE', 'YetiRank'] for fs in ['base', 'context']]


def fit_predict(d, pos, year, spec):
    tr, te = d[d.season < year], d[d.season.eq(year)]
    features = list(DST_V4_FEATURES if pos == 'DST' else K_V42_FEATURES)
    if spec['features'] == 'context':
        features += pbp_columns(pos)
        features += [c for c in d if c.startswith(('qb_', 'opp_qb_'))]
    imp = SimpleImputer(strategy='median')
    x = imp.fit_transform(tr[features])
    xt = imp.transform(te[features])
    m = CatBoostRanker(iterations=350, depth=3, learning_rate=.035,
                      l2_leaf_reg=20, loss_function=spec['objective'],
                      random_seed=20260909, thread_count=2, verbose=False,
                      allow_writing_files=False)
    target = tr.relevance if spec['objective'] == 'YetiRank' else tr.y
    m.fit(x, target, group_id=tr.slate)
    return m.predict(xt)


def baseline_frame(pos, year):
    path = OUT / (f'{pos}_{year}_predictions.csv' if year <= 2023
                  else f'confirmation_{pos}_{year}.csv')
    cols = ['season', 'week', 'game_id', 'team', 'y', 'v42']
    if pos == 'K':
        cols += ['player_id']
    return pd.read_csv(path)[cols]


def rescale(p, score):
    z = p[['season', 'week', 'v42']].copy()
    z['score'] = score
    g = z.groupby(['season', 'week'])
    center = g.score.transform('mean')
    sd = g.score.transform('std').clip(lower=1e-9)
    return (z.score - center) / sd * g.v42.transform('std') + g.v42.transform('mean')


def development():
    RESULT.mkdir(parents=True, exist_ok=True)
    lock = {'development': [2021, 2022, 2023], 'confirmation': [2024, 2025],
            'selection': 'Highest weekly Spearman with MAE and RMSE <= baseline and top3 >= baseline.',
            'promotion': 'Pooled confirmation MAE and RMSE improve, Spearman improves, top3 declines <= .1 and RMSE improves in both individual seasons.',
            'caveat': 'Retrospective confirmation only. Later seasons were examined in prior research.',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'selected': {}}
    rows = []
    for pos, d in ranking_data().items():
        frames = []
        for year in lock['development']:
            p = baseline_frame(pos, year)
            te = d[d.season.eq(year)]
            assert list(zip(p.game_id, p.team)) == list(zip(te.game_id, te.team))
            for spec in specifications():
                score = fit_predict(d, pos, year, spec)
                p[spec['name'] + '_raw'] = score
                scaled = rescale(p, score)
                for w in [.25, .5, 1.]:
                    p[spec['name'] + f'_{w}'] = (1-w)*p.v42 + w*scaled
            frames.append(p)
            print(pos, year, 'ranking fits complete', flush=True)
        p = pd.concat(frames, ignore_index=True)
        p.to_csv(RESULT / f'{pos}_development_predictions.csv', index=False)
        base = metrics(p, p.v42)
        eligible = [(base['weekly_spearman'], {'name': 'v42'}, base)]
        rows.append({'position': pos, 'model': 'v42', 'eligible': True, **base})
        for spec in specifications():
            for w in [.25, .5, 1.]:
                name = spec['name'] + f'_{w}'
                m = metrics(p, p[name])
                ok = m['mae'] <= base['mae'] and m['rmse'] <= base['rmse'] and m['top3'] >= base['top3']
                rows.append({'position': pos, 'model': name, 'eligible': ok, **m})
                if ok:
                    eligible.append((m['weekly_spearman'], {**spec, 'name': name, 'weight': w}, m))
        _, spec, m = max(eligible, key=lambda x: x[0])
        lock['selected'][pos] = {'config': spec, 'metrics': m, 'baseline': base}
        print('SELECTED', pos, spec, m, flush=True)
    pd.DataFrame(rows).to_csv(RESULT / 'development_metrics.csv', index=False)
    (RESULT / 'selection_lock.json').write_text(json.dumps(lock, indent=2) + '\n')


def confirmation():
    from confirm import bootstrap
    lock = json.loads((RESULT / 'selection_lock.json').read_text())
    assert lock['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows, decisions = [], {}
    for pos, d in ranking_data().items():
        spec = lock['selected'][pos]['config']
        if spec['name'] == 'v42':
            decisions[pos] = {'promote': False, 'reason': 'No challenger passed development.'}
            continue
        frames, years_ok = [], []
        for year in lock['confirmation']:
            p = baseline_frame(pos, year)
            te = d[d.season.eq(year)]
            assert list(zip(p.game_id, p.team)) == list(zip(te.game_id, te.team))
            score = fit_predict(d, pos, year, spec)
            p['candidate'] = (1-spec['weight'])*p.v42 + spec['weight']*rescale(p, score)
            frames.append(p)
            b, m = metrics(p, p.v42), metrics(p, p.candidate)
            years_ok.append(m['rmse'] < b['rmse'])
            for name, v in [('v42', b), ('candidate', m)]:
                rows.append({'position': pos, 'period': str(year), 'model': name, **v})
        p = pd.concat(frames, ignore_index=True)
        p.to_csv(RESULT / f'{pos}_confirmation_predictions.csv', index=False)
        b, m = metrics(p, p.v42), metrics(p, p.candidate)
        for name, v in [('v42', b), ('candidate', m)]:
            rows.append({'position': pos, 'period': '2024-2025', 'model': name, **v})
        ok = (m['mae'] < b['mae'] and m['rmse'] < b['rmse']
              and m['weekly_spearman'] > b['weekly_spearman']
              and m['top3'] >= b['top3']-.1 and all(years_ok))
        decisions[pos] = {'config': spec, 'promote': bool(ok), 'baseline': b,
                          'candidate': m, 'bootstrap': bootstrap(p)}
        print(pos, decisions[pos], flush=True)
    pd.DataFrame(rows).to_csv(RESULT / 'confirmation_metrics.csv', index=False)
    (RESULT / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['development', 'confirmation'])
    {'development': development, 'confirmation': confirmation}[p.parse_args().stage]()
