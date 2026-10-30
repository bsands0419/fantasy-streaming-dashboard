"""Secondary, explicitly exploratory ranking-first selection from existing trials.

This does not overwrite the strict point-accuracy promotion decisions. The
alternative objective was specified after reviewing the strict experiment.
"""
from sequential_search import *
RESULT = OUT / 'rank_tradeoff'


def select():
    RESULT.mkdir(exist_ok=True)
    report = pd.read_csv(OUT / 'sequential/development_metrics.csv')
    lock = {'development': [2022, 2023], 'confirmation': [2024, 2025],
            'secondary_exploratory_analysis': True,
            'selection': 'Highest weekly Spearman, tie break lowest RMSE then name, with <=1% higher MAE/RMSE and no top3 decline.',
            'confirmation_gate': 'Weekly Spearman improves in each year and pooled; pooled top3 does not decline; pooled MAE/RMSE <=1% worse.',
            'caveat': 'Specified after strict experiment review; retrospective confirmation is not an untouched holdout.',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'selected': {}}
    for pos, g in report.groupby('position'):
        b = g[g.model.eq('v42')].iloc[0]
        q = g[(g.mae <= b.mae*1.01) & (g.rmse <= b.rmse*1.01) & (g.top3 >= b.top3)]
        win = q.sort_values(['weekly_spearman', 'rmse', 'model'], ascending=[False, True, True]).iloc[0]
        cs = list(configs(pos))
        cs += [{**cfg, 'name': cfg['name'] + f'_centered{w}', 'center': True, 'weight': w}
               for cfg in list(cs) if cfg['method'] in ['group', 'ridge'] for w in [.5, 1.]]
        lock['selected'][pos] = next(c for c in cs if c['name'] == win.model)
    (RESULT / 'selection_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(lock, indent=2), flush=True)


def confirm_tradeoff():
    from confirm import bootstrap
    lock = json.loads((RESULT / 'selection_lock.json').read_text())
    assert lock['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows, decisions = [], {}
    for pos, d in data().items():
        cfg = lock['selected'][pos]
        p = predictions(d, pos, [cfg], lock['confirmation'])
        p['candidate'] = p[cfg['name']]
        p.to_csv(RESULT / f'{pos}_confirmation_predictions.csv', index=False)
        year_improvement = []
        for period, g in [('2024', p[p.season.eq(2024)]), ('2025', p[p.season.eq(2025)]), ('2024-2025', p)]:
            b, c = metrics(g, g.v42), metrics(g, g.candidate)
            if period != '2024-2025':
                year_improvement.append(c['weekly_spearman'] > b['weekly_spearman'])
            for name, m in [('v42', b), ('candidate', c)]:
                rows.append({'position': pos, 'period': period, 'model': name, **m})
        ok = (all(year_improvement) and c['weekly_spearman'] > b['weekly_spearman']
              and c['top3'] >= b['top3'] and c['mae'] <= b['mae']*1.01 and c['rmse'] <= b['rmse']*1.01)
        decisions[pos] = {'passes_secondary_gate': bool(ok), 'config': cfg, 'baseline': b,
                          'candidate': c, 'bootstrap': bootstrap(p)}
        print(pos, json.dumps(decisions[pos]), flush=True)
    pd.DataFrame(rows).to_csv(RESULT / 'confirmation_metrics.csv', index=False)
    (RESULT / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['select', 'confirm'])
    {'select': select, 'confirm': confirm_tradeoff}[p.parse_args().stage]()
