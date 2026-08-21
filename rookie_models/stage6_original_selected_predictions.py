from __future__ import annotations

from pathlib import Path
import pandas as pd

import stage6_original_models as reg
import stage6_original_classifiers as clf

ROOT = Path(__file__).resolve().parent
POOL = ROOT / 'results_v4' / 'prospect_pool_v4.csv'
REG_OUT = ROOT / 'results_stage6_newmodels' / 'original_branch'
CLF_OUT = ROOT / 'results_stage6_newmodels' / 'original_classifiers'


def regression_confirmation(d: pd.DataFrame) -> int:
    sel_path = REG_OUT / 'selected_models.csv'
    if not sel_path.exists():
        print('regression selected_models not available; skipping')
        return 0
    ss = pd.read_csv(sel_path)
    picked = []
    for _, r in ss.iterrows():
        pos = str(r['position'])
        sets, _ = reg.feature_sets(d, pos)
        sname = str(r['feature_set'])
        kind = str(r['kind'])
        if sname not in sets:
            raise KeyError(f'{pos}: missing feature set {sname}')
        o = reg.walk(d, pos, sets[sname], kind, [reg.CONFIRM])
        o['feature_set'] = sname
        picked.append(o)
    if picked:
        out = pd.concat(picked, ignore_index=True)
        out.to_csv(REG_OUT / 'confirmation_2023_predictions.csv', index=False)
        print('wrote regression confirmation predictions', len(out))
        return len(out)
    return 0


def classification_confirmation(d: pd.DataFrame) -> int:
    sel_path = CLF_OUT / 'selected_models.csv'
    if not sel_path.exists():
        print('classifier selected_models not available; skipping')
        return 0
    ss = pd.read_csv(sel_path)
    picked = []
    for _, r in ss.iterrows():
        pos = str(r['position'])
        target = str(r['target'])
        sets, _ = clf.feature_sets(d, pos, target)
        sname = str(r['feature_set'])
        kind = str(r['kind'])
        if sname not in sets:
            raise KeyError(f'{pos}/{target}: missing feature set {sname}')
        o = clf.walk(d, pos, target, sets[sname], kind, [clf.CONFIRM])
        o['feature_set'] = sname
        picked.append(o)
    if picked:
        out = pd.concat(picked, ignore_index=True)
        out.to_csv(CLF_OUT / 'confirmation_2023_predictions.csv', index=False)
        print('wrote classifier confirmation predictions', len(out))
        return len(out)
    return 0


def main():
    d = pd.read_csv(POOL, low_memory=False)
    d['season'] = pd.to_numeric(d['season'], errors='coerce')
    d['target_valid'] = pd.to_numeric(d['target_valid'], errors='coerce').fillna(0)
    d['primary_ppg'] = pd.to_numeric(d['primary_ppg'], errors='coerce')
    for t in ['hit3', 'star3']:
        d[t] = pd.to_numeric(d[t], errors='coerce')
    nr = regression_confirmation(d)
    nc = classification_confirmation(d)
    print('total rows', nr + nc)


if __name__ == '__main__':
    main()
