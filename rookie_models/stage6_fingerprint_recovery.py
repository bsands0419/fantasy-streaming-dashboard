from __future__ import annotations

import base64
import gzip
import io
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import stage2 as s2
import stage6_original_classifiers as orig

ROOT = Path(__file__).resolve().parent
POOL = ROOT / 'results_v4' / 'prospect_pool_v4.csv'
REF = ROOT / 'results_stage6_newmodels' / 'advanced_selected_validation_classification_oof.b64'
MODELS = ROOT / 'trained_v3'
OUT = ROOT / 'results_stage6_fingerprint_recovery'
OUT.mkdir(parents=True, exist_ok=True)

VAL_YEARS = [2019, 2020, 2021, 2022]
SEED = 20260821
LOCKED = {
    ('RB', 'hit3'): 'cat',
    ('RB', 'star3'): 'hist',
    ('WR', 'hit3'): 'logit',
    ('WR', 'star3'): 'logit',
    ('TE', 'hit3'): 'cat',
    ('TE', 'star3'): 'logit',
}
FORBIDDEN = (
    'primary_ppg', 'peak3_ppg', 'rookie_ppg', 'avg3_ppg', 'total3_ppr',
    'hit3', 'star3', 'best_rank3', 'target_valid', 'prospect_model',
    'outcome_percentile', 'pred_'
)


def decode_reference(path: Path) -> pd.DataFrame:
    enc = path.read_text().strip()
    raw = base64.b64decode(enc)
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    d = pd.read_csv(io.BytesIO(raw))
    if 'name' in d.columns and 'pfr_name' not in d.columns:
        d = d.rename(columns={'name': 'pfr_name'})
    if 'prob' in d.columns and 'pred' not in d.columns:
        d = d.rename(columns={'prob': 'pred'})
    d['season'] = pd.to_numeric(d['season'], errors='coerce')
    d['pred'] = pd.to_numeric(d['pred'], errors='coerce')
    d = d[d.season.isin(VAL_YEARS)].copy()
    if 'variant' in d.columns:
        z = d[d.variant.astype(str).eq('baseline')].copy()
        if len(z):
            d = z
    return d


def clean_features(fs, columns):
    out = []
    for c in fs:
        if c not in columns:
            continue
        lc = c.lower()
        if any(x in lc for x in FORBIDDEN):
            continue
        if c not in out:
            out.append(c)
    return out


def candidate_sets(d: pd.DataFrame, pos: str, target: str) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    groups = s2.feature_groups(d, pos)
    for name, fs in groups.items():
        use = clean_features(fs, d.columns)
        if use:
            sets[f'group__{name}'] = use

    # Recovered original Stage 6 target-specific ranking logic, fit only on <=2018.
    ranked_sets, audit = orig.feature_sets(d, pos, target)
    for name, fs in ranked_sets.items():
        use = clean_features(fs, d.columns)
        if use:
            sets[f'orig__{name}'] = use

    ranked = [c for c in audit.feature.tolist() if c in d.columns]
    draft_capital = clean_features(groups.get('capital', []), d.columns)
    ranked_noncap = [c for c in ranked if c not in draft_capital]
    for n in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
        top = clean_features(ranked[:n], d.columns)
        if top:
            sets[f'ranked_top{n}'] = top
        cap_top = clean_features(draft_capital + ranked_noncap[:n], d.columns)
        if cap_top:
            sets[f'draftcap_plus_top{n}'] = cap_top

    # Exact frozen Stage 3 feature lists are useful architecture fingerprints even
    # though the frozen score itself is never used as a feature here.
    jp = MODELS / f'{pos}.joblib'
    if jp.exists():
        job = joblib.load(jp)
        for key in ['features', 'features_b', 'base_features']:
            fs = job.get(key) or []
            use = clean_features(fs, d.columns)
            if use:
                sets[f'stage3__{key}'] = use
        union = clean_features(list(job.get('features') or []) + list(job.get('features_b') or []), d.columns)
        if union:
            sets['stage3__union'] = union

    # De-duplicate identical lists while preserving the first, most interpretable name.
    unique = {}
    seen = set()
    for name, fs in sets.items():
        sig = tuple(fs)
        if sig in seen:
            continue
        seen.add(sig)
        unique[name] = fs
    return unique


def model_specs(family: str):
    if family == 'logit':
        return [('logit_exact', None)]
    if family == 'hist':
        return [('hist_exact', None)]
    if family == 'cat':
        # Small, project-grounded CatBoost family. The depth/lr/l2 pairs come from
        # the Stage 6 CatBoost regression constructors; weighting modes cover the
        # plausible classification implementations without outcome-based tuning.
        return [
            ('cat3_unweighted', dict(depth=3, learning_rate=.03, l2_leaf_reg=8, weight='none', imp=False)),
            ('cat4_unweighted', dict(depth=4, learning_rate=.025, l2_leaf_reg=16, weight='none', imp=False)),
            ('cat3_balanced', dict(depth=3, learning_rate=.03, l2_leaf_reg=8, weight='balanced', imp=False)),
            ('cat4_balanced', dict(depth=4, learning_rate=.025, l2_leaf_reg=16, weight='balanced', imp=False)),
            ('cat3_scaled', dict(depth=3, learning_rate=.03, l2_leaf_reg=8, weight='scaled', imp=False)),
            ('cat4_scaled', dict(depth=4, learning_rate=.025, l2_leaf_reg=16, weight='scaled', imp=False)),
            ('cat3_balanced_imp', dict(depth=3, learning_rate=.03, l2_leaf_reg=8, weight='balanced', imp=True)),
            ('cat4_balanced_imp', dict(depth=4, learning_rate=.025, l2_leaf_reg=16, weight='balanced', imp=True)),
        ]
    raise KeyError(family)


def make_model(spec_name: str, spec: dict | None, positive_rate: float):
    if spec_name == 'logit_exact':
        return Pipeline([
            ('imp', SimpleImputer(strategy='median', add_indicator=True)),
            ('sc', StandardScaler()),
            ('m', LogisticRegression(C=.25, max_iter=3000, class_weight='balanced', random_state=SEED)),
        ])
    if spec_name == 'hist_exact':
        return Pipeline([
            ('imp', SimpleImputer(strategy='median', add_indicator=True)),
            ('m', HistGradientBoostingClassifier(
                max_iter=220, learning_rate=.035, max_leaf_nodes=7,
                l2_regularization=5, min_samples_leaf=12, random_state=SEED,
            )),
        ])

    kwargs = dict(
        iterations=450,
        depth=int(spec['depth']),
        learning_rate=float(spec['learning_rate']),
        l2_leaf_reg=float(spec['l2_leaf_reg']),
        loss_function='Logloss',
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    if spec['weight'] == 'balanced':
        kwargs['auto_class_weights'] = 'Balanced'
    elif spec['weight'] == 'scaled':
        kwargs['scale_pos_weight'] = min((1.0 - positive_rate) / max(positive_rate, 1e-4), 8.0)
    cat = CatBoostClassifier(**kwargs)
    if spec.get('imp'):
        return Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)), ('m', cat)])
    return cat


def walk(d: pd.DataFrame, pos: str, target: str, fs: list[str], spec_name: str, spec: dict | None) -> pd.DataFrame:
    out = []
    dd = d[d.position.eq(pos)].copy()
    for yr in VAL_YEARS:
        tr = dd[(dd.season.lt(yr)) & dd.target_valid.eq(1) & dd[target].notna()].copy()
        va = dd[(dd.season.eq(yr)) & dd.target_valid.eq(1) & dd[target].notna()].copy()
        use = [c for c in fs if pd.to_numeric(tr[c], errors='coerce').notna().any()]
        if not use or va.empty or tr[target].nunique() < 2:
            continue
        X = tr[use].apply(pd.to_numeric, errors='coerce')
        V = va[use].apply(pd.to_numeric, errors='coerce')
        pr = float(tr[target].mean())
        m = make_model(spec_name, spec, pr)
        m.fit(X, tr[target].astype(int))
        p = m.predict_proba(V)[:, 1]
        out.append(pd.DataFrame({
            'season': yr,
            'position': pos,
            'target': target,
            'pfr_name': va.pfr_name.values,
            'candidate_pred': p,
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fingerprint(candidate: pd.DataFrame, ref: pd.DataFrame, pos: str, target: str) -> dict:
    r = ref[(ref.position.eq(pos)) & ref.target.eq(target)][['season', 'position', 'target', 'pfr_name', 'pred']].copy()
    m = r.merge(candidate, on=['season', 'position', 'target', 'pfr_name'], how='inner')
    if not len(m):
        return dict(reference_n=len(r), matched_n=0, rmse=np.inf, mae=np.inf, max_abs=np.inf, pearson=np.nan, spearman=np.nan)
    diff = m.candidate_pred.to_numpy(float) - m.pred.to_numpy(float)
    try:
        pe = float(pearsonr(m.candidate_pred, m.pred).statistic)
    except Exception:
        pe = np.nan
    try:
        sp = float(spearmanr(m.candidate_pred, m.pred).statistic)
    except Exception:
        sp = np.nan
    return {
        'reference_n': len(r),
        'matched_n': len(m),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'mae': float(np.mean(np.abs(diff))),
        'max_abs': float(np.max(np.abs(diff))),
        'pearson': pe,
        'spearman': sp,
    }


def main():
    ref = decode_reference(REF)
    d = pd.read_csv(POOL, low_memory=False)
    d['season'] = pd.to_numeric(d.season, errors='coerce')
    d['target_valid'] = pd.to_numeric(d.target_valid, errors='coerce').fillna(0)
    for t in ['hit3', 'star3']:
        d[t] = pd.to_numeric(d[t], errors='coerce')

    rows = []
    feature_manifest = {}
    for (pos, target), family in LOCKED.items():
        sets = candidate_sets(d, pos, target)
        feature_manifest[f'{pos}_{target}'] = {k: v for k, v in sets.items()}
        ref_n = len(ref[(ref.position.eq(pos)) & ref.target.eq(target)])
        print(f'FINGERPRINT {pos}/{target} family={family} reference_n={ref_n} sets={len(sets)}')
        for sname, fs in sets.items():
            # First pass keeps CatBoost to realistic compact/medium families. The cheap
            # linear/hist models can test the larger native families as well.
            if family == 'cat' and len(fs) > 520:
                continue
            for spec_name, spec in model_specs(family):
                try:
                    cand = walk(d, pos, target, fs, spec_name, spec)
                    met = fingerprint(cand, ref, pos, target)
                    rows.append({
                        'position': pos, 'target': target, 'family': family,
                        'feature_set': sname, 'n_features_listed': len(fs),
                        'model_spec': spec_name, **met,
                    })
                    print(pos, target, sname, spec_name, 'rmse', met['rmse'], 'max', met['max_abs'])
                except Exception as exc:
                    rows.append({
                        'position': pos, 'target': target, 'family': family,
                        'feature_set': sname, 'n_features_listed': len(fs),
                        'model_spec': spec_name, 'reference_n': ref_n, 'matched_n': 0,
                        'rmse': np.inf, 'mae': np.inf, 'max_abs': np.inf,
                        'pearson': np.nan, 'spearman': np.nan, 'error': repr(exc),
                    })
                    print('ERROR', pos, target, sname, spec_name, repr(exc))

    allr = pd.DataFrame(rows)
    allr.to_csv(OUT / 'fingerprint_candidates.csv', index=False)
    best_rows = []
    for (pos, target), g in allr.groupby(['position', 'target']):
        z = g.copy()
        z['complete_match'] = z.matched_n.eq(z.reference_n)
        z = z.sort_values(['complete_match', 'rmse', 'max_abs'], ascending=[False, True, True])
        top = z.head(25).copy()
        top.to_csv(OUT / f'{pos}_{target}_top25.csv', index=False)
        best = z.iloc[0].to_dict()
        best['pass_1e6'] = bool(best['complete_match'] and np.isfinite(best['max_abs']) and best['max_abs'] <= 1e-6)
        best_rows.append(best)

    best = pd.DataFrame(best_rows)
    best.to_csv(OUT / 'fingerprint_best.csv', index=False)
    (OUT / 'candidate_feature_manifest.json').write_text(json.dumps(feature_manifest, indent=2))
    summary = {
        'selection_data': 'canonical frozen 2019-2022 OOF predictions only',
        'uses_2023_for_search': False,
        'uses_2024_2026_outcomes': False,
        'strict_pass_threshold_max_abs': 1e-6,
        'all_locked_stage6_classifier_baselines_reproduced': bool(best.pass_1e6.all()) if len(best) else False,
        'components': best[['position', 'target', 'feature_set', 'model_spec', 'reference_n', 'matched_n', 'rmse', 'max_abs', 'pass_1e6']].to_dict('records'),
    }
    (OUT / 'fingerprint_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
