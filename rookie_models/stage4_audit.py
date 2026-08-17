from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
IN = ROOT / 'results_v3'
OUT = ROOT / 'results_v4_audit'
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = ['QB', 'RB', 'WR', 'TE']
RNG = np.random.default_rng(20260817)


def met(df):
    y = pd.to_numeric(df['primary_ppg'], errors='coerce').to_numpy(float)
    p = pd.to_numeric(df['pred'], errors='coerce').to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 4:
        return {'n': len(y), 'mae': np.nan, 'rmse': np.nan, 'pearson': np.nan, 'spearman': np.nan}
    return {
        'n': len(y),
        'mae': mean_absolute_error(y, p),
        'rmse': mean_squared_error(y, p) ** 0.5,
        'pearson': pearsonr(y, p).statistic,
        'spearman': spearmanr(y, p).statistic,
    }


def bootstrap(df, n=3000):
    d = df.reset_index(drop=True)
    vals = []
    for _ in range(n):
        ix = RNG.integers(0, len(d), len(d))
        vals.append(met(d.iloc[ix]))
    z = pd.DataFrame(vals)
    out = {}
    for c in ['mae', 'rmse', 'pearson', 'spearman']:
        out[c + '_lo'] = z[c].quantile(.025)
        out[c + '_med'] = z[c].quantile(.5)
        out[c + '_hi'] = z[c].quantile(.975)
    return out


def yearly(df):
    rows = []
    for y, g in df.groupby('season'):
        m = met(g)
        m['season'] = int(y)
        rows.append(m)
    return pd.DataFrame(rows)


def calibration(df):
    y = pd.to_numeric(df.primary_ppg, errors='coerce')
    p = pd.to_numeric(df.pred, errors='coerce')
    ok = y.notna() & p.notna()
    y, p = y[ok].to_numpy(), p[ok].to_numpy()
    if len(y) < 5:
        return {'intercept': np.nan, 'slope': np.nan, 'mean_bias': np.nan}
    slope, intercept = np.polyfit(p, y, 1)
    return {'intercept': intercept, 'slope': slope, 'mean_bias': float(np.mean(p - y))}


def topk_capture(df, k):
    vals = []
    for _, g in df.groupby('season'):
        n = min(k, len(g))
        if n == 0:
            continue
        predset = set(g.nlargest(n, 'pred').pfr_name)
        actualset = set(g.nlargest(n, 'primary_ppg').pfr_name)
        vals.append(len(predset & actualset) / n)
    return float(np.mean(vals)) if vals else np.nan


def hit_metrics(path):
    if not path.exists():
        return {'hit_n': 0, 'hit_auc': np.nan, 'hit_brier': np.nan, 'hit_null_brier': np.nan, 'hit_skill': np.nan}
    d = pd.read_csv(path)
    y = pd.to_numeric(d.get('hit3'), errors='coerce')
    p = pd.to_numeric(d.get('prob'), errors='coerce')
    ok = y.notna() & p.notna()
    y, p = y[ok].to_numpy(), p[ok].to_numpy()
    if len(y) == 0:
        return {'hit_n': 0, 'hit_auc': np.nan, 'hit_brier': np.nan, 'hit_null_brier': np.nan, 'hit_skill': np.nan}
    brier = brier_score_loss(y, p)
    base = float(np.mean(y))
    null_brier = float(np.mean((y - base) ** 2))
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    skill = 1 - brier / null_brier if null_brier > 0 else np.nan
    return {'hit_n': len(y), 'hit_auc': auc, 'hit_brier': brier, 'hit_null_brier': null_brier, 'hit_skill': skill}


def main():
    if not IN.exists():
        raise SystemExit('results_v3 not present yet')

    model_summary_path = IN / 'model_summary.csv'
    if not model_summary_path.exists():
        raise SystemExit('stage3 model_summary.csv not present')
    model_summary = pd.read_csv(model_summary_path)

    market_path = IN / 'market_benchmark.csv'
    market = pd.read_csv(market_path) if market_path.exists() else pd.DataFrame()

    summary = []
    yearly_all = []
    decisions = []

    for pos in POSITIONS:
        val_path = IN / f'{pos}_validation_predictions.csv'
        test_path = IN / f'{pos}_final_2023_predictions.csv'
        if not val_path.exists() or not test_path.exists():
            raise SystemExit(f'missing stage3 outputs for {pos}')

        # Promotion robustness is measured on the pre-2023 validation window only.
        # The frozen 2023 class remains confirmatory and is reported separately.
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        vm = met(val)
        ci = bootstrap(val)
        cal = calibration(val)
        tm = met(test)
        yr = yearly(val)
        yr['position'] = pos
        yearly_all.append(yr)
        hm = hit_metrics(IN / f'{pos}_validation_hit_probs.csv')

        row3 = model_summary[model_summary.position.eq(pos)]
        if row3.empty:
            raise SystemExit(f'missing model summary row for {pos}')
        row3 = row3.iloc[0]

        cap_val_mae = pd.to_numeric(pd.Series([row3.get('capital_validation_mae')]), errors='coerce').iloc[0]
        cap_val_sp = pd.to_numeric(pd.Series([row3.get('capital_validation_spearman')]), errors='coerce').iloc[0]
        cap_test_mae = pd.to_numeric(pd.Series([row3.get('capital_final_2023_mae')]), errors='coerce').iloc[0]
        cap_test_sp = pd.to_numeric(pd.Series([row3.get('capital_final_2023_spearman')]), errors='coerce').iloc[0]

        market_row = market[market.position.eq(pos)] if not market.empty and 'position' in market else pd.DataFrame()
        market_n = 0
        market_sp = np.nan
        market_model_sp = np.nan
        market_required = False
        market_pass = True
        if not market_row.empty:
            mr = market_row.iloc[0]
            market_n = int(pd.to_numeric(pd.Series([mr.get('n')]), errors='coerce').fillna(0).iloc[0])
            market_sp = pd.to_numeric(pd.Series([mr.get('market_spearman')]), errors='coerce').iloc[0]
            market_model_sp = pd.to_numeric(pd.Series([mr.get('model_spearman_same_rows')]), errors='coerce').iloc[0]
            # Sparse historical ADP matches are retained as descriptive evidence, not a hard gate.
            market_required = market_n >= 20 and np.isfinite(market_sp) and np.isfinite(market_model_sp)
            if market_required:
                market_pass = bool(market_model_sp > market_sp)

        capital_pass = bool(
            np.isfinite(cap_val_mae) and np.isfinite(cap_val_sp) and
            vm['mae'] < cap_val_mae and vm['spearman'] > cap_val_sp
        )
        robustness_pass = bool(
            vm['spearman'] > 0.10 and ci['spearman_lo'] > -0.05 and
            0.45 < cal['slope'] < 1.55 and
            float((yr.spearman > 0).mean()) >= 0.60
        )
        hit_pass = bool(
            hm['hit_n'] >= 20 and np.isfinite(hm['hit_auc']) and np.isfinite(hm['hit_skill']) and
            hm['hit_auc'] > 0.50 and hm['hit_skill'] > 0
        )
        # Frozen 2023 is confirmatory only. Allow normal single-class noise, but block a severe reversal.
        final_confirmation = bool(
            np.isfinite(cap_test_mae) and np.isfinite(cap_test_sp) and
            tm['mae'] <= 1.20 * cap_test_mae and tm['spearman'] >= cap_test_sp - 0.10
        )

        promote = bool(capital_pass and robustness_pass and hit_pass and market_pass and final_confirmation)
        reasons = []
        if not capital_pass:
            reasons.append('does not beat draft-capital baseline on both validation MAE and Spearman')
        if not robustness_pass:
            reasons.append('fails validation stability/calibration guardrails')
        if not hit_pass:
            reasons.append('hit model does not beat prevalence baseline with positive discrimination')
        if not market_pass:
            reasons.append('does not beat historical rookie ADP on sufficiently matched rows')
        if not final_confirmation:
            reasons.append('frozen 2023 shows a severe baseline-relative reversal')
        if not reasons:
            reasons = ['pass']

        srow = {
            'position': pos,
            **{'validation_' + k: v for k, v in vm.items()},
            **{'validation_' + k: v for k, v in ci.items()},
            **{'validation_' + k: v for k, v in cal.items()},
            **{'final_2023_' + k: v for k, v in tm.items()},
            **hm,
            'capital_validation_mae': cap_val_mae,
            'capital_validation_spearman': cap_val_sp,
            'capital_final_2023_mae': cap_test_mae,
            'capital_final_2023_spearman': cap_test_sp,
            'top5_capture': topk_capture(val, 5),
            'top10_capture': topk_capture(val, 10),
            'yearly_spearman_positive_rate': float((yr.spearman > 0).mean()),
            'yearly_mae_sd': float(yr.mae.std()),
            'market_n': market_n,
            'market_spearman': market_sp,
            'model_spearman_market_rows': market_model_sp,
            'market_required': market_required,
            'capital_pass': capital_pass,
            'robustness_pass': robustness_pass,
            'hit_pass': hit_pass,
            'market_pass': market_pass,
            'final_confirmation': final_confirmation,
        }
        summary.append(srow)
        decisions.append({'position': pos, 'promotion_pass': promote, 'reason': '; '.join(reasons)})

    s = pd.DataFrame(summary)
    s.to_csv(OUT / 'robustness_summary.csv', index=False)
    pd.concat(yearly_all, ignore_index=True).to_csv(OUT / 'yearly_robustness.csv', index=False)
    pd.DataFrame(decisions).to_csv(OUT / 'promotion_guardrails.csv', index=False)

    report = [
        '# Stage 4 robustness audit', '',
        'This stage does not tune the model. Promotion rules were frozen before reading the completed Stage 3 results.',
        'Robustness and baseline comparisons use only 2019-2022 walk-forward validation. The 2023 draft class remains a separate confirmatory check.',
        'A complex model must beat draft capital on both MAE and Spearman, show stable/calibrated validation behavior, add skill in the hit-probability model, and beat historical rookie ADP when at least 20 matched market rows are available.', ''
    ]
    decdf = pd.DataFrame(decisions).set_index('position')
    for _, r in s.iterrows():
        decision = decdf.loc[r.position]
        report += [
            f'## {r.position}',
            f"Validation MAE {r.validation_mae:.3f} vs capital {r.capital_validation_mae:.3f}; Spearman {r.validation_spearman:.3f} vs capital {r.capital_validation_spearman:.3f} (95% bootstrap {r.validation_spearman_lo:.3f} to {r.validation_spearman_hi:.3f}).",
            f"Calibration slope {r.validation_slope:.3f}; yearly positive-Spearman rate {100*r.yearly_spearman_positive_rate:.1f}%; top-5 capture {100*r.top5_capture:.1f}%; top-10 capture {100*r.top10_capture:.1f}%.",
            f"Hit model AUC {r.hit_auc:.3f}; Brier {r.hit_brier:.3f} vs prevalence-only {r.hit_null_brier:.3f}; Brier skill {r.hit_skill:.3f}.",
            f"Frozen 2023 MAE {r.final_2023_mae:.3f} vs capital {r.capital_final_2023_mae:.3f}; Spearman {r.final_2023_spearman:.3f} vs capital {r.capital_final_2023_spearman:.3f}.",
            f"Promotion: {'PASS' if decision.promotion_pass else 'FAIL'} — {decision.reason}.", ''
        ]

    (OUT / 'REPORT.md').write_text('\n'.join(report))
    (OUT / 'manifest.json').write_text(json.dumps({
        'stage': '4-audit',
        'bootstrap_reps': 3000,
        'seed': 20260817,
        'purpose': 'precommitted robustness and baseline-value audit only; no model tuning',
        'promotion_principle': 'complexity must beat draft capital and, when sufficiently observed, historical rookie ADP before promotion',
        'frozen_final_test_year': 2023,
    }, indent=2))


if __name__ == '__main__':
    main()
