# Weekly kicker and D/ST accuracy audit

**Date:** September 9, 2026  
**Repository baseline:** `d76f01c839b537354772402b68a0953e63b030e5`  
**Decision:** retain the frozen v4.2 model weights. Both development-selected challengers failed the later-season promotion criteria. Ship the separately tested input-integrity repairs through normal code review. This work does not establish a predictive accuracy improvement or an accuracy ceiling.

## What was actually tested

Recovered the original model source, fitted artifact, 4,254 D/ST team-games and 4,247 kicker-games covering 2018–2025 from `backend/assets/v42_runtime_bundle.zip`. Downloaded eight seasons of official nflverse play-by-play and joined team/opponent histories to the existing data. Source URLs and SHA-256 hashes are in `results/pbp_sources.json`; the original bundle and selection hashes are in `results/provenance.json`.

The search evaluated **320 position-specific configurations including blends**, not 320 independent model families. It covered Ridge, Bayesian Ridge, Extra Trees, CatBoost, histogram gradient boosting, XGBoost, LightGBM, and Poisson event models. It also tested recent-season training, rare-event shrinkage, integer points-allowed boundaries, kicker component weights, opportunity regressions, attempt-weighted field-goal share, and recent distance-specific kicking skill.

New variables included:

- Opponent field-goal and long-field-goal attempts conceded, scoring chances, and fantasy points conceded.
- Drives reaching the opponent's 35-yard line, red-zone opportunities and touchdown/stall rates.
- Fourth-down go-for-it rate in plausible field-goal range.
- Pace and passing rate in relatively close games, no-huddle usage, pressure and sack conversion.
- Recent and longer-term play volume, fumble-recovery residuals, performance relative to market expectations, and weather interactions.

Raw game outcomes were shifted before constructing history features. New feature tests perturb current/future outcomes and verify that earlier predictions' inputs do not change. No 2026 outcomes were used.

## Evaluation design

For each predicted season, fit using all prior seasons only. Recompute feature histories from previous games. All imputers, scalers, feature selectors and estimators fit exclusively within that training split. The models refit annually in this experiment, matching the frozen-season deployment approach.

- **Development:** 2021, 2022 and 2023, 54 weekly slates.
- **Confirmation:** 2024 and 2025, 36 weekly slates.
- **Targets:** existing standard no-yards D/ST scoring and standard distance-bucket kicker scoring, with no missed-kick deduction.
- **Measures:** MAE, RMSE, Pearson correlation, mean within-week Spearman correlation, and actual points from the top-ranked one, three and five recommendations.
- **Baseline:** reconstructed v4.2 annual fits. Kicker forecasts use the original analytic ensemble. D/ST forecasts use the exact expectation of the original simulation distribution, avoiding random ranking changes from finite Monte Carlo draws. A 100,000-draw comparison validates that expectation.

Before opening confirmation results, `selection_lock.json` recorded the candidates, weights, selection rule and promotion criteria. Development selection minimized RMSE subject to small MAE tolerance and ranking/top-three guardrails. Confirmation required lower MAE and RMSE, no decline in weekly rank correlation, and no material top-three decline. Neither candidate passed.

**These are retrospective confirmation seasons.** Earlier project work had already examined 2024 and 2025, so they cannot be presented as pristine holdouts. Screening many configurations can overfit development results even with chronological training. The later-season failures illustrate why development gains alone were insufficient.

## Results on 2024–2025

The selected defense challenger blended 75% v4.2 with 25% Extra Trees using expanded team-history features. The selected kicker challenger blended 75% v4.2 with 25% Extra Trees fitted to the preceding three seasons using expanded histories.

| Position | Forecast | Games | MAE ↓ | RMSE ↓ | Weekly rank correlation ↑ | Top recommendation points ↑ | Top-three mean points ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| D/ST | Existing v4.2 | 1,088 | **4.0524** | **5.1436** | **0.3348** | **10.6111** | **9.4444** |
| D/ST | Selected challenger | 1,088 | 4.0682 | 5.1556 | 0.3291 | 10.0833 | 9.3704 |
| Kicker | Existing v4.2 | 1,086 | **3.6658** | **4.6312** | **0.2182** | **10.8611** | 10.2222 |
| Kicker | Selected challenger | 1,086 | 3.6734 | 4.6355 | 0.2128 | 10.0278 | **10.2500** |

Lower error and higher correlation are better. Correlation is not a percentage of predictions that were correct. Top recommendation results are averages across entire historical weekly boards, not a waiver-availability simulation.

Week-block bootstrap, 5,000 resamples, keeping each slate together:

| Challenger minus baseline | D/ST 95% interval | Kicker 95% interval |
|---|---:|---:|
| MAE difference | +0.0076 to +0.0242 | +0.0029 to +0.0125 |
| RMSE difference | +0.0044 to +0.0196 | -0.0014 to +0.0101 |
| Weekly rank correlation difference | -0.0121 to +0.0006 | -0.0147 to +0.0035 |

The small development gains did not survive confirmation. Adding complexity or new variables did not justify replacing either existing model. Year-by-year results and every confirmation prediction are in `results/confirmation_metrics.csv` and `results/confirmation_*.csv`.

## Live input problems repaired

The frozen artifact is not modified. `backend/run_refresh.py` loads it and installs these input repairs:

1. **Restore six opponent-adjusted rate fields during the season.** The old refresh left their current-season raw values missing. The inference builder drops missing history values, so these signals could remain based on old games. The repaired formulas match all six stored historical definitions within floating-point tolerance.
2. **Refresh explosive-play rates from current-season play-by-play.** Refresh offensive 20+/40+ rates and the opposing defense's allowed rates. Log missing coverage explicitly. The updated 2024 source agrees with all historical 40+ rates and all but one 20+ count; that discrepancy is documented rather than silently forcing a match.
3. **Use actual kickoff time for weather.** Convert schedule Eastern time to UTC, request UTC hourly weather and select the nearest covered hour. This handles West Coast games, international fixtures and UTC date changes. Missing kickoff times or forecast coverage are reported.
4. **Apply the same week cutoff to all kicker-history fields.** Previously, some base kicker features could consume already-completed games in the requested week during a midweek refresh. Every kicker history field now observes the beginning-of-week cutoff used elsewhere in the model.

Nine integrity tests cover leakage, scoring targets, rate reconstruction, preserved frozen history, weather-hour selection, same-week exclusion, explosive-play mirroring and deterministic-versus-simulated forecasts. A pull-request workflow runs the core integrity suite; the optional full play-by-play comparison requires the downloaded 2024 input and was run locally.

### Paired replay of the missing-input repair

A separate 2024–2025 D/ST replay used the same frozen annual model fits and prediction builder for both paths. One path removed current-season explosive-play and adjusted-rate history to reproduce the missing-input behavior; the other supplied the complete historical fields. It isolates those missing fields, not every behavior of the full old refresh pipeline. It does not measure the weather-time or kicker-cutoff repairs.

| D/ST input path | MAE ↓ | RMSE ↓ | Weekly rank correlation ↑ | Top-five mean points ↑ |
|---|---:|---:|---:|---:|
| Missing current-season fields | 4.0485 | 5.1410 | 0.3337 | 8.8778 |
| Restored fields | 4.0524 | 5.1436 | 0.3348 | 9.0389 |

The repair slightly improved ranking and top-five selection in this replay, but did not reduce overall prediction error. It restores the intended model inputs; it is not evidence of a general accuracy upgrade.

## Limits and the next useful data

- Historical schedule lines and recorded game weather are not timestamped Tuesday-waiver or Sunday-morning input snapshots. Therefore these results cannot establish accuracy at a specific earlier decision time.
- The kicker sample contains players recorded in the historical stats feed. It does not reconstruct every pregame roster or zero-opportunity starter. Full-board selection is not a simulation of available waiver options.
- In this experiment, neither new model nor new feature set demonstrated enough consistent improvement to promote. Other methods or genuinely better data could still help; no exhaustive optimum has been established.
- Further material research should prioritize timestamped expected starting quarterback, offensive-line availability, kicker job status, betting-line and forecast snapshots. Replacing those with final starters, end-of-season injury knowledge, or hindsight closing information would produce misleading gains.
- nflverse documents that its injury source stopped after 2024 and that recent participation data are postseason releases. Those feeds cannot simply be assumed available for every historical weekly decision. See the [official data availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

The new drive and play-level variables use [official nflverse play-by-play releases](https://github.com/nflverse/nflverse-data/releases/tag/pbp). Kickoff handling follows the [nflverse schedule data](https://nflreadr.nflverse.com/reference/load_schedules.html).

## Reproduce

Run from the repository root with Python 3.12:

```bash
mkdir -p backend/runtime
unzip -q backend/assets/v42_runtime_bundle.zip -d backend/runtime
python -m pip install -r research/streaming/requirements.txt
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python research/streaming/pbp_features.py
python research/streaming/evaluate.py
python research/streaming/extended_search.py
python research/streaming/component_search.py
python research/streaming/select_candidate.py
python research/streaming/confirm.py
python research/streaming/test_integrity.py
```

Run selection only on the development prediction files. `confirm.py` evaluates the already selected candidates without reselection. Cached fits and raw downloads are ignored by Git; source hashes permit data-drift checks. The input replay additionally needs the official schedule CSV saved as `research/streaming/cache/games.csv`, then `python research/streaming/replay_inputs.py`.
