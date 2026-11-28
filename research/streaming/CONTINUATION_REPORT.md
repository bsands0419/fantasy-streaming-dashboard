# Continued kicker and D/ST accuracy research

Date: September 9, 2026. Continues commit `14744c9` and the [first audit](REPORT.md).

**Decision: retain v4.2 in production.** This continuation tested 200 additional
challenger configurations and blends. A kicker candidate improved historical
point-error and ranking measures in both later seasons, but its top-three
recommendations scored worse overall and the uncertainty intervals include no
improvement. It is a research candidate, not a validated production upgrade.

## What this round added

The first audit already tested direct regressions, tree ensembles, event models,
play-by-play opportunity histories, and component changes. This round examines
different questions:

1. Can errors in previous out-of-year forecasts predict the next week's errors?
2. Does that relationship depend on team, opponent, betting lines, weather,
   recent opportunities, or quarterback tendencies?
3. Does correcting only within-week ordering help without shifting the entire
   slate's average projected score?
4. Does a model trained explicitly on slate-relative outcomes rank starters
   better than a model trained on individual point totals?

The 176 sequential candidates include decayed global error corrections,
team/opponent residual pooling, and regularized residual regressions with two
recency settings and two penalty strengths. Centered variants isolate ranking
changes from changes in the whole slate's mean. The 24 ranking candidates use
CatBoost QueryRMSE or YetiRank, original or expanded features, and three blend
weights. Counts include configurations and blends, not independent model
families. Baseline rows are excluded from these 200 challengers.

New quarterback variables measure prior dropback experience, sack rate,
interception rate, hit rate, EPA per dropback, scramble rate, and the last game's
primary quarterback share. Both offensive and opposing quarterback histories
are available. Source data are the same eight pinned official nflverse
play-by-play files as the first audit, documented in `results/pbp_sources.json`.

**The quarterback identity is a last-used-passer proxy.** All games in a weekly
slate receive features before that week's player/team state is updated. The
proxy cannot anticipate an injury replacement or offseason change, and is not
presented as an archived expected-starter feed. No realized current-game
starter is used to construct that game's inputs.

## Evaluation and selection

Baseline forecasts come from the original annual expanding fits, each trained
only on seasons preceding the forecast season. Residual calibrators learn from
those out-of-year predictions, not fitted training errors. Every weekly update
excludes the entire week being predicted.

- Sequential development uses 2022-2023, with 2021 as residual-training warmup.
- Dedicated ranking development uses 2021-2023, fitting only earlier seasons.
- Later-season confirmation uses 2024 and 2025.
- Selection rules, chosen configurations and source hashes are saved before
  each corresponding confirmation run.
- The initial sequential search was followed by centered variants during
  development, before its confirmation run.
- No 2026 outcomes enter this work.

These later seasons have been examined in previous project research. They are
**retrospective confirmation, not untouched holdouts**. More searching also
increases development-selection bias. None of these results establishes an
accuracy ceiling or an exhaustive optimum.

The strict test required improved point errors without weaker weekly rankings
or a material decline in top-three selection. One D/ST calibration passed
development by a tiny margin, then worsened both MAE and RMSE in confirmation.
No kicker sequential candidate passed strict development. No dedicated
ranking candidate was selected for either position.

## Secondary test: prioritize ranking with a small point-error tolerance

After reviewing the strict experiment, a separately labeled exploratory test
selected from the already-computed development predictions. It allowed at most
1% higher MAE/RMSE and required no development top-three decline, then selected
the highest weekly rank correlation. Its own selection was frozen before
opening its candidate confirmation results. This did not replace or revise the
strict experiment's recorded decisions.

The selected kicker candidate uses a strongly regularized residual regression
on the baseline forecast, market/environment inputs, and the quarterback
proxy. Residual training weights halve every 18 weekly slates. Half of the
within-slate centered correction is applied. The selected D/ST candidate uses
market, quarterback and compact opportunity features without centering.

### 2024-2025 confirmation

| Position | Forecast | MAE, lower is better | RMSE, lower is better | Weekly Spearman, higher is better | Top-three actual mean points |
|---|---|---:|---:|---:|---:|
| Kicker | v4.2 | 3.6658 | 4.6312 | 0.2182 | 10.2222 |
| Kicker | Exploratory candidate | 3.6572 | 4.6211 | 0.2261 | 10.0185 |
| D/ST | v4.2 | 4.0524 | 5.1436 | 0.3348 | 9.4444 |
| D/ST | Exploratory candidate | 4.0548 | 5.1384 | 0.3310 | 9.6111 |

The kicker candidate reduces MAE by 0.0086 points, about 0.23%, and RMSE by
0.0100 points, about 0.22%. Its mean weekly Spearman rises by 0.0079. That is
a correlation change, not a percentage-point change in predictions correct.
Top-three recommendations lose 0.204 actual points on average; the top-ranked
recommendation also declines from 10.8611 to 10.5833 points.

Kicker MAE, RMSE and weekly Spearman each improve in both individual years.
Top-three performance improves in 2024 but drops substantially in 2025:

| Season | Forecast | MAE | RMSE | Weekly Spearman | Top-three actual mean points |
|---|---|---:|---:|---:|---:|
| 2024 | v4.2 | 3.6761 | 4.6977 | 0.2166 | 9.0000 |
| 2024 | Candidate | 3.6667 | 4.6909 | 0.2286 | 9.4815 |
| 2025 | v4.2 | 3.6555 | 4.5637 | 0.2197 | 11.4444 |
| 2025 | Candidate | 3.6477 | 4.5504 | 0.2235 | 10.5556 |

Week-block bootstrap uses 5,000 resamples, keeping each slate together. For
the kicker candidate minus v4.2, 95% intervals are:

- MAE: -0.0285 to +0.0105 points.
- RMSE: -0.0284 to +0.0082 points.
- Weekly Spearman: -0.0168 to +0.0313.

All intervals include zero. They also do not account for all historical model
selection. The secondary kicker gate fails on top-three performance; the
secondary D/ST gate fails on ranking. Neither candidate is promoted.

For comparison, the strict D/ST candidate's confirmation MAE was 4.0533 versus
4.0524 and RMSE 5.1449 versus 5.1436, with identical weekly ordering. Full
strict and secondary decisions, yearly metrics and every prediction are saved
under `results/sequential`, `results/ranking`, and `results/rank_tradeoff`.

## Integrity and practical limits

Five additional tests verify that changing current/future quarterback identities
and outcomes cannot change earlier inputs, that weekly row order cannot change
features or centered corrections, that same-week outcomes cannot affect a
calibrator, and that kicker identities and scoring targets align with baseline
rows. The nine original input/scoring integrity tests were also rerun.

The first audit's limitations still apply: historical lines and weather are
not timestamped waiver-time forecasts, the kicker sample does not reconstruct
every pregame starter with zero opportunities, and top-three results use the
entire board rather than actual waiver availability. Standard bucket kicker
scoring and standard no-yards D/ST scoring are the evaluated targets; these
results do not establish gains for other scoring presets or probabilities.

The most useful next data investment is a timestamped expected-starter and
availability feed, particularly quarterback replacements and offensive-line
absences, joined to archived lines and weather forecasts at the intended
decision time. The last-used-passer experiment provides a reproducible candidate
to revisit with that stronger input, but does not establish that the input will
improve accuracy. The [official nflverse availability documentation](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
explains why retrospective participation and injury feeds cannot automatically
be treated as complete contemporaneous weekly inputs.

## Reproduce this continuation

First follow the [first audit's setup and data instructions](REPORT.md#reproduce).
The baseline prediction files and play-by-play files must already be available.
Run from the repository root:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python research/streaming/sequential_search.py development
python research/streaming/sequential_search.py confirmation
python research/streaming/ranking_search.py development
python research/streaming/ranking_search.py confirmation
python research/streaming/rank_tradeoff.py select
python research/streaming/rank_tradeoff.py confirm
python -W ignore research/streaming/test_sequential.py
python -W ignore research/streaming/test_integrity.py
```

Raw quarterback aggregates are reconstructed from pinned play-by-play inputs
and cached locally. Saved predictions permit inspection of all reported numbers
without refitting. This branch includes the earlier input-integrity repairs,
but leaves model weights and the published dashboard's forecasts unchanged
until reviewed and merged.
