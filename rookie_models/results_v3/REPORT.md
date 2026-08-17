# Dynasty Rookie Prospect Models v3.0

Generated 2026-08-17T13:31:29.954212+00:00

Stage 3 was specified before inspecting the 2023 final-test result. It adds leakage-safe post-draft landing-spot context, a prior-year college-to-draft-capital scouting-surprise feature, regularized XGBoost candidates, classifier model selection, and empirical prediction intervals.

The primary target remains the average of a prospect's two best full-schedule PPR point rates in his first three NFL seasons. Draft capital remains the mandatory baseline.

## Validation protocol

2013-2018 is development screening, 2019-2022 is walk-forward model selection, and the 2023 draft class is a one-shot frozen final test. No 2023 result is used to select features, algorithms, weights, or probability models.

## QB

Selected capital_core_landing / rf with 125 features; blend {'weight': 0.8, 'other': ('capital_core_landing', 'extra4')}; hit classifier xgb_cls.

2019-22 validation: MAE 3.790, RMSE 5.328, Pearson 0.675, Spearman 0.614, R² 0.421. Capital-only MAE 3.679, Spearman 0.561.
Frozen 2023: MAE 5.000, RMSE 5.682, Pearson 0.844, Spearman 0.700. Capital-only MAE 3.005, Spearman 0.900.
Hit model: validation AUC 0.861, Brier 0.104; frozen 2023 AUC 1.000, Brier 0.179. Empirical absolute-error bands: 80% 5.87 PPG, 90% 8.30 PPG.

## RB

Selected capital_core_landing / ridge30 with 113 features; blend {'weight': 0.5, 'other': ('capital', 'elastic')}; hit classifier xgb_cls.

2019-22 validation: MAE 2.919, RMSE 3.682, Pearson 0.719, Spearman 0.728, R² 0.500. Capital-only MAE 2.950, Spearman 0.668.
Frozen 2023: MAE 4.976, RMSE 6.032, Pearson 0.837, Spearman 0.893. Capital-only MAE 5.173, Spearman 0.607.
Hit model: validation AUC 0.856, Brier 0.077; frozen 2023 AUC 0.917, Brier 0.151. Empirical absolute-error bands: 80% 4.64 PPG, 90% 6.32 PPG.

## WR

Selected capital_production_landing / extra4 with 371 features; blend {'weight': 1.0, 'other': None}; hit classifier xgb_cls.

2019-22 validation: MAE 3.509, RMSE 4.303, Pearson 0.553, Spearman 0.580, R² 0.294. Capital-only MAE 3.235, Spearman 0.614.
Frozen 2023: MAE 3.366, RMSE 4.227, Pearson 0.485, Spearman 0.640. Capital-only MAE 3.613, Spearman 0.596.
Hit model: validation AUC 0.765, Brier 0.139; frozen 2023 AUC 0.750, Brier 0.159. Empirical absolute-error bands: 80% 5.51 PPG, 90% 6.83 PPG.

## TE

Selected capital_core_landing / extra4 with 137 features; blend {'weight': 0.2, 'other': ('full', 'extra7')}; hit classifier xgb_cls.

2019-22 validation: MAE 1.885, RMSE 2.437, Pearson 0.637, Spearman 0.596, R² 0.384. Capital-only MAE 1.936, Spearman 0.559.
Frozen 2023: MAE 2.853, RMSE 3.191, Pearson 0.446, Spearman 0.517. Capital-only MAE 2.495, Spearman 0.567.
Hit model: validation AUC 0.910, Brier 0.074; frozen 2023 AUC 0.778, Brier 0.187. Empirical absolute-error bands: 80% 2.88 PPG, 90% 4.56 PPG.

## Interpretation

A more complex model is promoted only if pre-2023 validation supports it. The frozen 2023 column is confirmatory evidence, not a tuning target. Market comparisons use historical FantasyPros rookie ADP when the page can be fetched and matched; unavailable years or players are excluded rather than imputed.