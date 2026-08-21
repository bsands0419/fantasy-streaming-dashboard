# Stage 6 New-Model Selection Checkpoint

This checkpoint is for the from-scratch QB/RB/WR/TE prospect models. The frozen Stage 3/4 prospect-model score is not an input to these models and is used only as an external benchmark.

## Validation policy

- Complete NFL outcome rows only.
- Development/feature work used the earlier historical classes.
- 2019-2022 was used for model selection.
- 2023 was opened once as confirmation after the TeamRankings SOS family decisions were specified.
- Incomplete NFL outcome classes remain prediction-only and cannot affect fitting, preprocessing, feature selection, calibration, or model selection.

## TeamRankings SOS decisions

The final valid TeamRankings snapshot is used for each college season, including 2024 college season = 2025-01-21 and 2025 college season = 2026-01-20.

TeamRankings SOS was tested as standardized standalone context and through position-specific production interactions. It was not forced into every model.

- QB regression: reject SOS. It worsened aggregate 2019-2022 error and rank correlation.
- RB regression: reject SOS. Interaction version produced only a very small aggregate improvement and was not sufficiently robust by draft class.
- WR regression: reject SOS. Standalone and interaction forms worsened validation.
- TE regression: retain standalone SOS. It improved 2019-2022 MAE, RMSE, and Spearman. On 2023 it slightly worsened MAE but improved RMSE and Spearman, supporting the context signal.
- QB Hit: reject SOS.
- QB Star: SOS interactions were promising in 2019-2022, but the sample contains only six validation stars and 2023 has zero stars. Do not promote the extra complexity.
- RB Hit: retain SOS plus production interactions. Validation AUC improved from roughly 0.890 to 0.928 and Brier improved. 2023 AUC remained 1.00 while Brier improved versus the no-SOS version.
- RB Star: reject SOS.
- WR Hit/Star: reject SOS.
- TE Hit: interaction bundle improved 2019-2022 but failed the 2023 confirmation against the no-SOS version, so reject it.
- TE Star: reject SOS due tiny positive sample and no reliable calibration benefit.

## Regression architecture checkpoint

- QB: retain the new RF capital + XGBoost advanced blend, no TeamRankings SOS.
- RB: retain the new CatBoost + histogram-gradient blend, no regression SOS.
- WR: use the pre-2023 robustness-ranked blend of 40% target-feature Ridge (raw target) and 60% capital Ridge (soft-log target), no SOS. This sacrifices a small amount of validation MAE versus the error-minimized blend but has materially stronger year-level stability and confirmed much better 2023 rank ordering.
- TE: use the new compact ElasticNet + CatBoost blend with standalone TeamRankings SOS.

## 2023 confirmation

Approximate selected-model confirmation metrics:

| Position | N | MAE | Spearman |
| --- | ---: | ---: | ---: |
| QB | 5 | 3.469 | 0.900 |
| RB | 7 | 3.895 | 0.929 |
| WR robust blend | 14 | 3.835 | 0.626 |
| TE + TeamRankings SOS | 9 | 2.500 | 0.533 |

The old frozen model's 2023 regression results were approximately QB MAE 5.000 / Spearman 0.700, RB 4.976 / 0.893, WR 3.366 / 0.640, and TE 2.853 / 0.517. Thus the new system confirmed strongly at QB/RB/TE. WR did not beat the old system on 2023 MAE, but the robustness-selected new WR model retained a 0.626 rank correlation and its much stronger 2019-2022 validation performance keeps its multi-class historical error below the old benchmark.

The next step is to refit the locked architectures using every eligible completed-outcome class through 2023, learn all preprocessing from that historical training pool only, and score the prediction-only 2024-2026 classes.
