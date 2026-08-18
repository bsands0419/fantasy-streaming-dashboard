# Stage 4 robustness audit

This stage does not tune the model. Promotion rules were frozen before reading the completed Stage 3 results.
Robustness and baseline comparisons use only 2019-2022 walk-forward validation. The 2023 draft class remains a separate confirmatory check.
A complex model must beat draft capital on both MAE and Spearman, show stable/calibrated validation behavior, add skill in the hit-probability model, and beat historical rookie ADP when at least 20 matched market rows are available.

## QB
Validation MAE 3.790 vs capital 3.679; Spearman 0.614 vs capital 0.561 (95% bootstrap 0.376 to 0.783).
Calibration slope 0.784; yearly positive-Spearman rate 75.0%; top-5 capture 75.0%; top-10 capture 97.5%.
Hit model AUC 0.861; Brier 0.104 vs prevalence-only 0.139; Brier skill 0.251.
Frozen 2023 MAE 5.000 vs capital 3.005; Spearman 0.700 vs capital 0.900.
Promotion: FAIL — does not beat draft-capital baseline on both validation MAE and Spearman; frozen 2023 shows a severe baseline-relative reversal.

## RB
Validation MAE 2.919 vs capital 2.950; Spearman 0.728 vs capital 0.668 (95% bootstrap 0.590 to 0.827).
Calibration slope 1.188; yearly positive-Spearman rate 100.0%; top-5 capture 75.0%; top-10 capture 77.5%.
Hit model AUC 0.856; Brier 0.077 vs prevalence-only 0.112; Brier skill 0.313.
Frozen 2023 MAE 4.976 vs capital 5.173; Spearman 0.893 vs capital 0.607.
Promotion: PASS — pass.

## WR
Validation MAE 3.509 vs capital 3.235; Spearman 0.580 vs capital 0.614 (95% bootstrap 0.464 to 0.676).
Calibration slope 0.841; yearly positive-Spearman rate 100.0%; top-5 capture 50.0%; top-10 capture 55.0%.
Hit model AUC 0.765; Brier 0.139 vs prevalence-only 0.154; Brier skill 0.101.
Frozen 2023 MAE 3.366 vs capital 3.613; Spearman 0.640 vs capital 0.596.
Promotion: FAIL — does not beat draft-capital baseline on both validation MAE and Spearman.

## TE
Validation MAE 1.885 vs capital 1.936; Spearman 0.596 vs capital 0.559 (95% bootstrap 0.355 to 0.766).
Calibration slope 1.005; yearly positive-Spearman rate 100.0%; top-5 capture 55.0%; top-10 capture 87.5%.
Hit model AUC 0.910; Brier 0.074 vs prevalence-only 0.119; Brier skill 0.379.
Frozen 2023 MAE 2.853 vs capital 2.495; Spearman 0.517 vs capital 0.567.
Promotion: PASS — pass.
