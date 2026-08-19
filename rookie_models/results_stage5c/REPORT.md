# Rookie Models Stage 5C: Uncertainty and Diagnosis

Generated 2026-08-19T02:48:20.088227+00:00

Stage 5C tests whether Stage 5B model-familiarity confidence historically relates to smaller frozen-model OOF errors. Confidence may change displayed prediction bands only if both average-error and tail-error diagnostics support it.

## Confidence/error test

|Pos|N|Spearman(confidence, abs error)|p|Tail errors monotonic?|Adaptive bands?|
|---|---:|---:|---:|---|---|
|QB|176|-0.189|0.012|No|No|
|RB|373|0.089|0.088|No|No|
|WR|525|-0.028|0.525|No|No|
|TE|232|-0.060|0.363|Yes|No|

The tail guard requires the historical 80th- and 90th-percentile absolute errors to satisfy High <= Medium <= Low with at least 15 observations in each group. This prevents a significant average-error correlation from being used to narrow/widen intervals when the actual prediction tails do not behave monotonically.

## Highest diagnostic disagreement in 2026

|Pos|Player|Prospect pct|Classifier hit|Bucket hit|5-comp hit|Confidence|Priority|
|---|---|---:|---:|---:|---:|---|---|
|RB|Jadarian Price|96.8|25.5%|73.7%|40.0%|Medium|High|
|WR|KC Concepcion|91.8|67.3%|46.2%|20.0%|Medium|High|
|WR|Jordyn Tyson|77.7|29.2%|22.9%|60.0%|Medium|High|
|WR|Colbie Young|25.7|4.3%|3.8%|40.0%|High|High|
|WR|Malachi Fields|65.0|6.0%|22.9%|40.0%|Medium|High|
|RB|Eli Heidenreich|59.8|13.9%|9.5%|40.0%|Low|High|
|WR|Chris Brazzell II|56.2|18.7%|9.5%|40.0%|High|High|
|WR|De'Zhaun Stribling|78.9|14.9%|22.9%|40.0%|High|High|
|QB|Ty Simpson|89.2|43.2%|38.9%|20.0%|Medium|High|
|WR|Omar Cooper Jr.|69.1|7.0%|22.9%|0.0%|Medium|High|
|WR|Germie Bernard|66.9|7.2%|22.9%|0.0%|Medium|High|
|QB|Carson Beck|73.3|5.9%|22.9%|0.0%|Medium|High|
|QB|Drew Allar|63.1|10.0%|22.9%|0.0%|High|High|
|WR|Antonio Williams|62.5|9.2%|22.9%|0.0%|High|Medium|
|TE|Sam Roush|68.1|41.6%|19.6%|40.0%|High|High|
|TE|Nate Boerkircher|62.1|22.0%|19.6%|0.0%|High|High|
|TE|Kenyon Sadiq|96.1|36.6%|58.3%|40.0%|Medium|High|
|RB|Jeremiyah Love|100.0|59.0%|73.7%|80.0%|High|High|
|QB|Athan Kaliakmanis|34.1|6.4%|0.0%|20.0%|High|Medium|
|QB|Cole Payton|25.6|5.0%|0.0%|20.0%|Low|High|

## Interpretation

Prospect percentile is a grade, not a success probability. Historical bucket hit rate is empirical calibration. The classifier is a separate supervised probability model. Five-neighbor hit rate is a descriptive comp summary. Model confidence measures familiarity with the frozen feature space. Stage 5C keeps these signals separate and surfaces disagreement rather than averaging them into a false single probability.