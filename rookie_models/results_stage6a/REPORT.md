# Rookie Models Stage 6A: Scouting-Surprise Ablation

Generated 2026-08-19T14:20:59.900537+00:00

Stage 6A holds model classes, hyperparameters, blend weights, target definitions, and walk-forward windows fixed. The experimental comparison is between two fresh refits on the identical saved historical snapshot: one with the frozen scout fields and one with only `scout_expected_log_pick` and `scout_boost` removed. The serialized Stage 3 model is retained as a separate published reference. This avoids confusing tree-model refit drift after CSV round-tripping with the effect of the scout variables.

The scout fields are pre-NFL and leakage-safe with respect to NFL outcomes. This is a feature-taxonomy and robustness test, not an outcome-leakage correction.

## Serialized model versus snapshot refit drift

|Pos|N|Max score drift|Mean score drift|Max hit-prob drift|Mean hit-prob drift|
|---|---:|---:|---:|---:|---:|
|WR|36|0.0461|0.0165|0.0000|0.0000|
|TE|22|0.0283|0.0073|0.0000|0.0000|

## Scout ablation deltas

The table below isolates the scout-variable effect because both sides are fit from the same snapshot. Negative MAE/RMSE/Brier and positive Spearman/Pearson/AUC deltas favor removing the scout fields.

|Pos|Split|Removed P/S|ΔMAE|ΔRMSE|ΔSpearman|ΔPearson|ΔHit AUC|ΔHit Brier|
|---|---|---|---:|---:|---:|---:|---:|---:|
|WR|validation_2019_2022|2/0|+0.001|+0.002|-0.003|-0.001|-0.008|+0.003|
|WR|final_2023|2/0|+0.059|+0.168|-0.044|-0.023|+0.000|+0.000|
|TE|validation_2019_2022|0/2|-0.007|-0.008|-0.002|+0.001|+0.000|+0.000|
|TE|final_2023|0/2|-0.017|-0.005|+0.083|+0.002|+0.000|+0.000|

## Largest 2026 ablation effects

|Pos|Player|With-scout refit|No-scout refit|ΔPPG|With-scout hit|No-scout hit|Δ hit pp|Serialized frozen PPG|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|WR|Omar Cooper Jr.|5.95|5.23|-0.72|7.0%|6.8%|-0.1|5.96|
|WR|Reggie Virgil|3.59|4.23|+0.64|2.5%|2.9%|+0.4|3.62|
|WR|Makai Lemon|10.23|9.69|-0.54|58.0%|55.9%|-2.1|10.26|
|WR|Antonio Williams|5.09|4.57|-0.52|9.2%|8.0%|-1.3|5.09|
|WR|Elijah Sarratt|5.42|4.89|-0.52|14.7%|13.4%|-1.3|5.43|
|WR|Bryce Lance|3.02|3.42|+0.39|2.9%|2.9%|+0.1|3.03|
|WR|Kendrick Law|3.17|2.84|-0.33|1.7%|1.8%|+0.1|3.18|
|WR|Cyrus Allen|2.21|2.52|+0.31|2.7%|3.1%|+0.4|2.22|
|WR|De'Zhaun Stribling|7.57|7.87|+0.30|14.9%|15.7%|+0.9|7.56|
|TE|Max Klare|5.50|5.78|+0.29|32.0%|32.0%|+0.0|5.47|
|TE|Eli Stowers|6.34|6.63|+0.29|37.1%|37.1%|+0.0|6.31|
|WR|Ted Hurst|4.95|4.66|-0.29|22.6%|22.0%|-0.6|4.92|
|WR|Kaden Wetjen|2.88|2.59|-0.29|2.0%|2.0%|+0.0|2.86|
|WR|Chris Brazzell II|4.15|4.42|+0.27|18.7%|20.3%|+1.5|4.12|
|WR|Malachi Fields|5.40|5.66|+0.26|6.0%|7.2%|+1.2|5.40|
|WR|Denzel Boston|8.71|8.95|+0.24|40.1%|38.4%|-1.6|8.74|
|WR|Skyler Bell|2.76|2.98|+0.23|1.8%|2.0%|+0.2|2.78|
|WR|Jordyn Tyson|7.35|7.13|-0.22|29.2%|25.4%|-3.8|7.35|
|WR|Zachariah Branch|3.80|4.02|+0.21|8.4%|8.6%|+0.2|3.78|
|TE|Matthew Hibner|3.31|3.51|+0.20|6.2%|6.2%|+0.0|3.31|
|WR|Kevin Coleman Jr.|2.99|2.80|-0.18|4.9%|4.7%|-0.2|2.96|
|WR|Colbie Young|2.14|2.31|+0.17|4.3%|3.5%|-0.8|2.12|
|WR|Germie Bernard|5.69|5.84|+0.15|7.2%|6.6%|-0.7|5.66|
|TE|Josh Cuevas|2.10|2.25|+0.15|3.4%|3.4%|+0.0|2.11|
|WR|Anthony Smith|1.35|1.49|+0.14|1.8%|1.9%|+0.1|1.34|
|WR|Zavion Thomas|2.55|2.68|+0.13|9.0%|10.3%|+1.3|2.51|
|TE|Marlin Klein|4.30|4.20|-0.10|21.6%|21.6%|+0.0|4.30|
|WR|Brenen Thompson|4.14|4.04|-0.10|3.0%|3.4%|+0.4|4.14|
|WR|Ja'Kobi Lane|3.98|4.07|+0.09|6.2%|7.9%|+1.6|3.99|
|WR|Chris Bell|4.66|4.75|+0.09|9.3%|8.5%|-0.8|4.66|

## Decision framework

No Stage 6 experiment replaces the frozen model automatically. Pre-2023 validation is the selection evidence; the 2023 class remains a frozen confirmation check. If scout removal is better or effectively neutral on validation, Stage 6B should explicitly exclude scout fields from generic production groups. If scout inclusion is clearly beneficial, Stage 6B may retain them, but only as an explicitly named scouting-surprise feature family so their contribution is deliberate and auditable.
