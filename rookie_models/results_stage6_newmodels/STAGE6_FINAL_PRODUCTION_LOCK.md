# Stage 6 Final Production Lock

Status: FROZEN SELECTION, FINAL REFIT PENDING REPRODUCTION GATE

## Governing protocol
- Selection window: 2019-2022 only.
- 2023: one-shot confirmation only, never a tuning window.
- 2024-2026: prediction-only.
- Training/refit eligibility: complete three-year NFL outcome rows only.
- Calibration: preserve the selected raw probability behavior. Do not add post-hoc tuning after the 2023 confirmation.
- A reconstructed implementation cannot inherit the Stage 6 label unless it reproduces the frozen canonical 2019-2022 OOF predictions under `stage6_reproduction_gate.py`.

## Frozen production components

| Task | Position | Target | Frozen architecture | SOS decision |
| --- | --- | --- | --- | --- |
| Regression | QB | continuous | RF capital + XGBoost advanced blend | no SOS |
| Regression | RB | continuous | CatBoost + histogram-gradient blend | no SOS |
| Regression | WR | continuous | 40% target-feature Ridge raw-target + 60% capital Ridge soft-log robustness blend | no SOS |
| Regression | TE | continuous | compact ElasticNet + CatBoost blend | standalone TeamRankings SOS retained |
| Classification | QB | hit3 | canonical Stage 6 no-SOS classifier | no SOS |
| Classification | QB | star3 | canonical Stage 6 no-SOS classifier | no SOS |
| Classification | RB | hit3 | canonical Stage 6 classifier | TeamRankings SOS + production interactions retained |
| Classification | RB | star3 | canonical Stage 6 no-SOS classifier | no SOS |
| Classification | WR | hit3 | canonical Stage 6 no-SOS classifier | no SOS |
| Classification | WR | star3 | canonical Stage 6 no-SOS classifier | no SOS |
| Classification | TE | hit3 | canonical Stage 6 no-SOS classifier | SOS interactions rejected after 2023 confirmation |
| Classification | TE | star3 | canonical Stage 6 no-SOS classifier | no SOS |

## Canonical confirmation metrics

Regression:
- QB: validation MAE 3.638185, Spearman 0.727635; 2023 MAE 3.469145, Spearman 0.900000.
- RB: validation MAE 2.691814, Spearman 0.726652; 2023 MAE 3.894667, Spearman 0.928571.
- WR robustness blend: validation MAE 3.242626, Spearman 0.622927; 2023 MAE 3.834810, Spearman 0.626374.
- TE standalone SOS: validation MAE 1.772091, Spearman 0.599243; 2023 MAE 2.499808, Spearman 0.533333.

Classification:
- QB hit3 no-SOS: validation AUC 0.836735, Brier 0.114089; 2023 AUC 0.250000, Brier 0.232574. Retained because the one-positive 2023 sample was too small to override the pre-confirmation selection.
- QB star3 no-SOS: validation AUC 0.824074, Brier 0.108479; 2023 had zero positives, so SOS complexity was not promoted.
- RB hit3 SOS + production interactions: validation AUC 0.928485, Brier 0.067678; 2023 AUC 1.000000, Brier 0.079854.
- RB star3 no-SOS: validation AUC 0.958333, Brier 0.046819; 2023 AUC 1.000000, Brier 0.114578.
- WR hit3 no-SOS: validation AUC 0.834967, Brier 0.123883; 2023 AUC 0.725000, Brier 0.169407.
- WR star3 no-SOS: validation AUC 0.743622, Brier 0.091582; 2023 AUC 0.636364, Brier 0.165730.
- TE hit3 no-SOS: validation AUC 0.905000, Brier 0.069275; 2023 AUC 0.722222, Brier 0.172336. This explicitly supersedes the later manual chat lock that incorrectly promoted TE hit3 SOS interactions.
- TE star3 no-SOS: validation AUC 0.939394, Brier 0.030006; 2023 AUC 0.750000, Brier 0.153405.

## Superseded artifacts
Any manually assembled lock that retained frozen pre-Stage-6 WR regression or QB classifiers, or promoted TE hit3 SOS interactions, is superseded by this checkpoint-derived lock. Those manual decisions were produced before the full repository provenance was recovered.

## Final refit gate
The next production action is not additional model selection. It is:
1. Reproduce the frozen canonical 2019-2022 OOF predictions.
2. If and only if the strict reproduction gate passes, refit the frozen architectures on all eligible completed 2014-2023 outcome rows.
3. Score 2024, 2025, and 2026 prospects without using any future outcome information.
