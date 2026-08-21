# Stage 6B Historical Analysis Policy

This policy is locked before Stage 6B feature/model selection.

1. 2026 prospects are excluded from feature selection, model selection, validation, confirmation, and accuracy claims.
2. NFL outcome targets requiring three seasons are evaluated only on draft classes with complete target windows. At the 2026 analysis date, this means labeled outcome analysis ends with the 2023 draft class.
3. TeamRankings SOS uses the final valid TeamRankings snapshot for each college season, discovered from the January following that season. Do not hard-code January 20. Example regression checks: 2023 college season -> 2024-01-09, 2024 college season -> 2025-01-21, 2025 college season -> 2026-01-20.
4. A college season Y maps to draft year Y+1 when Y is the player's final college season.
5. Advanced feature families may use shorter historical windows when source coverage begins later. Maximum predictive accuracy is the objective, subject to sufficient sample size and leakage-safe evaluation.
6. Any shortened-window challenger must be compared with the baseline on the identical matched sample and identical years. Apparent gains caused only by changing the evaluation population do not count.
7. Coverage, sample size, draft-year span, missingness, and year-by-year performance are reported for every candidate feature family.
8. Feature-family promotion is based on walk-forward validation, with the latest fully labeled class retained as confirmation evidence rather than used for tuning.
9. Position-specific windows are allowed. There is no requirement that QB, RB, WR, and TE use the same starting year if source coverage differs.
10. No missing advanced statistic is treated as a true zero unless the source definition establishes that zero is the correct value.
11. TeamRankings SOS is tested as incremental information because the existing model already contains SportsDataverse opponent/team-strength context. It is not automatically retained.
12. When a later-starting feature family materially improves matched-sample out-of-sample accuracy and the sample is sufficient, retaining the shorter-window model is allowed even if it sacrifices older training classes.
