# Full Rookie Model Results, 2007-2026

Generated 2026-08-19T18:14:44.665745+00:00

The master CSV contains 1,550 QB/RB/WR/TE prospect rows.

- Historical rows through 2023 use leakage-safe out-of-fold scores.
- 2024-2026 use frozen prospect-class scores.
- Realized `primary_ppg`, `hit3`, `star3`, and `best_rank3` are populated only when the full three-season outcome window is complete.
- `prospect_model_percentile_pooled` is a same-position historical percentile, not a success probability.
- `prospect_model_percentile_temporal` is the historical as-of-draft percentile when enough prior OOF reference rows existed.
- `hit_probability` is available for the reconstructed 2024-2026 frozen prospect classes; it is intentionally blank in this master file for earlier OOF rows rather than mixing non-equivalent classifier estimates.
