# Fantasy Streaming Dashboard v4.2

Live GitHub Pages dashboard for the frozen v4.2 D/ST and kicker streaming model.

## Dashboard
- D/ST: no-yards, yards-allowed, and custom scoring
- Kicker: bucket, decimal, and custom scoring
- P(10+), P(Top 5), and P(Top 10)
- 32-team weekly board with kicker job-battle flags
- GitHub Pages: `https://bsands0419.github.io/fantasy-streaming-dashboard/`

## Automated refresh architecture

`GitHub Pages -> Cloudflare Worker -> GitHub Actions -> frozen v4.2 inference -> snapshot.js/components.js -> GitHub Pages`

The trained-through-2025 v4.2 model remains frozen. Refreshes update inputs only: current nflverse schedule/team/player data, current Sleeper kicker/depth-chart data, betting market lines when `ODDS_API_KEY` is configured, and Open-Meteo weather when games are within forecast range.

The workflow also runs automatically twice per day and can be run manually from GitHub Actions. A 10-minute cooldown protects the free odds quota and prevents duplicate refreshes.

## September 2026 model and input audit

See [the full accuracy audit](research/streaming/REPORT.md) for chronological model comparisons, new play-by-play variables, confirmation results, and reproducible commands. Neither selected challenger passed the later-season promotion criteria, so the frozen v4.2 models remain the production models.

The [continuation report](research/streaming/CONTINUATION_REPORT.md) tests 200 additional sequential-calibration and slate-ranking challengers, including prior quarterback tendencies. An exploratory kicker candidate slightly improves later-season point errors and rank correlation, but loses top-three selection performance and does not pass promotion. All experiment decisions and predictions are preserved.

The refresh entry point is now `backend/run_refresh.py`. It restores in-season opponent-adjusted and explosive-play inputs, aligns weather with kickoff in UTC, and applies the beginning-of-week cutoff consistently to kicker history. The model artifact remains trained through 2025. Input repairs are tested separately from statistical model promotion.

## One-time setup
1. Upload `v42_runtime_bundle.zip` to `backend/assets/v42_runtime_bundle.zip`.
2. Optional but recommended: add repository Actions secret `ODDS_API_KEY` from a free The Odds API account. Without it, the workflow uses the latest nflverse schedule line as the market fallback.
3. Deploy the Worker in `worker/` to Cloudflare Workers.
4. Add Worker secrets `GITHUB_TOKEN` and `REFRESH_PIN`. The GitHub token should be a fine-grained token restricted to this repository with Actions read/write access.
5. Put the deployed Worker URL in `config.js`.

After those steps, the dashboard button **Refresh v4.2 Model** triggers the entire pipeline and changes its status while the model runs and publishes.
