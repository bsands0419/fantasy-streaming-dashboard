# Cloudflare refresh gateway

This Worker is the authenticated gateway between the public GitHub Pages dashboard and the GitHub Actions v4.2 refresh workflow. It keeps GitHub credentials out of browser JavaScript.

Required Worker secrets:
- `GITHUB_TOKEN`: a fine-grained GitHub personal access token scoped only to `bsands0419/fantasy-streaming-dashboard` with Actions read/write access.
- `REFRESH_PIN`: a PIN/password you choose for the dashboard refresh button.

Deploy this Worker in Cloudflare, then put its `https://...workers.dev` URL into the repository's `config.js` as `window.V42_REFRESH_ENDPOINT`.
