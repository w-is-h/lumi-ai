#!/usr/bin/env bash
# Start the lumi client: remote backend if the ASR server responds, local MLX otherwise.
# Extra arguments are passed through to lumi.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE_URL="${LUMI_REMOTE_URL:-http://nel:8010}"

if curl -fsS --max-time 2 "$REMOTE_URL/health" >/dev/null 2>&1; then
    echo "ASR server up at $REMOTE_URL — remote backend."
    exec uv run lumi --service remote "$@"
else
    echo "No ASR server at $REMOTE_URL — local MLX."
    exec uv run lumi "$@"
fi
