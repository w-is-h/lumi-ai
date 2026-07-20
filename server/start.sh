#!/usr/bin/env bash
# Start the ASR server. Backend via ASR_MODEL (qwen | ark | parakeet), default qwen;
# bind address and port via ASR_HOST / ASR_PORT. Extra arguments go to uvicorn.
# Each backend syncs into its own venv (.venv-<model>) since the extras conflict.
set -euo pipefail
cd "$(dirname "$0")"

export ASR_MODEL="${ASR_MODEL:-qwen}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-$ASR_MODEL}"
HOST="${ASR_HOST:-0.0.0.0}"
PORT="${ASR_PORT:-8010}"

uv sync --extra "$ASR_MODEL"
exec uv run uvicorn asr_server:app --host "$HOST" --port "$PORT" "$@"
