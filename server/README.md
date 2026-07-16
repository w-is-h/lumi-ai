# Lumi ASR server

Self-hosted transcription server for lumi's `remote` backend. One FastAPI app,
three switchable models — the dependency stacks conflict, so exactly one is
installed at a time via uv extras.

| ASR_MODEL | Model | Extra | Notes |
|---|---|---|---|
| `parakeet` | nvidia/parakeet-tdt-0.6b-v3 | `--extra parakeet` | NeMo. 25 EU languages incl. Croatian. Fastest. |
| `qwen` | Qwen/Qwen3-ASR-1.7B | `--extra qwen` | qwen-asr package. Best English WER, no Croatian. |
| `ark` | AutoArk-AI/ARK-ASR-0.6B | `--extra ark` | transformers + `trust_remote_code`. Croatian + best leaderboard WER, least proven. |

## Running (on the GPU box)

```bash
cd server
uv sync --extra parakeet   # switching models: re-run with the other extra
ASR_MODEL=parakeet uv run uvicorn asr_server:app --host 0.0.0.0 --port 8010
```

Swap `parakeet` for `qwen` or `ark` in both places to test another model.
First start downloads the weights from Hugging Face (~1.5–4 GB per model).
The model loads at startup; the server is ready when uvicorn logs
"Application startup complete".

If the box is reachable beyond your own network, bind a private interface
(e.g. a VPN/tailnet IP) instead of 0.0.0.0 — the server has no auth.

## API

```bash
curl http://localhost:8010/health
curl -X POST http://localhost:8010/transcribe -F "file=@audio.wav"
# -> {"text": "...", "duration": 1.23, "model_name": "...", "file_name": "audio.wav"}
```

The lumi client points here via `LUMI_REMOTE_URL` (see the repo README).
Audio is expected as 16 kHz mono WAV — the lumi client records in that format.
