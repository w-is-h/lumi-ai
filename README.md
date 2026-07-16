# Lumi

Speech-to-text for macOS. Double-tap the Option key, speak, tap once to stop — the transcription is pasted where your cursor is and copied to the clipboard.

This was `vibe-coded` using claude-code, R1 and o3-mini.

## Quick Start

```bash
brew install portaudio   # required by pyaudio
pip install lumi-ai      # or: uv pip install lumi-ai
lumi
```

Then:
1. **Double-tap Option** → start recording
2. Speak
3. **Single-tap Option** → stop recording; the transcription is pasted and in your clipboard
4. **Ctrl+C** in the terminal exits

NOTE: on first run the MLX backend downloads its model, which takes a while. `Speech-to-text service started.` means it's ready.

## Backends

- `mlx` (default): local Whisper via [mlx-whisper](https://github.com/ml-explore/mlx-examples). Apple Silicon only, no API key, nothing leaves the machine.
- `remote`: a self-hosted ASR server. Lumi sends the recording as multipart WAV to `POST {LUMI_REMOTE_URL}/transcribe` and reads `text` from the JSON response. Set the server address with the `LUMI_REMOTE_URL` environment variable. The server itself lives in [`server/`](server/) — one FastAPI app with three switchable models (parakeet / qwen / ark).

## Usage

```bash
lumi                          # hotkey mode, local MLX Whisper
lumi --service remote         # hotkey mode, self-hosted server
lumi --model mlx-community/whisper-large-v3-turbo   # pick the MLX model
lumi recording.wav            # transcribe a file and exit
lumi --no-auto-paste          # copy to clipboard only, don't paste
lumi --debug                  # verbose logging
```

Environment variables: `MLX_WHISPER_MODEL` (MLX model name), `LUMI_REMOTE_URL` (remote server base URL).

## Requirements

- macOS on Apple Silicon (for the `mlx` backend), Python 3.12+
- PortAudio (`brew install portaudio`)
- Recordings are written to the system temp directory and left for the OS to clean up

## Development

```bash
uv sync           # install dependencies
uv run lumi       # run from source
ruff check .      # lint
ruff format .     # format
uv run pytest     # tests
```
