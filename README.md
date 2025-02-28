# Lumi

Lumi is a speech-to-text utility that allows quick voice input activated by double-tapping the Option key.

## Features

- Easy activation with double-tap Option key hotkey
- Single-tap to stop recording (more intuitive)
- Automatically transcribes speech when recording stops using Groq, ElevenLabs APIs, or local MLX Whisper
- Copies transcription to clipboard and automatically pastes it
- Plays lightweight notification sounds when recording starts/stops
- Command-line interface with configuration options
- Primarily tested on macOS (may work on other platforms but not fully tested)

## Requirements

- Python 3.12+
- PortAudio library (`brew install portaudio` on macOS)
- For local transcription (default):
  - MLX (Apple Silicon Macs only) - no API key needed
- For cloud transcription (optional):
  - Groq API key or
  - ElevenLabs API key

## Installation

### From PyPI
```bash
pip install lumi-ai
```
Or with uv:
```bash
uv pip install lumi-ai
```

After installation, you can use the `lumi` command directly from your terminal.

### From Source
1. Clone this repository
2. Install dependencies with `uv sync`
3. Run with `uv run -m src.lumi.s2t`

### Development Installation
For development:
```bash
uv pip install -e .
```

## Usage

### Command-line Options

```bash
# Choose transcription service
lumi --service mlx            # Use local MLX Whisper (default, no API key needed)
lumi --service groq           # Use Groq API
lumi --service elevenlabs     # Use ElevenLabs API

# API keys
lumi --api-key YOUR_GROQ_API_KEY
lumi --service elevenlabs --elevenlabs-api-key YOUR_ELEVENLABS_API_KEY

# Specify models
lumi --service groq --model whisper-large-v3-turbo
lumi --service elevenlabs --model scribe_v1
lumi --service mlx --model mlx-community/whisper-large-v3-turbo

# Other options
lumi --no-auto-paste          # Disable auto-pasting
lumi --debug                  # Enable debug logging
```

### Basic Usage

1. Double-tap the Option key to START recording
2. Speak clearly into your microphone
3. Single-tap Option to STOP recording
4. The transcription will be copied to your clipboard and automatically pasted at your cursor position

### Keyboard Controls

- **Double-tap Option**: Start recording
- **Single-tap Option**: Stop recording
- **Ctrl+C**: Exit application

## Advanced Features

- Automatic fallback to alternative audio input devices
- Temporary recordings stored in system temp directory
- Sound notifications for recording start/stop
- Auto-paste functionality (can be disabled with --no-auto-paste)
- Platform-aware keyboard shortcuts (Cmd+V on macOS, Ctrl+V elsewhere)
- Debug mode for troubleshooting with detailed logging

## Platform Support

- **macOS**: Fully supported and tested
- **Linux/Windows**: Basic functionality may work but not extensively tested
- **MLX Whisper**: Only works on Apple Silicon Macs

## Development

- Install dependencies: `uv sync`
- Add dependency: `uv add package_name`
- Lint: `ruff check .`
- Format: `ruff format .`
- Test: `pytest`
- Test specific file: `pytest path/to/test_file.py`
