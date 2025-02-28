# Lumi

Lumi is a speech-to-text utility that allows quick voice input activated by double-tapping the Option key.

## Features

- Easy activation with double-tap Option key hotkey
- Single-tap to stop recording (more intuitive)
- Automatically transcribes speech when recording stops using Groq, ElevenLabs APIs, or local MLX Whisper
- Copies transcription to clipboard and automatically pastes it
- Plays lightweight notification sounds when recording starts/stops
- Command-line interface with configuration options
- Cross-platform support for Windows, macOS, and Linux

## Requirements

- Python 3.12+
- PortAudio library (`brew install portaudio` on macOS)
- For cloud transcription (optional):
  - Groq API key or
  - ElevenLabs API key
- For local transcription:
  - MLX (Apple Silicon optimized) - no API key needed

## Installation

### From Source
1. Clone this repository
2. Install dependencies with `uv sync`
3. Run with `uv run -m src.lumi.s2t`

### As Command-line Tool
After installing dependencies:
```bash
uv pip install -e .
```

Then you can use the `lumi` command directly.

## Usage

### Command-line Options

```bash
# Choose transcription service
lumi --service groq           # Use Groq API (default)
lumi --service elevenlabs     # Use ElevenLabs API
lumi --service mlx            # Use local MLX Whisper (no API key needed)

# API keys
lumi --api-key YOUR_GROQ_API_KEY
lumi --service elevenlabs --elevenlabs-api-key YOUR_ELEVENLABS_API_KEY

# Specify models
lumi --service groq --model whisper-tiny
lumi --service elevenlabs --model custom_model
lumi --service mlx --model mlx-community/whisper-large-mlx-q4

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

## Development

- Install dependencies: `uv sync`
- Add dependency: `uv add package_name`
- Lint: `ruff check .`
- Format: `ruff format .`
- Test: `pytest`
- Test specific file: `pytest path/to/test_file.py`
