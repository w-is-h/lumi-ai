"""Transcription backends: local MLX Whisper and a self-hosted remote ASR server."""

import logging
import os
import traceback

import requests

logger = logging.getLogger(__name__)

SERVICES = ("mlx", "remote")

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_REMOTE_URL = "http://localhost:8010"


def transcribe(audio_file: str, service: str = "mlx", model: str | None = None) -> str:
    """Transcribe an audio file. Returns the text, or an error marker string on failure.

    Args:
        audio_file: Path to the audio file.
        service: "mlx" (local Whisper) or "remote" (self-hosted ASR server).
        model: MLX model name; falls back to MLX_WHISPER_MODEL env var, then the default.
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return "[Transcription error: File not found]"
    if os.path.getsize(audio_file) == 0:
        logger.error("Audio file is empty")
        return "[Transcription error: File is empty]"

    try:
        if service == "mlx":
            return _transcribe_mlx(audio_file, model)
        elif service == "remote":
            return _transcribe_remote(audio_file)
        raise ValueError(f"Unknown service: {service} (options: {', '.join(SERVICES)})")
    except Exception as e:
        logger.error(f"Error during {service} transcription: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        return f"[Transcription error: {e}]"


def _transcribe_mlx(audio_file: str, model: str | None) -> str:
    import mlx_whisper  # deferred: importing mlx is slow and remote mode doesn't need it

    model = model or os.environ.get("MLX_WHISPER_MODEL", DEFAULT_MLX_MODEL)
    logger.debug(f"Transcribing {audio_file} with MLX Whisper model {model}")
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=model)
    return result["text"]


def _transcribe_remote(audio_file: str) -> str:
    base_url = os.environ.get("LUMI_REMOTE_URL", DEFAULT_REMOTE_URL)
    url = f"{base_url}/transcribe"
    logger.debug(f"Sending {audio_file} to {url}")
    with open(audio_file, "rb") as audio:
        files = {"file": (os.path.basename(audio_file), audio, "audio/wav")}
        response = requests.post(url, files=files, timeout=300)
    response.raise_for_status()
    return response.json().get("text", "")
