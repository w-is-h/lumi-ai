"""Transcription backends: local MLX Whisper and a self-hosted remote ASR server."""

import logging
import os
import tempfile
import traceback

import requests

from lumi import audio as audio_prep

logger = logging.getLogger(__name__)

SERVICES = ("mlx", "remote")

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_REMOTE_URL = "http://nel:8010"


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
        audio_file = _gate_silences_to_temp(audio_file)
        if service == "mlx":
            return _transcribe_mlx(audio_file, model)
        elif service == "remote":
            return _transcribe_remote(audio_file)
        raise ValueError(f"Unknown service: {service} (options: {', '.join(SERVICES)})")
    except Exception as e:
        logger.error(f"Error during {service} transcription: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        return f"[Transcription error: {e}]"


def _gate_silences_to_temp(audio_file: str) -> str:
    """Collapse long silences; returns a gated temp copy, or the original on failure.

    Best-effort: a recording that can't be preprocessed (odd format) is
    transcribed as-is rather than failing.
    """
    try:
        samples, rate = audio_prep.load_wav(audio_file)
        gated = audio_prep.gate_silences(samples, rate)
        if len(gated) == len(samples):
            return audio_file
        fd, gated_file = tempfile.mkstemp(suffix=".wav", prefix="lumi_gated_")
        os.close(fd)
        audio_prep.save_wav(gated_file, gated, rate)
        logger.debug(
            f"Silence gating: {len(samples) / rate:.1f}s -> {len(gated) / rate:.1f}s"
        )
        return gated_file
    except Exception as e:
        logger.warning(f"Silence gating skipped: {e}")
        return audio_file


def _transcribe_mlx(audio_file: str, model: str | None) -> str:
    import mlx_whisper  # deferred: importing mlx is slow and remote mode doesn't need it

    model = model or os.environ.get("MLX_WHISPER_MODEL", DEFAULT_MLX_MODEL)
    logger.debug(f"Transcribing {audio_file} with MLX Whisper model {model}")
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=model)
    return result["text"]


def _transcribe_remote(audio_file: str) -> str:
    """POST to the remote server, splitting long audio into chunks the server's
    models can take in one forward pass (~30s windows) and sending them all in
    a single batch request."""
    base_url = os.environ.get("LUMI_REMOTE_URL", DEFAULT_REMOTE_URL)

    try:
        samples, rate = audio_prep.load_wav(audio_file)
    except Exception as e:
        # Not a plain mono WAV (e.g. an mp3 passed on the CLI): send as-is,
        # the server decodes other formats itself.
        logger.debug(f"Chunking skipped ({e}); sending {audio_file} whole")
        return _post_single(base_url, audio_file)

    chunks = audio_prep.split_at_silences(samples, rate)
    logger.debug(f"Sending {audio_file} to {base_url} in {len(chunks)} chunk(s)")
    if len(chunks) == 1:
        return _post_single(base_url, audio_file)

    chunk_files = []
    try:
        for i, chunk in enumerate(chunks):
            fd, chunk_file = tempfile.mkstemp(suffix=".wav", prefix=f"lumi_chunk{i}_")
            os.close(fd)
            audio_prep.save_wav(chunk_file, chunk, rate)
            chunk_files.append(chunk_file)
        files = [
            ("files", (os.path.basename(p), open(p, "rb").read(), "audio/wav"))
            for p in chunk_files
        ]
    finally:
        for p in chunk_files:
            os.unlink(p)

    response = requests.post(f"{base_url}/transcribe_batch", files=files, timeout=300)
    response.raise_for_status()
    texts = response.json().get("texts", [])
    return " ".join(t.strip() for t in texts if t.strip())


def _post_single(base_url: str, audio_file: str) -> str:
    with open(audio_file, "rb") as audio:
        files = {"file": (os.path.basename(audio_file), audio, "audio/wav")}
        response = requests.post(f"{base_url}/transcribe", files=files, timeout=300)
    response.raise_for_status()
    return response.json().get("text", "")
