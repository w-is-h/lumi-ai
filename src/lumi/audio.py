"""Audio preprocessing: silence gating and chunking for long recordings.

Pure numpy on 16-bit mono PCM. Two jobs, both driven by per-frame RMS energy:

- gate_silences: collapse long silences so dead air isn't transcribed.
- split_at_silences: cut audio into chunks the remote server can take in one
  request (its models see at most ~30s per forward pass), cutting at the
  quietest point near each boundary so words stay intact.
"""

import wave

import numpy as np

FRAME_MS = 30  # energy analysis frame

# Gating: silences longer than MAX_SILENCE_S are shortened to KEPT_SILENCE_S.
# Short pauses are kept — they carry phrasing/punctuation cues.
MAX_SILENCE_S = 0.8
KEPT_SILENCE_S = 0.3

# A frame is silent if its RMS is below this fraction of the recording's loud
# level (95th percentile frame RMS), with an absolute floor so an all-quiet
# recording isn't gated to nothing.
SILENCE_REL_THRESHOLD = 0.05
SILENCE_ABS_FLOOR = 100.0  # int16 RMS units

MAX_CHUNK_S = 28.0  # stay under the server's 30s window


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit WAV into an int16 array. Returns (samples, rate)."""
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit WAV")
        rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return samples, rate


def save_wav(path: str, samples: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _frame_rms(samples: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    """Per-frame RMS energy. Returns (rms array, frame length in samples)."""
    frame_len = max(1, int(rate * FRAME_MS / 1000))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.array([]), frame_len
    frames = samples[: n_frames * frame_len].astype(np.float64).reshape(n_frames, frame_len)
    return np.sqrt((frames**2).mean(axis=1)), frame_len


def _silent_frames(rms: np.ndarray) -> np.ndarray:
    """Boolean mask of silent frames."""
    if len(rms) == 0:
        return np.array([], dtype=bool)
    loud = np.percentile(rms, 95)
    threshold = max(loud * SILENCE_REL_THRESHOLD, SILENCE_ABS_FLOOR)
    return rms < threshold


def gate_silences(samples: np.ndarray, rate: int) -> np.ndarray:
    """Shorten every silence run longer than MAX_SILENCE_S to KEPT_SILENCE_S."""
    rms, frame_len = _frame_rms(samples, rate)
    silent = _silent_frames(rms)
    if not silent.any():
        return samples

    max_run = max(1, int(MAX_SILENCE_S * 1000 / FRAME_MS))
    keep_run = max(1, int(KEPT_SILENCE_S * 1000 / FRAME_MS))

    keep = np.ones(len(silent), dtype=bool)
    run_start = None
    for i, s in enumerate([*silent, False]):  # sentinel closes a trailing run
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            run_len = i - run_start
            if run_len > max_run:
                keep[run_start + keep_run : i] = False
            run_start = None

    sample_mask = np.repeat(keep, frame_len)
    tail = samples[len(sample_mask) :]  # partial frame at the end, always kept
    return np.concatenate([samples[: len(sample_mask)][sample_mask], tail])


def split_at_silences(samples: np.ndarray, rate: int) -> list[np.ndarray]:
    """Split into chunks of at most MAX_CHUNK_S, cutting at the quietest frame
    in the trailing 40% of each window so cuts land in pauses, not words."""
    max_samples = int(MAX_CHUNK_S * rate)
    if len(samples) <= max_samples:
        return [samples]

    rms, frame_len = _frame_rms(samples, rate)
    chunks = []
    start = 0
    while len(samples) - start > max_samples:
        window_lo = start + int(max_samples * 0.6)
        window_hi = start + max_samples
        lo_f, hi_f = window_lo // frame_len, window_hi // frame_len
        cut_f = lo_f + int(np.argmin(rms[lo_f:hi_f])) if hi_f > lo_f else hi_f
        cut = cut_f * frame_len
        chunks.append(samples[start:cut])
        start = cut
    chunks.append(samples[start:])
    return chunks
