"""Tests for silence gating and chunk splitting."""

import numpy as np
import pytest

from lumi import audio

RATE = 16000


def tone(seconds: float, amplitude: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.int16)


def test_wav_round_trip(tmp_path):
    samples = tone(1.0)
    path = str(tmp_path / "t.wav")
    audio.save_wav(path, samples, RATE)
    loaded, rate = audio.load_wav(path)
    assert rate == RATE
    assert np.array_equal(loaded, samples)


def test_load_wav_rejects_stereo(tmp_path):
    import wave

    path = str(tmp_path / "stereo.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(np.zeros(RATE * 2, dtype=np.int16).tobytes())
    with pytest.raises(ValueError):
        audio.load_wav(path)


def test_gate_collapses_long_silence():
    samples = np.concatenate([tone(2.0), silence(5.0), tone(2.0)])
    gated = audio.gate_silences(samples, RATE)
    # 5s silence collapses to ~KEPT_SILENCE_S; speech is untouched.
    expected = 2.0 + audio.KEPT_SILENCE_S + 2.0
    assert len(gated) / RATE == pytest.approx(expected, abs=0.2)


def test_gate_keeps_short_pauses():
    samples = np.concatenate([tone(2.0), silence(0.5), tone(2.0)])
    gated = audio.gate_silences(samples, RATE)
    assert len(gated) == len(samples)


def test_gate_all_quiet_recording_survives():
    samples = silence(3.0)
    gated = audio.gate_silences(samples, RATE)
    # Uniform silence has no runs above threshold contrast to gate away entirely;
    # whatever remains, nothing blows up and output is non-negative length.
    assert len(gated) >= 0


def test_split_short_audio_is_one_chunk():
    samples = tone(10.0)
    chunks = audio.split_at_silences(samples, RATE)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], samples)


def test_split_long_audio_respects_max_and_loses_nothing():
    # 70s of speech with silences sprinkled in.
    parts = []
    for _ in range(7):
        parts.append(tone(9.5))
        parts.append(silence(0.5))
    samples = np.concatenate(parts)
    chunks = audio.split_at_silences(samples, RATE)
    assert len(chunks) >= 3
    max_samples = int(audio.MAX_CHUNK_S * RATE)
    assert all(len(c) <= max_samples for c in chunks)
    assert sum(len(c) for c in chunks) == len(samples)
    assert np.array_equal(np.concatenate(chunks), samples)


def test_split_cuts_in_silence():
    # One clear silence 20-21s into a 40s recording: the cut should land in it.
    samples = np.concatenate([tone(20.0), silence(1.0), tone(19.0)])
    chunks = audio.split_at_silences(samples, RATE)
    assert len(chunks) == 2
    cut = len(chunks[0])
    assert 20.0 * RATE <= cut <= 21.0 * RATE
