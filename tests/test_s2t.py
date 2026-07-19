import os
from unittest.mock import MagicMock, patch

import numpy as np

from lumi import s2t


def _reset_tap_state():
    s2t.recording = False
    s2t.last_option_press_time = 0
    s2t.last_cmd_press_time = 0


def _run_stop_recording(frames):
    """Drive stop_recording with the given captured frames; returns the transcribe mock."""
    s2t.recording = True
    s2t.frames = frames
    s2t.stream = None
    s2t.audio = MagicMock()
    s2t.audio.get_sample_size.return_value = 2
    with (
        patch.object(s2t, "play_sound"),
        patch.object(s2t, "notify") as mock_notify,
        patch.object(s2t, "transcribe", return_value="hello") as mock_transcribe,
        patch.object(s2t, "auto_paste"),
        patch.object(s2t.pyperclip, "copy"),
    ):
        s2t.stop_recording()
    return mock_transcribe, mock_notify


def test_silent_recording_skips_transcription():
    """An all-zero capture (dead mic stream) is caught: no transcription, user notified."""
    silent = np.zeros(s2t.RATE, dtype=np.int16).tobytes()
    mock_transcribe, mock_notify = _run_stop_recording([silent])
    mock_transcribe.assert_not_called()
    mock_notify.assert_called_once()


def test_loud_recording_is_transcribed():
    """A capture with real signal goes through to transcription."""
    loud = (np.ones(s2t.RATE, dtype=np.int16) * 5000).tobytes()
    mock_transcribe, mock_notify = _run_stop_recording([loud])
    mock_transcribe.assert_called_once()
    mock_notify.assert_not_called()


def test_get_default_input_device():
    """The system default input device is returned when no preferred mic exists."""
    with patch("pyaudio.PyAudio") as mock_pyaudio:
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_device_count.return_value = 0
        mock_instance.get_default_input_device_info.return_value = {"index": 1, "name": "Mock Mic"}

        s2t.audio = mock_instance
        assert s2t.get_default_input_device() == 1
        mock_instance.get_default_input_device_info.assert_called_once()


def test_early_silence_kills_recording():
    """Mid-take check stops an all-zero take, leaves real takes and stopped state alone."""
    silent = np.zeros(s2t.RATE, dtype=np.int16).tobytes()
    loud = (np.ones(s2t.RATE, dtype=np.int16) * 5000).tobytes()

    with patch.object(s2t, "stop_recording") as mock_stop:
        s2t.recording = True
        s2t.frames = [silent]
        s2t._kill_if_silent()
        mock_stop.assert_called_once()

        mock_stop.reset_mock()
        s2t.frames = [loud]
        s2t._kill_if_silent()
        mock_stop.assert_not_called()

        s2t.recording = False
        s2t.frames = [silent]
        s2t._kill_if_silent()
        mock_stop.assert_not_called()


def test_preferred_mic_order():
    """AirPods win over the built-in mic regardless of device index order."""
    devices = [
        {"index": 0, "name": "MacBook Air Microphone", "maxInputChannels": 1},
        {"index": 1, "name": "Some Speakers", "maxInputChannels": 0},
        {"index": 2, "name": "Zeljko's AirPods", "maxInputChannels": 1},
    ]
    s2t.audio = MagicMock()
    s2t.audio.get_device_count.return_value = len(devices)
    s2t.audio.get_device_info_by_index.side_effect = lambda i: devices[i]
    assert s2t.get_default_input_device() == 2


def test_double_tap_starts_recording():
    """Two Option presses within DOUBLE_TAP_TIME start recording."""
    _reset_tap_state()
    with (
        patch.object(s2t.time, "time", side_effect=[100.0, 100.1]),
        patch.object(s2t, "start_recording") as mock_start,
    ):
        s2t.on_press(s2t.keyboard.Key.alt)
        s2t.on_press(s2t.keyboard.Key.alt)
    mock_start.assert_called_once()


def test_double_tap_cmd_resends():
    """Two bare Cmd presses within DOUBLE_TAP_TIME resend the last recording."""
    _reset_tap_state()
    with (
        patch.object(s2t.time, "time", side_effect=[100.0, 100.1]),
        patch.object(s2t, "resend_last_recording") as mock_resend,
    ):
        s2t.on_press(s2t.keyboard.Key.cmd)
        s2t.on_press(s2t.keyboard.Key.cmd)
    mock_resend.assert_called_once()


def test_cmd_chords_do_not_resend():
    """Cmd+key chords (e.g. fast cmd+c cmd+v) never count as a double-tap."""
    _reset_tap_state()
    with (
        patch.object(s2t.time, "time", side_effect=[100.0, 100.2]),
        patch.object(s2t, "resend_last_recording") as mock_resend,
    ):
        s2t.on_press(s2t.keyboard.Key.cmd)
        s2t.on_press(s2t.keyboard.KeyCode.from_char("c"))
        s2t.on_press(s2t.keyboard.Key.cmd)
        s2t.on_press(s2t.keyboard.KeyCode.from_char("v"))
    mock_resend.assert_not_called()


def test_cmd_double_tap_ignored_while_recording():
    """A Cmd double-tap during an active take does nothing."""
    _reset_tap_state()
    s2t.recording = True
    try:
        with patch.object(s2t, "resend_last_recording") as mock_resend:
            s2t.on_press(s2t.keyboard.Key.cmd)
            s2t.on_press(s2t.keyboard.Key.cmd)
        mock_resend.assert_not_called()
    finally:
        s2t.recording = False


def test_resend_picks_latest_recording(tmp_path, monkeypatch):
    """Resend transcribes the newest wav in TEMP_DIR."""
    monkeypatch.setattr(s2t, "TEMP_DIR", str(tmp_path))
    old = tmp_path / "recording_20260719-100000.wav"
    new = tmp_path / "recording_20260719-200000.wav"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    with (
        patch.object(s2t, "play_sound"),
        patch.object(s2t, "transcribe", return_value="hello") as mock_transcribe,
        patch.object(s2t, "auto_paste"),
        patch.object(s2t.pyperclip, "copy"),
    ):
        s2t.resend_last_recording()

    assert mock_transcribe.call_args[0][0] == str(new)


def test_resend_with_no_recordings(tmp_path, monkeypatch):
    """Resend with an empty TEMP_DIR notifies instead of crashing."""
    monkeypatch.setattr(s2t, "TEMP_DIR", str(tmp_path))
    with (
        patch.object(s2t, "notify") as mock_notify,
        patch.object(s2t, "transcribe") as mock_transcribe,
    ):
        s2t.resend_last_recording()
    mock_notify.assert_called_once()
    mock_transcribe.assert_not_called()


def test_single_tap_stops_recording():
    """A single Option press while recording stops it."""
    s2t.recording = True
    s2t.last_option_press_time = 0
    try:
        with patch.object(s2t, "stop_recording") as mock_stop:
            s2t.on_press(s2t.keyboard.Key.alt)
        mock_stop.assert_called_once()
    finally:
        s2t.recording = False
