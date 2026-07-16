from unittest.mock import patch

from lumi import s2t


def test_get_default_input_device():
    """The system default input device is returned when no preferred mic exists."""
    with patch("pyaudio.PyAudio") as mock_pyaudio:
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_device_count.return_value = 0
        mock_instance.get_default_input_device_info.return_value = {"index": 1, "name": "Mock Mic"}

        s2t.audio = mock_instance
        assert s2t.get_default_input_device() == 1
        mock_instance.get_default_input_device_info.assert_called_once()


def test_double_tap_starts_recording():
    """Two Option presses within DOUBLE_TAP_TIME start recording."""
    s2t.recording = False
    s2t.last_option_press_time = 0
    with patch.object(s2t, "start_recording") as mock_start:
        s2t.on_press(s2t.keyboard.Key.alt)
        s2t.on_press(s2t.keyboard.Key.alt)
    mock_start.assert_called_once()


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
