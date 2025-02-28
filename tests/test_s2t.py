import os

# Import the module for testing
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from lumi import s2t


def test_get_default_input_device():
    """Test that mock default input device is returned correctly."""
    with patch("pyaudio.PyAudio") as mock_pyaudio:
        # Setup the mock
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_default_input_device_info.return_value = {"index": 1}

        # Initialize audio
        s2t.audio = mock_instance

        # Call the function
        result = s2t.get_default_input_device()

        # Check the result
        assert result == 1
        mock_instance.get_default_input_device_info.assert_called_once()


def test_transcribe_audio():
    """Test that transcribe_audio returns text."""
    # For now, this is just testing the placeholder implementation
    result = s2t.transcribe_audio("dummy_file.wav")
    assert isinstance(result, str)
    assert len(result) > 0
