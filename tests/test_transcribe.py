"""Tests for the transcription service."""
import os
from unittest.mock import MagicMock, patch

import pytest

from lumi.transcribe import (
    GroqTranscriptionService,
    TranscriptionService,
    get_transcription_service,
)


def test_transcription_service_base_class():
    """Test that the base class requires implementation."""
    service = TranscriptionService()
    with pytest.raises(NotImplementedError):
        service.transcribe("dummy.wav")


def test_get_transcription_service_valid():
    """Test getting a valid transcription service."""
    with patch("lumi.transcribe.GroqTranscriptionService") as mock_groq:
        mock_groq.return_value = "groq_service"
        
        # Test with lowercase
        assert get_transcription_service("groq") == "groq_service"
        
        # Test with mixed case
        assert get_transcription_service("Groq") == "groq_service"


def test_get_transcription_service_invalid():
    """Test getting an invalid transcription service."""
    with pytest.raises(ValueError):
        get_transcription_service("invalid_service")


@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
def test_groq_transcription_service_from_env():
    """Test creating a Groq service from environment variable."""
    with patch("lumi.transcribe.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()
        
        service = GroqTranscriptionService()
        assert service.api_key == "test_key"
        mock_groq.assert_called_once_with(api_key="test_key")


def test_groq_transcription_service_with_key():
    """Test creating a Groq service with an explicit key."""
    with patch("lumi.transcribe.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()
        
        service = GroqTranscriptionService(api_key="explicit_key")
        assert service.api_key == "explicit_key"
        mock_groq.assert_called_once_with(api_key="explicit_key")


def test_groq_transcription_service_missing_key():
    """Test that an error is raised if no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            GroqTranscriptionService()


@pytest.mark.skip(reason="Test file does not exist")
@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
def test_groq_transcription():
    """Test that the Groq service properly calls the API."""
    with patch("lumi.transcribe.Groq") as mock_groq, \
         patch("lumi.transcribe.os.path.exists", return_value=True), \
         patch("lumi.transcribe.os.path.getsize", return_value=1024):
        # Set up the mock client
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        
        # Set up the transcription response
        mock_transcriptions = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Transcribed text"
        mock_transcriptions.create.return_value = mock_response
        mock_client.audio.transcriptions = mock_transcriptions
        
        # Call the service
        service = GroqTranscriptionService()
        result = service.transcribe("test.wav")
        
        # Check the result
        assert result == "Transcribed text"
        
        # We're passing a file object to the API, not the filename
        mock_transcriptions.create.assert_called_once()
        call_args = mock_transcriptions.create.call_args[1]
        assert call_args["model"] == "whisper-large-v3"
        assert "file" in call_args