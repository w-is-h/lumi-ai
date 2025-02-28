"""Tests for the transcription service."""

import os
from unittest.mock import MagicMock, patch

import pytest

from lumi.transcribe import (
    ElevenLabsTranscriptionService,
    GroqTranscriptionService,
    MLXWhisperTranscriptionService,
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
    with (
        patch("lumi.transcribe.GroqTranscriptionService") as mock_groq,
        patch("lumi.transcribe.ElevenLabsTranscriptionService") as mock_elevenlabs,
        patch("lumi.transcribe.MLXWhisperTranscriptionService") as mock_mlx,
    ):
        mock_groq.return_value = "groq_service"
        mock_elevenlabs.return_value = "elevenlabs_service"
        mock_mlx.return_value = "mlx_service"

        # Test Groq with lowercase
        assert get_transcription_service("groq") == "groq_service"

        # Test Groq with mixed case
        assert get_transcription_service("Groq") == "groq_service"

        # Test ElevenLabs with lowercase
        assert get_transcription_service("elevenlabs") == "elevenlabs_service"

        # Test ElevenLabs with mixed case
        assert get_transcription_service("ElevenLabs") == "elevenlabs_service"

        # Test MLX with lowercase
        assert get_transcription_service("mlx") == "mlx_service"

        # Test MLX with mixed case
        assert get_transcription_service("MLX") == "mlx_service"


def test_get_transcription_service_invalid():
    """Test getting an invalid transcription service."""
    with pytest.raises(ValueError):
        get_transcription_service("invalid_service")


def test_get_transcription_service_with_model():
    """Test getting a transcription service with a specific model."""
    with (
        patch("lumi.transcribe.GroqTranscriptionService") as mock_groq,
        patch("lumi.transcribe.ElevenLabsTranscriptionService") as mock_elevenlabs,
        patch("lumi.transcribe.MLXWhisperTranscriptionService") as mock_mlx,
    ):
        # Set up mocks
        mock_groq.return_value = "groq_service"
        mock_elevenlabs.return_value = "elevenlabs_service"
        mock_mlx.return_value = "mlx_service"

        # Test with models
        get_transcription_service("groq", model_name="whisper-tiny")
        mock_groq.assert_called_with(model_name="whisper-tiny")

        get_transcription_service("elevenlabs", model_name="custom_model")
        mock_elevenlabs.assert_called_with(model_name="custom_model")

        get_transcription_service("mlx", model_name="mlx-community/whisper-large-mlx-q4")
        mock_mlx.assert_called_with(model_name="mlx-community/whisper-large-mlx-q4")


@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
def test_groq_transcription_service_from_env():
    """Test creating a Groq service from environment variable."""
    with patch("lumi.transcribe.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()

        service = GroqTranscriptionService()
        assert service.api_key == "test_key"
        assert service.model_name == "whisper-large-v3"  # Default model
        mock_groq.assert_called_once_with(api_key="test_key")


def test_groq_transcription_service_with_key():
    """Test creating a Groq service with an explicit key."""
    with patch("lumi.transcribe.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()

        service = GroqTranscriptionService(api_key="explicit_key")
        assert service.api_key == "explicit_key"
        assert service.model_name == "whisper-large-v3"  # Default model
        mock_groq.assert_called_once_with(api_key="explicit_key")


def test_groq_transcription_service_with_custom_model():
    """Test creating a Groq service with a custom model."""
    with patch("lumi.transcribe.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()

        service = GroqTranscriptionService(api_key="test_key", model_name="whisper-tiny")
        assert service.api_key == "test_key"
        assert service.model_name == "whisper-tiny"  # Custom model
        mock_groq.assert_called_once_with(api_key="test_key")


@patch.dict(os.environ, {"GROQ_API_KEY": "test_key", "GROQ_MODEL": "whisper-medium"})
def test_groq_transcription_service_model_from_env():
    """Test creating a Groq service with model from environment variable."""
    with patch("lumi.transcribe.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()

        service = GroqTranscriptionService()
        assert service.api_key == "test_key"
        assert service.model_name == "whisper-medium"  # From environment
        mock_groq.assert_called_once_with(api_key="test_key")


def test_groq_transcription_service_missing_key():
    """Test that an error is raised if no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            GroqTranscriptionService()


@patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"})
def test_elevenlabs_transcription_service_from_env():
    """Test creating an ElevenLabs service from environment variable."""
    with patch("lumi.transcribe.ElevenLabs") as mock_elevenlabs:
        mock_elevenlabs.return_value = MagicMock()

        service = ElevenLabsTranscriptionService()
        assert service.api_key == "test_key"
        assert service.model_name == "scribe_v1"  # Default model
        mock_elevenlabs.assert_called_once_with(api_key="test_key")


def test_elevenlabs_transcription_service_with_key():
    """Test creating an ElevenLabs service with an explicit key."""
    with patch("lumi.transcribe.ElevenLabs") as mock_elevenlabs:
        mock_elevenlabs.return_value = MagicMock()

        service = ElevenLabsTranscriptionService(api_key="explicit_key")
        assert service.api_key == "explicit_key"
        assert service.model_name == "scribe_v1"  # Default model
        mock_elevenlabs.assert_called_once_with(api_key="explicit_key")


def test_elevenlabs_transcription_service_with_custom_model():
    """Test creating an ElevenLabs service with a custom model."""
    with patch("lumi.transcribe.ElevenLabs") as mock_elevenlabs:
        mock_elevenlabs.return_value = MagicMock()

        service = ElevenLabsTranscriptionService(api_key="test_key", model_name="custom_model")
        assert service.api_key == "test_key"
        assert service.model_name == "custom_model"  # Custom model
        mock_elevenlabs.assert_called_once_with(api_key="test_key")


@patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key", "ELEVENLABS_MODEL": "alt_model"})
def test_elevenlabs_transcription_service_model_from_env():
    """Test creating an ElevenLabs service with model from environment variable."""
    with patch("lumi.transcribe.ElevenLabs") as mock_elevenlabs:
        mock_elevenlabs.return_value = MagicMock()

        service = ElevenLabsTranscriptionService()
        assert service.api_key == "test_key"
        assert service.model_name == "alt_model"  # From environment
        mock_elevenlabs.assert_called_once_with(api_key="test_key")


def test_elevenlabs_transcription_service_missing_key():
    """Test that an error is raised if no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            ElevenLabsTranscriptionService()


@pytest.mark.skip(reason="Test file does not exist")
@patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"})
def test_elevenlabs_transcription():
    """Test that the ElevenLabs service properly calls the API."""
    with (
        patch("lumi.transcribe.ElevenLabs") as mock_elevenlabs,
        patch("lumi.transcribe.os.path.exists", return_value=True),
        patch("lumi.transcribe.os.path.getsize", return_value=1024),
    ):
        # Set up the mock client
        mock_client = MagicMock()
        mock_elevenlabs.return_value = mock_client

        # Set up the transcription response
        mock_stt = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Transcribed text from ElevenLabs"
        mock_stt.convert.return_value = mock_response
        mock_client.speech_to_text = mock_stt

        # Call the service
        service = ElevenLabsTranscriptionService()
        result = service.transcribe("test.wav")

        # Check the result
        assert result == "Transcribed text from ElevenLabs"

        # We're passing a file object to the API, not the filename
        mock_stt.convert.assert_called_once()
        call_args = mock_stt.convert.call_args[1]
        assert call_args["model_id"] == "scribe_v1"
        assert "file" in call_args


@pytest.mark.skip(reason="Test file does not exist")
@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
def test_groq_transcription():
    """Test that the Groq service properly calls the API."""
    with (
        patch("lumi.transcribe.Groq") as mock_groq,
        patch("lumi.transcribe.os.path.exists", return_value=True),
        patch("lumi.transcribe.os.path.getsize", return_value=1024),
    ):
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


def test_mlx_whisper_transcription_service_default_model():
    """Test creating an MLX Whisper service with the default model."""
    with patch.dict(os.environ, {}, clear=True):
        service = MLXWhisperTranscriptionService()
        assert service.model_name == "mlx-community/whisper-medium-mlx-q4"


@patch.dict(os.environ, {"MLX_WHISPER_MODEL": "mlx-community/whisper-large-mlx-q4"})
def test_mlx_whisper_transcription_service_from_env():
    """Test creating an MLX Whisper service with model from environment variable."""
    service = MLXWhisperTranscriptionService()
    assert service.model_name == "mlx-community/whisper-large-mlx-q4"


def test_mlx_whisper_transcription_service_with_model():
    """Test creating an MLX Whisper service with an explicit model."""
    service = MLXWhisperTranscriptionService(model_name="mlx-community/whisper-tiny-mlx-q4")
    assert service.model_name == "mlx-community/whisper-tiny-mlx-q4"


@pytest.mark.skip(reason="Test file does not exist")
def test_mlx_whisper_transcription():
    """Test that the MLX Whisper service properly transcribes audio."""
    with (
        patch("lumi.transcribe.mlx_whisper.transcribe") as mock_transcribe,
        patch("lumi.transcribe.os.path.exists", return_value=True),
        patch("lumi.transcribe.os.path.getsize", return_value=1024),
    ):
        # Set up the mock response
        mock_transcribe.return_value = {"text": "Transcribed text from MLX Whisper"}

        # Call the service
        service = MLXWhisperTranscriptionService(model_name="mlx-community/whisper-tiny-mlx-q4")
        result = service.transcribe("test.wav")

        # Check the result
        assert result == "Transcribed text from MLX Whisper"

        # Verify the correct parameters were passed
        model_name = "mlx-community/whisper-tiny-mlx-q4"
        mock_transcribe.assert_called_once_with("test.wav", path_or_hf_repo=model_name)
