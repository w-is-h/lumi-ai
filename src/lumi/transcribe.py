"""
Transcription service implementations for Lumi.
"""

import logging
import os
import traceback
from typing import Optional

import mlx_whisper
from elevenlabs.client import ElevenLabs
from groq import Groq

# Get logger
logger = logging.getLogger(__name__)


class TranscriptionService:
    """Base class for transcription services."""

    def transcribe(self, audio_file: str) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_file: Path to the audio file to transcribe.

        Returns:
            Transcribed text.
        """
        raise NotImplementedError("Subclasses must implement this method")


class GroqTranscriptionService(TranscriptionService):
    """Transcription service using Groq API."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize the Groq transcription service.

        Args:
            api_key: Groq API key. If not provided, it will be read from
                the GROQ_API_KEY environment variable.
            model_name: Name of the Whisper model to use. If not provided, it will be read from
                the GROQ_MODEL environment variable or use a default model.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.error("Groq API key not provided")
            raise ValueError(
                "Groq API key not provided. Please set the GROQ_API_KEY environment variable."
            )

        # Set model name with priority: parameter > environment variable > default
        self.model_name = model_name or os.environ.get("GROQ_MODEL", "whisper-large-v3")
        logger.debug(f"Initializing Groq client with model: {self.model_name}")
        self.client = Groq(api_key=self.api_key)

    def transcribe(self, audio_file: str) -> str:
        """Transcribe an audio file using Groq's Whisper API.

        Args:
            audio_file: Path to the audio file to transcribe.

        Returns:
            Transcribed text.
        """
        logger.debug(f"Transcribing audio file: {audio_file}")

        try:
            # Simple check if file exists
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return "[Transcription error: File not found]"

            # Simple check if file is empty
            if os.path.getsize(audio_file) == 0:
                logger.error("Audio file is empty")
                return "[Transcription error: File is empty]"

            # Open file and send to API
            with open(audio_file, "rb") as audio:
                logger.debug(f"Sending file to Groq API with model: {self.model_name}")
                transcription = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio,
                )

            logger.debug("Transcription successful")
            return transcription.text

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return f"[Transcription error: {str(e)}]"


class ElevenLabsTranscriptionService(TranscriptionService):
    """Transcription service using ElevenLabs API."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize the ElevenLabs transcription service.

        Args:
            api_key: ElevenLabs API key. If not provided, it will be read from
                the ELEVENLABS_API_KEY environment variable.
            model_name: Name of the ElevenLabs model to use. If not provided, it will be read from
                the ELEVENLABS_MODEL environment variable or use a default model.
        """
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error("ElevenLabs API key not provided")
            raise ValueError(
                "ElevenLabs API key not provided. "
                "Please set the ELEVENLABS_API_KEY environment variable."
            )

        # Set model name with priority: parameter > environment variable > default
        self.model_name = model_name or os.environ.get("ELEVENLABS_MODEL", "scribe_v1")
        logger.debug(f"Initializing ElevenLabs client with model: {self.model_name}")
        self.client = ElevenLabs(api_key=self.api_key)

    def transcribe(self, audio_file: str) -> str:
        """Transcribe an audio file using ElevenLabs API.

        Args:
            audio_file: Path to the audio file to transcribe.

        Returns:
            Transcribed text.
        """
        logger.debug(f"Transcribing audio file with ElevenLabs: {audio_file}")

        try:
            # Simple check if file exists
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return "[Transcription error: File not found]"

            # Simple check if file is empty
            if os.path.getsize(audio_file) == 0:
                logger.error("Audio file is empty")
                return "[Transcription error: File is empty]"

            # Open file and send to API
            with open(audio_file, "rb") as audio:
                logger.debug("Sending file to ElevenLabs API")
                transcription = self.client.speech_to_text.convert(
                    file=audio,
                    model_id=self.model_name,
                    tag_audio_events=False,
                    diarize=False,
                )

            logger.debug("ElevenLabs transcription successful")
            return transcription.text

        except Exception as e:
            logger.error(f"Error during ElevenLabs transcription: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return f"[Transcription error: {str(e)}]"


# MLX Whisper transcription service
class MLXWhisperTranscriptionService(TranscriptionService):
    """Transcription service using local MLX Whisper model."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the MLX Whisper transcription service.

        Args:
            model_name: Name of the Whisper model to use. If not provided, it will be read from
                the MLX_WHISPER_MODEL environment variable or use a default model.
        """
        default_model = "mlx-community/whisper-medium-mlx-q4"
        self.model_name = model_name or os.environ.get("MLX_WHISPER_MODEL", default_model)
        logger.debug(f"Initializing MLX Whisper with model: {self.model_name}")

    def transcribe(self, audio_file: str) -> str:
        """Transcribe an audio file using local MLX Whisper model.

        Args:
            audio_file: Path to the audio file to transcribe.

        Returns:
            Transcribed text.
        """
        logger.debug(f"Transcribing audio file with MLX Whisper: {audio_file}")

        try:
            # Simple check if file exists
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return "[Transcription error: File not found]"

            # Simple check if file is empty
            if os.path.getsize(audio_file) == 0:
                logger.error("Audio file is empty")
                return "[Transcription error: File is empty]"

            # Transcribe using MLX Whisper
            logger.debug(f"Transcribing with MLX Whisper model: {self.model_name}")
            result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=self.model_name)
            transcript = result["text"]

            logger.debug("MLX Whisper transcription successful")
            return transcript

        except Exception as e:
            logger.error(f"Error during MLX Whisper transcription: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return f"[Transcription error: {str(e)}]"


# Factory function to get the appropriate transcription service
def get_transcription_service(
    service_name: str, model_name: Optional[str] = None
) -> TranscriptionService:
    """Get a transcription service by name.

    Args:
        service_name: Name of the transcription service to use.
        model_name: Optional model name to use with the service. If provided,
            it overrides any environment variable or default setting.

    Returns:
        A transcription service instance.

    Raises:
        ValueError: If the service name is not recognized.
    """
    model_str = model_name or "default"
    logger.debug(f"Getting transcription service: {service_name} with model: {model_str}")

    if service_name.lower() == "groq":
        logger.debug("Using Groq transcription service")
        return GroqTranscriptionService(model_name=model_name)
    elif service_name.lower() == "elevenlabs":
        logger.debug("Using ElevenLabs transcription service")
        return ElevenLabsTranscriptionService(model_name=model_name)
    elif service_name.lower() == "mlx":
        logger.debug("Using MLX Whisper transcription service")
        return MLXWhisperTranscriptionService(model_name=model_name)
    else:
        logger.error(f"Unknown transcription service: {service_name}")
        raise ValueError(f"Unknown transcription service: {service_name}")
