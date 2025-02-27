"""
Transcription service implementations for Lumi.
"""

import logging
import os
import traceback
from typing import Optional

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

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Groq transcription service.

        Args:
            api_key: Groq API key. If not provided, it will be read from
                the GROQ_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.error("Groq API key not provided")
            raise ValueError(
                "Groq API key not provided. Please set the GROQ_API_KEY environment variable."
            )
        logger.debug("Initializing Groq client")
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
                logger.debug("Sending file to Groq API")
                # TODO: Make the model configurable via CLI or environment variable
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-large-v3",  # Currently hardcoded, could be configurable
                    file=audio,
                )
                
            logger.debug("Transcription successful")
            return transcription.text
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return f"[Transcription error: {str(e)}]"


# Factory function to get the appropriate transcription service
def get_transcription_service(service_name: str) -> TranscriptionService:
    """Get a transcription service by name.

    Args:
        service_name: Name of the transcription service to use.

    Returns:
        A transcription service instance.

    Raises:
        ValueError: If the service name is not recognized.
    """
    logger.debug(f"Getting transcription service: {service_name}")

    if service_name.lower() == "groq":
        logger.debug("Using Groq transcription service")
        return GroqTranscriptionService()
    else:
        logger.error(f"Unknown transcription service: {service_name}")
        raise ValueError(f"Unknown transcription service: {service_name}")
