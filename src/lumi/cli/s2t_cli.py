#!/usr/bin/env python
"""
Command-line interface for running Lumi speech-to-text.
"""

import argparse
import logging
import os
import time
import traceback

from pynput import keyboard

import lumi.s2t
from lumi.s2t import (
    logger,
    on_press,
    on_release,
    recording,
    setup_audio,
    stop_recording,
)


def main():
    """Main entry point for the speech-to-text service."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Lumi Speech-to-Text")
    parser.add_argument("--api-key", help="API key for the transcription service")
    parser.add_argument(
        "--elevenlabs-api-key", help="API key for ElevenLabs (if using ElevenLabs service)"
    )
    parser.add_argument(
        "--service",
        help="Transcription service to use (default: groq, options: groq, elevenlabs, mlx)",
    )
    parser.add_argument("--model", help="Model name for the selected transcription service")
    parser.add_argument(
        "--mlx-model", help="Model name for MLX Whisper (deprecated, use --model instead)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--no-auto-paste", action="store_true", help="Disable automatic pasting of transcription"
    )
    args = parser.parse_args()

    # Set API keys if provided
    if args.api_key:
        os.environ["GROQ_API_KEY"] = args.api_key

    if args.elevenlabs_api_key:
        os.environ["ELEVENLABS_API_KEY"] = args.elevenlabs_api_key

    # Set universal model if provided (takes precedence)
    if args.model:
        if args.service and args.service.lower() == "groq":
            os.environ["GROQ_MODEL"] = args.model
        elif args.service and args.service.lower() == "elevenlabs":
            os.environ["ELEVENLABS_MODEL"] = args.model
        elif args.service and args.service.lower() == "mlx":
            os.environ["MLX_WHISPER_MODEL"] = args.model
        else:
            # Default to setting MLX model if service not specified
            os.environ["MLX_WHISPER_MODEL"] = args.model

    # For backward compatibility (deprecated)
    elif args.mlx_model:
        os.environ["MLX_WHISPER_MODEL"] = args.mlx_model

    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Update the transcription service if provided
    if args.service:
        lumi.s2t.TRANSCRIPTION_SERVICE = args.service

    # Update auto-paste setting if specified
    if args.no_auto_paste:
        lumi.s2t.AUTO_PASTE = False
        logger.info("Auto-pasting disabled")

    keyboard_listener = None

    try:
        # Setup audio
        setup_audio()

        # Start keyboard listener
        keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.start()

        logger.info("Speech-to-text service started.")
        logger.info(f"Using {lumi.s2t.TRANSCRIPTION_SERVICE} transcription service.")
        logger.info("Double-tap the Option key to START recording.")
        logger.info("Single-tap the Option key to STOP recording.")
        if lumi.s2t.AUTO_PASTE:
            logger.info("Auto-paste is enabled - transcriptions will be automatically pasted.")
        logger.info("Press Ctrl+C in this terminal to exit.")

        # Keep the script running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Exiting speech-to-text service...")
        if recording:
            stop_recording()

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")

    finally:
        # Clean up keyboard listener
        if keyboard_listener is not None and keyboard_listener.is_alive():
            keyboard_listener.stop()

        # Clean up resources
        if recording:
            stop_recording()

        # Clean up audio
        if lumi.s2t.audio is not None:
            try:
                lumi.s2t.audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating audio: {e}")
            lumi.s2t.audio = None

        # No additional cleanup needed for sound playing now

        # Final GC to clean up any remaining resources
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
