"""Lumi speech-to-text: double-tap the Option key to record, transcribe, and paste."""

import argparse
import logging
import os
import platform
import tempfile
import time
import traceback
import wave

import pyaudio
import pyperclip
from nava import play
from pynput import keyboard
from pynput.keyboard import Controller, Key

from lumi.transcribe import SERVICES, transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
DOUBLE_TAP_TIME = 0.3  # Maximum time between taps to count as double-tap
TEMP_DIR = os.path.join(tempfile.gettempdir(), "speech_to_text")
START_SOUND_FILE = "ding.mp3"  # Located in the same folder as this module
STOP_SOUND_FILE = "dong.mp3"
PREFERRED_MIC = "wireless microphone rx"  # Picked over the system default when present

# Runtime settings (overridden by CLI flags)
SERVICE = "mlx"
MODEL = None
AUTO_PASTE = True

keyboard_controller = Controller()

os.makedirs(TEMP_DIR, exist_ok=True)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# State tracking
recording = False
last_option_press_time = 0
frames = []
stream = None
audio = None


def setup_audio():
    """Set up PyAudio and handle any device changes."""
    global audio
    if audio is not None:
        audio.terminate()
    audio = pyaudio.PyAudio()
    return audio


def get_default_input_device():
    """Get the input device, preferring PREFERRED_MIC if available."""
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and PREFERRED_MIC in info["name"].lower():
            logger.info(f"Found preferred microphone: {info['name']}")
            return i

    try:
        info = audio.get_default_input_device_info()
        logger.info(f"Using system default input device: {info['name']}")
        return info["index"]
    except Exception as e:
        logger.error(f"Error getting default device: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")

        # Last resort: first available input device
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                logger.info(f"Using fallback input device: {info['name']}")
                return i

        raise Exception("No input devices found") from None


def play_sound(is_start=True):
    """Play the start or stop notification sound."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sound_file = START_SOUND_FILE if is_start else STOP_SOUND_FILE
        sound_path = os.path.join(script_dir, sound_file)

        if not os.path.exists(sound_path):
            logger.warning(f"Sound file not found: {sound_path}")
            return

        play(sound_path, async_mode=True, loop=False)
    except Exception as e:
        logger.error(f"Error playing sound: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")


def start_recording():
    """Start recording audio from the microphone."""
    global recording, stream, frames, audio

    if recording:
        logger.info("Already recording, ignoring start request")
        return

    try:
        # Always reinitialize audio to pick up device changes
        setup_audio()

        play_sound(is_start=True)
        time.sleep(0.5)  # Needed, otherwise the start sound does not play

        frames = []

        device_index = get_default_input_device()
        logger.info(f"Using input device index: {device_index}")

        def callback(in_data, frame_count, time_info, status):
            if recording:
                frames.append(in_data)
            return (None, pyaudio.paContinue)

        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            stream_callback=callback,
        )

        stream.start_stream()
        logger.info("Recording started...")
        recording = True

    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        recording = False

        if stream:
            try:
                stream.close()
            except Exception:
                pass
            stream = None

        setup_audio()


def stop_recording():
    """Stop recording, transcribe the audio, and paste the result.

    Recordings are kept in the system temp directory for troubleshooting;
    the OS cleans them up eventually.
    """
    global recording, stream, frames

    if not recording:
        return

    recording = False
    local_frames = list(frames)
    frames = []

    try:
        if stream:
            stream.stop_stream()
            stream.close()
            stream = None

        play_sound(is_start=False)
        logger.info("Recording stopped.")

        if not local_frames:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_file = os.path.join(TEMP_DIR, f"recording_{timestamp}.wav")

        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(local_frames))

        logger.info(f"Recording saved to {temp_file}")

        transcript = transcribe(temp_file, service=SERVICE, model=MODEL)
        if transcript:
            pyperclip.copy(transcript)
            logger.info(f"Transcript copied to clipboard: {transcript}")
            if AUTO_PASTE:
                auto_paste()

    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")

    finally:
        recording = False


def on_press(key):
    """Detect double-tap of the Option key to start, single tap to stop."""
    global last_option_press_time, recording

    if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
        current_time = time.time()
        time_diff = current_time - last_option_press_time

        if recording:
            stop_recording()
            last_option_press_time = 0
            return

        if 0 < time_diff < DOUBLE_TAP_TIME:
            start_recording()
            # Reset timer to prevent a triple-tap from toggling again
            last_option_press_time = 0
        else:
            last_option_press_time = current_time


def auto_paste():
    """Paste the clipboard content with the platform's paste shortcut."""
    try:
        time.sleep(0.5)  # Ensure the clipboard is ready

        modifier = Key.cmd if platform.system() == "Darwin" else Key.ctrl
        keyboard_controller.press(modifier)
        keyboard_controller.press("v")
        keyboard_controller.release("v")
        keyboard_controller.release(modifier)

        logger.info("Auto-paste completed")
    except Exception as e:
        logger.error(f"Error during auto-paste: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")


def main():
    """Entry point for the lumi command."""
    global SERVICE, MODEL, AUTO_PASTE, audio

    parser = argparse.ArgumentParser(description="Lumi Speech-to-Text")
    parser.add_argument("file", nargs="?", help="Audio file to transcribe (optional)")
    parser.add_argument(
        "--service",
        choices=SERVICES,
        default="mlx",
        help="Transcription backend: mlx (local Whisper) or remote (self-hosted server)",
    )
    parser.add_argument("--model", help="MLX Whisper model name")
    parser.add_argument(
        "--no-auto-paste", action="store_true", help="Disable automatic pasting of transcription"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    SERVICE = args.service
    MODEL = args.model
    AUTO_PASTE = not args.no_auto_paste

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # One-off file transcription mode
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return
        transcript = transcribe(args.file, service=SERVICE, model=MODEL)
        print(transcript)
        pyperclip.copy(transcript)
        logger.info("Transcript copied to clipboard")
        return

    keyboard_listener = None
    try:
        setup_audio()

        keyboard_listener = keyboard.Listener(on_press=on_press)
        keyboard_listener.start()

        logger.info("Speech-to-text service started.")
        logger.info(f"Using {SERVICE} transcription service.")
        logger.info("Double-tap the Option key to START recording.")
        logger.info("Single-tap the Option key to STOP recording.")
        if AUTO_PASTE:
            logger.info("Auto-paste is enabled - transcriptions will be automatically pasted.")
        logger.info("Press Ctrl+C in this terminal to exit.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Exiting speech-to-text service...")

    finally:
        if keyboard_listener is not None and keyboard_listener.is_alive():
            keyboard_listener.stop()

        if recording:
            stop_recording()

        if audio is not None:
            try:
                audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating audio: {e}")
            audio = None


if __name__ == "__main__":
    main()
