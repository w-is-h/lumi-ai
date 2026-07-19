"""Lumi speech-to-text: double-tap the Option key to record, transcribe, and paste."""

import argparse
import glob
import logging
import os
import platform
import subprocess
import tempfile
import threading
import time
import traceback
import wave

import numpy as np
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
PREFERRED_MICS = ("airpods", "macbook")  # Name substrings, first match wins; then system default
SILENT_PEAK = 100  # int16 peak below this means the mic delivered no usable signal
# (a dead capture stream — e.g. idle AirPods over Bluetooth — yields exact zeros;
# real speech peaks in the thousands)
EARLY_SILENCE_CHECK_S = 3  # Kill the take this many seconds in if the mic only delivered zeros

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

OPTION_KEYS = (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
CMD_KEYS = (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)

# State tracking
recording = False
last_option_press_time = 0
last_cmd_press_time = 0
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
    """Get the input device, walking PREFERRED_MICS in order before the system default."""
    inputs = [
        (i, audio.get_device_info_by_index(i)["name"])
        for i in range(audio.get_device_count())
        if audio.get_device_info_by_index(i)["maxInputChannels"] > 0
    ]
    for preferred in PREFERRED_MICS:
        for i, name in inputs:
            if preferred in name.lower():
                logger.info(f"Found preferred microphone: {name}")
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


def notify(message: str) -> None:
    """Show a desktop notification (macOS); elsewhere the log line has to do.

    Banners post under Script Editor's identity — it must be allowed under
    System Settings > Notifications.
    """
    if platform.system() != "Darwin":
        return
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "lumi"'],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        logger.debug(f"Notification failed: {e}")


def _kill_if_silent():
    """Timer callback EARLY_SILENCE_CHECK_S into a take: kill it if only zeros so far.

    stop_recording's own silence guard then handles the user-facing part
    (notification, kept wav, no transcription).
    """
    local_frames = list(frames)
    if not recording or not local_frames:
        return
    peak = int(np.abs(np.frombuffer(b"".join(local_frames), dtype=np.int16)).max())
    if peak < SILENT_PEAK:
        logger.error(
            f"Mic delivered only silence {EARLY_SILENCE_CHECK_S}s in — stopping the recording"
        )
        stop_recording()


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

        # Set before the stream starts so the callback keeps the first buffers
        recording = True

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
        threading.Timer(EARLY_SILENCE_CHECK_S, _kill_if_silent).start()
        logger.info("Recording started...")

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

        raw = b"".join(local_frames)
        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(raw)

        logger.info(f"Recording saved to {temp_file}")

        peak = int(np.abs(np.frombuffer(raw, dtype=np.int16)).max())
        if peak < SILENT_PEAK:
            logger.error(
                f"Recording is silent (peak {peak}) — the mic delivered no signal. "
                f"Check the input device (idle Bluetooth mics record zeros). Kept {temp_file}"
            )
            notify("Recording was silent — the mic delivered nothing. Check input device.")
            return

        transcribe_and_paste(temp_file)

    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")

    finally:
        recording = False


def transcribe_and_paste(audio_file):
    """Transcribe a wav and deliver the text: clipboard, then paste."""
    transcript = transcribe(audio_file, service=SERVICE, model=MODEL)
    if transcript:
        pyperclip.copy(transcript)
        logger.info(f"Transcript copied to clipboard: {transcript}")
        if AUTO_PASTE:
            auto_paste()
    else:
        logger.warning("Transcription came back empty — nothing to paste")
        notify("Transcription came back empty.")


def resend_last_recording():
    """Re-transcribe the newest saved recording and paste the result."""
    wavs = glob.glob(os.path.join(TEMP_DIR, "recording_*.wav"))
    if not wavs:
        logger.warning("No saved recordings to resend")
        notify("No saved recordings to resend.")
        return

    latest = max(wavs, key=os.path.getmtime)
    logger.info(f"Resending {latest}")
    play_sound(is_start=True)
    transcribe_and_paste(latest)


def on_press(key):
    """Double-tap Option starts a recording (single tap stops it);
    double-tap Cmd resends the last one. Bare taps only — any other key
    in between means the modifier was part of a chord, not a tap."""
    global last_option_press_time, last_cmd_press_time, recording

    if key in OPTION_KEYS:
        current_time = time.time()
        time_diff = current_time - last_option_press_time
        last_cmd_press_time = 0

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

    elif key in CMD_KEYS:
        current_time = time.time()
        time_diff = current_time - last_cmd_press_time
        last_option_press_time = 0

        if recording:
            return

        if 0 < time_diff < DOUBLE_TAP_TIME:
            last_cmd_press_time = 0
            resend_last_recording()
        else:
            last_cmd_press_time = current_time

    else:
        last_option_press_time = 0
        last_cmd_press_time = 0


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
