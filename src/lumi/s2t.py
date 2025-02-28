import logging
import os
import platform
import tempfile
import time
import traceback
import wave
import time

import pyaudio
import pyperclip
from nava import play
from pynput import keyboard
from pynput.keyboard import Controller, Key

from lumi.transcribe import get_transcription_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
DOUBLE_TAP_TIME = 0.3  # Maximum time between taps to count as double-tap
TEMP_DIR = os.path.join(tempfile.gettempdir(), "speech_to_text")
START_SOUND_FILE = "ding.mp3"  # Located in the same folder as the script
STOP_SOUND_FILE = "dong.mp3"  # Located in the same folder as the script
TRANSCRIPTION_SERVICE = "groq"  # Transcription service to use (options: "groq", "elevenlabs")

# State variables
SINGLE_TAP_STOPS = True  # Allow single tap to stop recording
AUTO_PASTE = True  # Automatically paste the transcription

# Initialize keyboard controller for pasting
keyboard_controller = Controller()

# Create temp directory if it doesn't exist
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

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

# Initialize PyAudio
audio = None


def setup_audio():
    """Set up PyAudio and handle any device changes."""
    global audio
    if audio is not None:
        audio.terminate()
    audio = pyaudio.PyAudio()
    return audio


def get_default_input_device():
    """Get the default input device, with fallback if not available."""
    try:
        info = audio.get_default_input_device_info()
        return info["index"]
    except Exception as e:
        logger.error(f"Error getting default device: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        # Fallback: find first available input device
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                logger.info(f"Using fallback input device: {info['name']}")
                return i
        err = Exception("No input devices found")
        raise err from None


def play_sound(is_start=True):
    """Play the notification sound using nava.

    Args:
        is_start: If True, play the start sound, otherwise play the stop sound.
    """
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sound_file = START_SOUND_FILE if is_start else STOP_SOUND_FILE
        sound_path = os.path.join(script_dir, sound_file)

        if not os.path.exists(sound_path):
            logger.warning(f"Sound file not found: {sound_path}")
            return

        # Play the sound asynchronously so it doesn't block
        play(sound_path, async_mode=True, loop=False)
    except Exception as e:
        logger.error(f"Error playing sound: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")


def start_recording():
    """Start recording audio from the microphone."""
    global recording, stream, frames, audio

    # Don't start if already recording
    if recording:
        logger.info("Already recording, ignoring start request")
        return

    try:
        # Always reinitialize audio to ensure we have the latest system state
        setup_audio()

        # Play start sound
        play_sound(is_start=True)
        time.sleep(0.4) # Needed, otherwise the start sound does not play

        # Reset frames
        frames = []

        # Get current default input device
        device_index = get_default_input_device()
        logger.info(f"Using input device index: {device_index}")

        # Open stream with callback function to collect frames
        def callback(in_data, frame_count, time_info, status):
            if recording:
                frames.append(in_data)
            return (None, pyaudio.paContinue)

        # Open stream with callback
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

        # Clean up resources if there was an error
        if stream:
            try:
                stream.close()
            except Exception:
                pass
            stream = None

        # Try to reconnect audio if there was an error
        setup_audio()


def stop_recording():
    """Stop recording and process the audio.

    Note: Temporary audio files are stored in the system temp directory.
    These files are not automatically deleted to allow for
    troubleshooting, but the OS will typically clean them up eventually.
    """
    global recording, stream, frames

    if not recording:
        return

    # Mark as not recording first
    recording = False

    # Make a copy of frames
    local_frames = list(frames)
    frames = []

    try:
        # Stop and close the stream
        if stream:
            stream.stop_stream()
            stream.close()
            stream = None

        # Play stop sound
        play_sound(is_start=False)

        logger.info("Recording stopped.")

        # Only process if we have frames
        if local_frames:
            # Save audio file
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            temp_file = os.path.join(TEMP_DIR, f"recording_{timestamp}.wav")

            try:
                # Write file
                with wave.open(temp_file, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b"".join(local_frames))

                logger.info(f"Recording saved to {temp_file}")

                # Process the audio file
                transcript = transcribe_audio(temp_file)
                if transcript:
                    pyperclip.copy(transcript)
                    logger.info(f"Transcript copied to clipboard: {transcript}")

                    # Auto-paste if enabled
                    if AUTO_PASTE:
                        auto_paste()
            except Exception as e:
                logger.error(f"Error processing audio file: {e}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")

    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")

    finally:
        # Make sure recording is set to False
        recording = False


def transcribe_audio(audio_file):
    """Send audio to speech-to-text API and return transcript."""
    logger.info(f"Sending {audio_file} to {TRANSCRIPTION_SERVICE} transcription service...")

    try:
        # Get the appropriate transcription service
        transcription_service = get_transcription_service(TRANSCRIPTION_SERVICE)

        # Transcribe the audio file
        transcript = transcription_service.transcribe(audio_file)

        return transcript
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        return f"[Transcription error: {str(e)}]"


def on_press(key):
    """Handle key press events to detect double-tap of option key."""
    global last_option_press_time, recording

    # Check if option key was pressed
    if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
        current_time = time.time()
        time_diff = current_time - last_option_press_time

        # Stop recording with a single tap if enabled and already recording
        if SINGLE_TAP_STOPS and recording:
            stop_recording()
            # Reset timer
            last_option_press_time = 0
            return

        # Check if this is a double-tap for starting recording
        if 0 < time_diff < DOUBLE_TAP_TIME and not recording:
            # Start recording
            start_recording()
            # Reset timer to prevent triple-tap from toggling again
            last_option_press_time = 0
        else:
            # Update the last press time
            last_option_press_time = current_time


def on_release(key):
    """Handle key release events."""
    pass  # We don't need special handling for key releases


def auto_paste():
    """Automatically paste the current clipboard content using keyboard shortcut."""
    try:
        # Small delay to ensure the clipboard is ready
        time.sleep(0.5)

        # Determine the correct paste shortcut based on OS
        if platform.system() == "Darwin":  # macOS
            logger.debug("Using macOS paste shortcut (Command+V)")
            keyboard_controller.press(Key.cmd)
            keyboard_controller.press("v")
            keyboard_controller.release("v")
            keyboard_controller.release(Key.cmd)
        else:  # Windows/Linux
            logger.debug("Using Windows/Linux paste shortcut (Control+V)")
            keyboard_controller.press(Key.ctrl)
            keyboard_controller.press("v")
            keyboard_controller.release("v")
            keyboard_controller.release(Key.ctrl)

        logger.info("Auto-paste completed")
    except Exception as e:
        logger.error(f"Error during auto-paste: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")


# For backward compatibility
if __name__ == "__main__":
    from lumi.cli.s2t_cli import main

    main()
