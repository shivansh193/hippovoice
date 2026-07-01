"""Speech synthesis via pyttsx3 (offline) with WAV file output."""

import numpy as np
import soundfile as sf
import tempfile, os


def synthesize(engine, text: str, output_path: str, sample_rate: int = 22050) -> str:
    """
    Synthesise text to speech and write a WAV file.

    pyttsx3 saves directly to file via engine.save_to_file().
    Returns the output_path on success.
    """
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path


def speak(engine, text: str):
    """Play audio directly through speakers (no file)."""
    engine.say(text)
    engine.runAndWait()
