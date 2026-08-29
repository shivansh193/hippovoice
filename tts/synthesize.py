"""Speech synthesis via pyttsx3 (offline) with WAV file output."""

import os
import subprocess
import sys

# Confirmed as a real, reproducible deadlock: an isolated test with two
# fresh, separate pyttsx3.init() calls in one process (no reuse of either
# engine object) hung indefinitely on the second call -- so every real
# init+speak/save sequence below runs in a brand-new OS process instead,
# which always gets a fresh COM apartment regardless of how many times
# this function (or tts.model.load_tts) has already been called in the
# parent process. See tts/model.py's TTSHandle and
# tts/_synthesize_worker.py for the rest of this fix.
SYNTHESIZE_TIMEOUT_SECONDS = 30

_WORKER_PATH = os.path.join(os.path.dirname(__file__), "_synthesize_worker.py")


def _get_property(engine, name: str, default):
    """`engine` is whatever tts.model.load_tts() returned -- a TTSHandle
    in real usage, but tests patch load_tts to return a MagicMock, and
    some call sites may pass None. Never let a rate/volume lookup on any
    of those raise and take down synthesis over a cosmetic setting; a
    Mock's default (a Mock object, not a number) is rejected the same as
    a missing attribute."""
    try:
        value = engine.getProperty(name)
        float(value)
        return value
    except Exception:
        return default


def _run_worker(mode: str, output_path: str, text: str, rate, volume) -> None:
    try:
        result = subprocess.run(
            [sys.executable, _WORKER_PATH, mode, output_path, str(rate), str(volume)],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=SYNTHESIZE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"TTS subprocess did not finish within {SYNTHESIZE_TIMEOUT_SECONDS}s -- "
            f"see tts/_synthesize_worker.py for what it runs."
        )
    if result.returncode != 0:
        raise RuntimeError(f"TTS subprocess failed (exit {result.returncode}): {result.stderr}")


def synthesize(engine, text: str, output_path: str, sample_rate: int = 22050) -> str:
    """
    Synthesise text to speech and write a WAV file.

    `engine` is kept as a parameter for backward compatibility with every
    existing call site (and so tests patching this function's signature
    don't need to change) -- rate/volume are read off it, since the
    actual pyttsx3.init()+save_to_file()+runAndWait() sequence now runs in
    a subprocess, not on `engine` directly. See module docstring for why.
    """
    rate = _get_property(engine, "rate", 175)
    volume = _get_property(engine, "volume", 1.0)
    _run_worker("file", output_path, text, rate, volume)
    return output_path


def speak(engine, text: str):
    """Play audio directly through speakers (no file). Same subprocess
    isolation as synthesize() -- see module docstring."""
    rate = _get_property(engine, "rate", 175)
    volume = _get_property(engine, "volume", 1.0)
    _run_worker("speak", "__unused__", text, rate, volume)
