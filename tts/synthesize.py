"""Speech synthesis via pyttsx3 (offline) with WAV file output."""

import os
import re
import subprocess
import sys
import tempfile

# Confirmed as a real, reproducible deadlock: an isolated test with two
# fresh, separate pyttsx3.init() calls in one process (no reuse of either
# engine object) hung indefinitely on the second call -- so every real
# init+speak/save sequence below runs in a subprocess instead, which
# always gets a fresh COM apartment regardless of how many times this
# function (or tts.model.load_tts) has already been called in the parent
# process. See tts/model.py's TTSHandle and tts/_synthesize_worker.py for
# the rest of this fix.
SYNTHESIZE_TIMEOUT_SECONDS = 30

# Confirmed as a second, distinct real bug -- not a guess, and not the
# same thing as the deadlock above -- found on a live AWS GPU benchmark
# run: pyttsx3's espeak driver on Linux silently fails to write ANY
# output for text longer than roughly 100-150 characters. No exception,
# exit code 0, empty stdout/stderr -- the worker subprocess just never
# creates the file. Isolated by direct testing on the same machine: pure
# repeated "word " text with zero special punctuation failed identically
# past ~100-150 chars (ruling out shell-escaping -- every common shell
# metacharacter tested individually in short text, ; : & |, worked
# fine), while the same text under ~100 chars always succeeded. This is
# a real content-length limitation in the espeak command-line invocation
# pyttsx3 shells out to, confirmed only on Linux (never seen on
# Windows). _build_context_audio's memory-summary text routinely exceeds
# this after a few real turns/retrieved memories, so this isn't a rare
# edge case for that caller.
_SAFE_CHUNK_CHARS = 100

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


def _chunk_text(text: str, max_len: int = _SAFE_CHUNK_CHARS) -> list[str]:
    """Splits text into pieces under pyttsx3/espeak's confirmed real
    length limit on Linux (see module docstring), breaking on sentence
    boundaries where possible so each chunk still reads naturally once
    synthesized and concatenated back together. Falls back to a
    word-boundary split for the rare single "sentence" that alone
    exceeds max_len, rather than letting an over-length piece through
    whole and risking the exact silent failure this function exists to
    avoid."""
    sentences = re.split(r"(?<=[.!?;])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    def _flush_word_split(piece: str):
        words = piece.split(" ")
        buf = ""
        for word in words:
            candidate = f"{buf} {word}".strip() if buf else word
            if len(candidate) <= max_len:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = word
        return buf

    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sentence) <= max_len:
                current = sentence
            else:
                current = _flush_word_split(sentence)
    if current:
        chunks.append(current)

    return chunks if chunks else [text]  # never return empty for non-empty input


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

    if mode == "file" and not os.path.exists(output_path):
        # Confirmed as a real, silent failure mode, not theoretical: the
        # worker can exit 0 with empty stdout/stderr and never create the
        # file at all (see _SAFE_CHUNK_CHARS's docstring -- this is what
        # that length limit looks like when hit). Failing loudly here, at
        # the actual synthesis boundary, beats the confusing
        # soundfile.LibsndfileError("...System error") that surfaces far
        # downstream (in pipeline_audio2audio.py's _concatenate_audio)
        # with nothing pointing back at the real cause.
        raise RuntimeError(
            f"TTS worker exited successfully (code 0, no stderr) but never "
            f"wrote {output_path!r} -- a real, confirmed pyttsx3/espeak "
            f"failure mode on Linux for text over ~100-150 characters. "
            f"text length was {len(text)}."
        )


def _synthesize_chunk_with_retry(chunk: str, rate, volume, chunk_paths: list, min_len: int = 15) -> None:
    """Synthesizes one chunk, appending its output path to chunk_paths on
    success. Confirmed real, not theoretical: a live benchmark run hit
    the silent-failure bug on a 96-character REAL chunk despite synthetic
    100-character test text succeeding cleanly, meaning the actual limit
    is content-dependent (almost certainly tied to espeak's phonetic
    processing of the specific words involved, not a clean character
    count) -- no fixed _SAFE_CHUNK_CHARS value can be trusted alone for
    arbitrary real content. On failure, splits the chunk in half at the
    nearest word boundary and retries each half recursively, rather than
    trusting any single size threshold to be universally safe. Stops
    splitting at min_len and lets the real RuntimeError surface -- a
    genuinely unsynthesizable few words (not just "still too long") is a
    real failure worth seeing, not something to keep halving forever."""
    chunk_path = tempfile.mktemp(suffix=".wav")
    try:
        _run_worker("file", chunk_path, chunk, rate, volume)
        chunk_paths.append(chunk_path)
    except RuntimeError:
        if len(chunk) <= min_len:
            raise
        mid = len(chunk) // 2
        split_at = chunk.rfind(" ", 0, mid)
        if split_at <= 0:
            split_at = mid
        left, right = chunk[:split_at].strip(), chunk[split_at:].strip()
        if left:
            _synthesize_chunk_with_retry(left, rate, volume, chunk_paths, min_len)
        if right:
            _synthesize_chunk_with_retry(right, rate, volume, chunk_paths, min_len)


def synthesize(engine, text: str, output_path: str, sample_rate: int = 22050) -> str:
    """
    Synthesise text to speech and write a WAV file.

    `engine` is kept as a parameter for backward compatibility with every
    existing call site (and so tests patching this function's signature
    don't need to change) -- rate/volume are read off it, since the
    actual pyttsx3.init()+save_to_file()+runAndWait() sequence now runs in
    a subprocess, not on `engine` directly. See module docstring for why.

    Text over _SAFE_CHUNK_CHARS gets split and synthesized in pieces,
    each in its own isolated subprocess call (confirmed safe -- this is
    exactly the same per-call isolation every synthesize() call already
    uses, just called more than once), then concatenated into one file
    so callers see the same single-file contract regardless of length.

    Confirmed for real that _SAFE_CHUNK_CHARS alone isn't a trustworthy
    hard boundary: a real 96-character chunk from actual pipeline content
    still hit the silent-failure bug despite synthetic 100-character test
    text succeeding cleanly -- the real limit is content-dependent (likely
    tied to espeak's phonetic processing of specific words, not a clean
    character count), not a fixed number this constant can pin exactly.
    So every individual chunk synthesis is wrapped with a retry-by-
    halving fallback (see _synthesize_chunk_with_retry) rather than
    trusting the chunk size alone to be universally safe.
    """
    rate = _get_property(engine, "rate", 175)
    volume = _get_property(engine, "volume", 1.0)

    if len(text) <= _SAFE_CHUNK_CHARS:
        try:
            _run_worker("file", output_path, text, rate, volume)
            return output_path
        except RuntimeError:
            pass  # fall through to the chunk-with-retry path below

    import numpy as np
    import soundfile as sf

    chunk_paths = []
    try:
        for chunk in _chunk_text(text):
            _synthesize_chunk_with_retry(chunk, rate, volume, chunk_paths)

        arrays = []
        sr = None
        for p in chunk_paths:
            data, this_sr = sf.read(p)
            if sr is None:
                sr = this_sr
            arrays.append(data)
        sf.write(output_path, np.concatenate(arrays), sr)
    finally:
        for p in chunk_paths:
            if os.path.exists(p):
                os.remove(p)

    return output_path


def speak(engine, text: str):
    """Play audio directly through speakers (no file). Same subprocess
    isolation as synthesize() -- see module docstring. Long-text
    chunking isn't applied here since this path plays live rather than
    writing a file; nothing in this project currently calls speak() with
    text anywhere near the confirmed length limit."""
    rate = _get_property(engine, "rate", 175)
    volume = _get_property(engine, "volume", 1.0)
    _run_worker("speak", "__unused__", text, rate, volume)
