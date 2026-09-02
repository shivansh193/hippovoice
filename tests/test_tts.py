"""
Tests for the pyttsx3 SAPI5 deadlock fix (tts/model.py + tts/synthesize.py
+ tts/_synthesize_worker.py).

Confirmed as a real, reproducible hang, not a guess: an isolated script
with nothing else involved -- two fresh, separate pyttsx3.init() calls in
a row, no reuse of either engine object -- hung indefinitely on the
second call (near-zero CPU, no exception, had to be force-killed). This
was a worse case than the "reusing one engine instance" deadlock this
project had already documented and worked around elsewhere (see
pipeline_audio2audio.py's _build_context_audio, which already built a
fresh engine per call and still hung on a real run) -- meaning
tts.model.load_tts() itself, not just tts.synthesize.synthesize(), was
the actual trigger every call site was hitting repeatedly.

test_synthesize_can_be_called_repeatedly_without_hanging is the direct
regression test for that: it calls the real (subprocess-isolated) path
three times in one process, which would have hung on call 2 before this
fix, using real pyttsx3 -- no GPU, no network, just real wall-clock time
(a few seconds per call for subprocess + engine startup).
"""
import os
import time

import numpy as np
import pytest
import soundfile as sf

from tts.model import TTSHandle, load_tts
from tts.synthesize import synthesize


def test_load_tts_returns_a_handle_without_touching_pyttsx3(monkeypatch, tmp_path):
    """The actual root cause: load_tts() itself used to call
    pyttsx3.init(), so calling it more than once per process was already
    unsafe before synthesize() was ever involved. Confirms load_tts() no
    longer imports/calls pyttsx3 at all -- makes pyttsx3.init raise, and
    load_tts() must still succeed."""
    import pyttsx3
    monkeypatch.setattr(pyttsx3, "init", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("load_tts() must not call pyttsx3.init()")
    ))

    handle = load_tts(rate=200, volume=0.5)

    assert isinstance(handle, TTSHandle)
    assert handle.getProperty("rate") == 200
    assert handle.getProperty("volume") == 0.5


def test_synthesize_produces_a_valid_wav_file(tmp_path):
    """One real, unmocked call through the full subprocess path -- confirms
    the actual dependency (pyttsx3, inside the worker subprocess) wires up
    correctly end to end."""
    engine = load_tts()
    out = str(tmp_path / "out.wav")

    result_path = synthesize(engine, "Testing one two three.", out)

    assert result_path == out
    assert os.path.exists(out)
    data, sr = sf.read(out)
    assert len(data) > 0


def test_synthesize_can_be_called_repeatedly_without_hanging(tmp_path):
    """Direct regression test for the confirmed hang: three real,
    subprocess-isolated synthesize() calls in one process, back to back.
    Before this fix (moving the real pyttsx3.init() into a subprocess),
    call 2 alone hung indefinitely with zero exception -- this test would
    never have completed."""
    engine = load_tts()

    for i in range(3):
        out = str(tmp_path / f"out_{i}.wav")
        t0 = time.perf_counter()
        synthesize(engine, f"This is test utterance number {i + 1}.", out)
        elapsed = time.perf_counter() - t0

        assert os.path.exists(out)
        assert elapsed < 30  # generous; real calls take a few seconds each


def test_synthesize_reads_rate_and_volume_from_the_handle(tmp_path, monkeypatch):
    """Confirms rate/volume actually reach the worker subprocess rather
    than silently falling back to defaults -- captures the argv the
    worker would have been invoked with instead of really spawning it."""
    import tts.synthesize as synthesize_mod

    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # A real successful worker run creates a real, valid WAV file --
        # synthesize() now verifies both that the file exists AND that it's
        # readable audio (see the real, confirmed silent-failure and
        # corrupt-file bugs this check exists for in tts/synthesize.py's
        # module docstring), so the fake here needs to produce real audio
        # too, not just touch an empty file.
        sf.write(cmd[3], np.zeros(50, dtype=np.int16), 16000)
        return FakeCompletedProcess()

    monkeypatch.setattr(synthesize_mod.subprocess, "run", fake_run)

    engine = load_tts(rate=210, volume=0.3)
    synthesize(engine, "hello", str(tmp_path / "out.wav"))

    assert captured["cmd"][-2:] == ["210", "0.3"]


def test_synthesize_raises_clear_error_on_subprocess_failure(tmp_path, monkeypatch):
    """A worker crash (bad pyttsx3 install, missing SAPI5 voice, ...)
    should surface as a clear RuntimeError with the subprocess's stderr,
    not a silent no-op or a confusing downstream failure."""
    import tts.synthesize as synthesize_mod

    class FakeCompletedProcess:
        returncode = 1
        stderr = "simulated worker crash"

    monkeypatch.setattr(synthesize_mod.subprocess, "run", lambda cmd, **kwargs: FakeCompletedProcess())

    with pytest.raises(RuntimeError, match="simulated worker crash"):
        synthesize(load_tts(), "hello", str(tmp_path / "out.wav"))


def test_synthesize_raises_timeout_error_instead_of_hanging_forever(tmp_path, monkeypatch):
    """If the worker subprocess itself somehow hangs (a different SAPI5
    issue than the one this fix targets, or a genuinely broken install),
    this should still fail loudly within SYNTHESIZE_TIMEOUT_SECONDS rather
    than hanging the caller forever -- same discipline as the
    GeminiLiveAudioModel timeout fix."""
    import subprocess as subprocess_mod
    import tts.synthesize as synthesize_mod

    def fake_run(cmd, **kwargs):
        raise subprocess_mod.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(synthesize_mod.subprocess, "run", fake_run)

    with pytest.raises(TimeoutError):
        synthesize(load_tts(), "hello", str(tmp_path / "out.wav"))


def test_run_worker_raises_clear_error_when_file_never_written(tmp_path, monkeypatch):
    """Direct regression test for a real, confirmed silent failure found
    on a live AWS benchmark run: pyttsx3's espeak driver on Linux can
    exit 0 with empty stdout/stderr and never write the output file at
    all (see _SAFE_CHUNK_CHARS's docstring -- this is exactly what
    happens for text over ~100-150 characters, before the chunking fix
    catches it first). Confirms this now fails loudly right here instead
    of surfacing as a confusing soundfile.LibsndfileError far downstream
    in a completely different file (pipeline_audio2audio.py)."""
    import tts.synthesize as synthesize_mod

    class FakeCompletedProcess:
        returncode = 0
        stderr = ""

    def fake_run_no_file(cmd, **kwargs):
        return FakeCompletedProcess()  # never creates cmd[3]'s file

    monkeypatch.setattr(synthesize_mod.subprocess, "run", fake_run_no_file)

    with pytest.raises(RuntimeError, match="never wrote"):
        synthesize(load_tts(), "short text under the chunk limit", str(tmp_path / "out.wav"))


def test_run_worker_raises_clear_error_when_file_is_corrupt(tmp_path, monkeypatch):
    """Direct regression test for a third, distinct real failure mode
    found on the very next benchmark run after the missing-file and
    GPU-memory fixes: the worker can exit 0 and create the file, but
    write invalid/truncated audio data --
    soundfile.LibsndfileError('...Format not recognised') reading it
    back. Confirms this is now caught right here (by actually reading
    the file back, not just checking it exists) instead of surfacing
    downstream in a completely different file."""
    import tts.synthesize as synthesize_mod

    class FakeCompletedProcess:
        returncode = 0
        stderr = ""

    def fake_run_corrupt_file(cmd, **kwargs):
        with open(cmd[3], "wb") as f:
            f.write(b"not a real wav file")
        return FakeCompletedProcess()

    monkeypatch.setattr(synthesize_mod.subprocess, "run", fake_run_corrupt_file)

    with pytest.raises(RuntimeError, match="isn't valid audio"):
        synthesize(load_tts(), "short text under the chunk limit", str(tmp_path / "out.wav"))


def test_chunk_text_keeps_every_chunk_under_the_limit():
    """The actual fix's core logic: confirmed real via direct testing on
    the affected Linux machine that pyttsx3/espeak silently fails past
    ~100-150 characters -- every chunk produced here must stay at or
    under _SAFE_CHUNK_CHARS regardless of input shape (short sentences,
    one very long run-on sentence with no punctuation at all, etc.)."""
    from tts.synthesize import _chunk_text, _SAFE_CHUNK_CHARS

    long_text = (
        "earlier in this conversation: I work as a nurse at the city "
        "hospital downtown; My favorite color is purple and I love "
        "hiking on weekends. some other things to remember: My dog's "
        "name is Max and he is a golden retriever."
    )
    chunks = _chunk_text(long_text)
    assert len(chunks) > 1  # this exact text is confirmed to exceed the limit
    for chunk in chunks:
        assert len(chunk) <= _SAFE_CHUNK_CHARS
    # No words dropped in the split -- every word from the original
    # appears somewhere across the chunks, in order.
    assert " ".join(chunks).split() == long_text.split()


def test_chunk_text_word_splits_a_single_over_length_sentence():
    """The rare fallback path: one "sentence" (no ./!/?/; anywhere) that
    alone exceeds the limit must still get split -- on word boundaries,
    not truncated or left whole to silently fail synthesis."""
    from tts.synthesize import _chunk_text, _SAFE_CHUNK_CHARS

    run_on = "word " * 40  # 200 chars, zero sentence-ending punctuation
    chunks = _chunk_text(run_on)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= _SAFE_CHUNK_CHARS


def test_synthesize_retries_by_halving_when_a_chunk_under_the_limit_still_fails(tmp_path, monkeypatch):
    """Direct regression test for the real reason a fixed chunk-size
    threshold alone isn't trustworthy: a live benchmark run hit the
    silent-failure bug on a real 96-character chunk -- UNDER
    _SAFE_CHUNK_CHARS=100 -- despite synthetic 100-character test text
    succeeding cleanly. Confirms a chunk that fails gets halved and
    retried rather than immediately surfacing an error, by failing only
    text longer than a threshold shorter than any single word here, so
    the first attempt fails and a retry at half the length must succeed."""
    import tts.synthesize as synthesize_mod

    class FakeCompletedProcess:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        text = kwargs["input"]
        output_path = cmd[3]
        if len(text) > 20:
            return FakeCompletedProcess()  # "succeeds" but never writes the file
        sf.write(output_path, np.zeros(50, dtype=np.int16), 16000)
        return FakeCompletedProcess()

    monkeypatch.setattr(synthesize_mod.subprocess, "run", fake_run)

    # Under _SAFE_CHUNK_CHARS (100) but over the fake failure threshold
    # (20) -- takes the short-text path first, fails, must fall through
    # to the chunk-and-retry path and succeed via halving.
    text = "this text is under the chunk limit but still too long for the fake worker"
    assert len(text) <= synthesize_mod._SAFE_CHUNK_CHARS

    out_path = str(tmp_path / "out.wav")
    result_path = synthesize(load_tts(), text, out_path)

    assert result_path == out_path
    data, sr = sf.read(out_path)
    assert sr == 16000
    assert len(data) > 0


def test_synthesize_chunks_and_concatenates_long_text(tmp_path, monkeypatch):
    """End-to-end (mocked worker) confirmation that long text produces
    ONE output file, transparently to the caller, by synthesizing each
    chunk separately and concatenating -- the actual fix for the real
    bug: _build_context_audio's memory-summary text routinely exceeds
    the confirmed length limit after a few real turns."""
    import tts.synthesize as synthesize_mod

    call_count = {"n": 0}

    class FakeCompletedProcess:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        # Each chunk call writes a small real WAV so the concatenation
        # step has real audio data to read and combine.
        samples = np.full(100, call_count["n"], dtype=np.int16)
        sf.write(cmd[3], samples, 16000)
        return FakeCompletedProcess()

    monkeypatch.setattr(synthesize_mod.subprocess, "run", fake_run)

    long_text = "word " * 40  # confirmed to exceed _SAFE_CHUNK_CHARS
    out_path = str(tmp_path / "out.wav")
    result_path = synthesize(load_tts(), long_text, out_path)

    assert result_path == out_path
    assert call_count["n"] > 1  # actually split into multiple worker calls
    data, sr = sf.read(out_path)
    assert sr == 16000
    assert len(data) == 100 * call_count["n"]  # all chunks' audio present, in order
