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
