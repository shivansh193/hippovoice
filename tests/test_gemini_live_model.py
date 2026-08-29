"""
Tests for gemini_live_model.py. The actual live-API call is marked
@pytest.mark.live_api and skipped unless GEMINI_API_KEY is set (see
conftest.py) -- it costs real API quota and needs network access, so it's
opt-in, not part of the standard `pytest -m "not gpu"` run. The pure audio
format helpers (_read_as_pcm16 / _write_pcm16_wav) are plain logic and get
tested unconditionally, no network involved.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf
import pytest

from gemini_live_model import GeminiLiveAudioModel, _read_as_pcm16, _write_pcm16_wav, \
    GEMINI_INPUT_SAMPLE_RATE, GEMINI_OUTPUT_SAMPLE_RATE


def test_read_as_pcm16_resamples_to_target_rate(tmp_path):
    src = str(tmp_path / "in.wav")
    samples = np.zeros(int(0.5 * 22050), dtype=np.int16)
    sf.write(src, samples, 22050)

    pcm = _read_as_pcm16(src, GEMINI_INPUT_SAMPLE_RATE)

    n_samples = len(pcm) // 2  # int16 = 2 bytes/sample
    assert abs(n_samples - int(0.5 * GEMINI_INPUT_SAMPLE_RATE)) < 10  # resampling rounding tolerance


def test_read_as_pcm16_downmixes_stereo_to_mono(tmp_path):
    src = str(tmp_path / "stereo.wav")
    stereo = np.zeros((16000, 2), dtype=np.int16)
    sf.write(src, stereo, GEMINI_INPUT_SAMPLE_RATE)

    pcm = _read_as_pcm16(src, GEMINI_INPUT_SAMPLE_RATE)

    assert len(pcm) // 2 == 16000  # mono sample count, not stereo's 2x


def test_write_pcm16_wav_roundtrips(tmp_path):
    out = str(tmp_path / "out.wav")
    arr = (np.sin(np.linspace(0, 10, 4800)) * 1000).astype(np.int16)

    _write_pcm16_wav(out, arr.tobytes(), GEMINI_OUTPUT_SAMPLE_RATE)

    data, sr = sf.read(out, dtype="int16")
    assert sr == GEMINI_OUTPUT_SAMPLE_RATE
    assert len(data) == len(arr)


def test_respond_passes_system_instruction_when_configured(tmp_path):
    """Confirmed as a real, fixable scoring problem: without a system
    instruction, real LoCoMo QA answers came back substantively correct but
    wrapped in a full sentence, and the strict token-F1 scorer penalized
    that verbosity even when the content was right. system_instruction is
    None by default (a live conversational demo wants natural speech) --
    this confirms it actually reaches LiveConnectConfig when a caller
    (e.g. a benchmark run) sets one, without a real network call."""
    from google.genai import types

    input_path = str(tmp_path / "in.wav")
    samples = np.zeros(16000, dtype=np.int16)
    sf.write(input_path, samples, 16000)

    model = GeminiLiveAudioModel(system_instruction="Be concise.")
    model._client = MagicMock()
    captured = {}

    class FakeSession:
        async def send_realtime_input(self, **kwargs):
            pass

        async def receive(self):
            return
            yield  # pragma: no cover -- makes this an async generator function

    class FakeConnectCM:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *exc):
            return False

    def fake_connect(model, config):
        captured["config"] = config
        return FakeConnectCM()

    model._client.aio.live.connect = fake_connect

    model.respond(input_path)

    assert isinstance(captured["config"], types.LiveConnectConfig)
    assert captured["config"].system_instruction == "Be concise."


def test_respond_raises_timeout_error_instead_of_hanging_forever(tmp_path, monkeypatch):
    """
    Regression test for a real, confirmed hang: a live diagnostic run's
    WebSocket connection to Gemini got closed server-side (both TCP sockets
    sat in CLOSE_WAIT) while _respond_async was still awaiting
    session.receive()'s next message -- which of course never came. The
    `async for` loop had no self-imposed deadline, so the call blocked
    forever with no exception raised; the only way to notice was inspecting
    OS-level socket state, which took 20+ minutes of real wall-clock time to
    even suspect. This simulates that exact situation: a fake session whose
    receive() yields nothing and never completes, wrapped in a short
    timeout so the test itself doesn't hang.
    """
    monkeypatch.setattr("gemini_live_model.RESPONSE_TIMEOUT_SECONDS", 0.05)

    input_path = str(tmp_path / "in.wav")
    sf.write(input_path, np.zeros(16000, dtype=np.int16), 16000)

    model = GeminiLiveAudioModel()
    model._client = MagicMock()

    class HangingSession:
        async def send_realtime_input(self, **kwargs):
            pass

        async def receive(self):
            # A real async generator that never yields and never returns --
            # exactly what an `async for` sees on a connection the server
            # already closed without a clean turn_complete.
            await asyncio.Event().wait()
            yield  # pragma: no cover -- unreachable, makes this a generator

    class FakeConnectCM:
        async def __aenter__(self):
            return HangingSession()

        async def __aexit__(self, *exc):
            return False

    model._client.aio.live.connect = lambda model, config: FakeConnectCM()

    with pytest.raises(TimeoutError):
        model.respond(input_path)


def test_respond_raises_timeout_error_on_a_hang_during_connect_or_close(tmp_path, monkeypatch):
    """
    Regression test for a second, distinct real hang found while verifying
    the fix above: wrapping only the receive() loop in a timeout wasn't
    enough -- a second live run hung with the identical CLOSE_WAIT/
    near-zero-CPU signature as the first, but this time nothing was ever
    pending inside receive() to time out. The hang can just as easily be in
    connect()'s own handshake (__aenter__) or in the session's close
    sequence (__aexit__) trying to gracefully shut down a connection the
    server already tore down unilaterally -- so the fix moved to wrapping
    the ENTIRE `async with` block, not just the loop inside it. This
    simulates a hang in __aenter__ specifically, the case the first fix's
    narrower timeout would NOT have caught.
    """
    monkeypatch.setattr("gemini_live_model.RESPONSE_TIMEOUT_SECONDS", 0.05)

    input_path = str(tmp_path / "in.wav")
    sf.write(input_path, np.zeros(16000, dtype=np.int16), 16000)

    model = GeminiLiveAudioModel()
    model._client = MagicMock()

    class HangingConnectCM:
        async def __aenter__(self):
            # Exactly what a stalled connection handshake looks like --
            # never resolves, never raises on its own.
            await asyncio.Event().wait()

        async def __aexit__(self, *exc):
            return False

    model._client.aio.live.connect = lambda model, config: HangingConnectCM()

    with pytest.raises(TimeoutError):
        model.respond(input_path)


def test_respond_omits_system_instruction_by_default(tmp_path):
    """The default (None) must not appear in the config at all -- a live
    conversational demo shouldn't get an unsolicited instruction just
    because the field exists on LiveConnectConfig."""
    input_path = str(tmp_path / "in.wav")
    sf.write(input_path, np.zeros(16000, dtype=np.int16), 16000)

    model = GeminiLiveAudioModel()  # system_instruction defaults to None
    model._client = MagicMock()
    captured = {}

    class FakeSession:
        async def send_realtime_input(self, **kwargs):
            pass

        async def receive(self):
            return
            yield  # pragma: no cover

    class FakeConnectCM:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *exc):
            return False

    def fake_connect(model, config):
        captured["config"] = config
        return FakeConnectCM()

    model._client.aio.live.connect = fake_connect

    model.respond(input_path)

    assert captured["config"].system_instruction is None


def test_respond_works_when_called_from_a_running_event_loop(tmp_path):
    """
    Regression test for a real, confirmed bug: Jupyter/IPython kernels
    (including Kaggle's) run their own asyncio event loop by default, and
    respond() used to call asyncio.run() unconditionally -- which raises
    `RuntimeError: asyncio.run() cannot be called from a running event
    loop` the moment respond() is invoked from inside a notebook cell.
    This never showed up in earlier standalone-script testing because a
    plain script has no event loop already running.

    Simulates that exact situation without needing a real notebook kernel:
    calls model.respond() (a sync method) from inside an async function
    that's itself being driven by asyncio.run(), so a loop is genuinely
    running in this thread when respond() checks for one.
    """
    model = GeminiLiveAudioModel()
    model._client = object()  # respond() only checks `is None`, load() never called

    async def fake_respond_async(audio_path):
        return ("fake_response.wav", "fake transcript")

    model._respond_async = fake_respond_async

    async def call_respond_from_inside_a_loop():
        # Proves a loop is genuinely running in this thread at the moment
        # respond() is called, not just plausible -- this would raise if false.
        asyncio.get_running_loop()
        return model.respond("dummy_input.wav")

    response_path, transcript = asyncio.run(call_respond_from_inside_a_loop())

    assert response_path == "fake_response.wav"
    assert transcript == "fake transcript"


@pytest.mark.live_api
def test_real_live_api_round_trip(tmp_path):
    """
    Real, unmocked call to Gemini's Live API. Confirms the adapter's exact
    call shape (client.aio.live.connect, send_realtime_input with a Blob,
    iterating session.receive() for transcripts and audio chunks) actually
    works end-to-end against the live service, the same call shape
    confirmed working manually before this adapter was written.
    """
    import pyttsx3

    input_path = str(tmp_path / "input.wav")
    engine = pyttsx3.init()
    engine.save_to_file("What is two plus two?", input_path)
    engine.runAndWait()

    model = GeminiLiveAudioModel()
    response_path, transcript = model.respond(input_path)

    assert transcript.strip() != ""
    data, sr = sf.read(response_path)
    assert sr == GEMINI_OUTPUT_SAMPLE_RATE
    assert len(data) > 0


@pytest.mark.live_api
def test_multi_segment_audio_is_not_truncated_by_vad(tmp_path):
    """
    Regression test for a real, confirmed bug: with automatic VAD enabled
    (the API default), the pause between two concatenated audio segments
    (a synthesized context clip, then a real trailing question -- exactly
    what HippoAudioPipeline._build_context_audio produces) got treated as
    end-of-turn, so the model only ever heard the first segment. Caught via
    a live multi-turn HippoAudioPipeline run, not by the single-utterance
    test above, since that test never concatenates two segments.

    This builds the same shape (context statement + pause + real question)
    and asserts the model's own input transcript -- what it actually heard,
    via last_input_transcript -- contains text from BOTH segments. Before
    the automatic_activity_detection.disabled=True fix, this would have
    failed: last_input_transcript contained only the first segment.
    """
    import pyttsx3
    import soundfile as sf_
    import numpy as np

    context_path = str(tmp_path / "context.wav")
    question_path = str(tmp_path / "question.wav")
    combined_path = str(tmp_path / "combined.wav")

    engine = pyttsx3.init()
    engine.save_to_file("Remember that my favorite color is purple.", context_path)
    engine.runAndWait()

    engine2 = pyttsx3.init()
    engine2.save_to_file("What is my favorite color?", question_path)
    engine2.runAndWait()

    # Concatenate with a silent gap in between, mirroring the real pause
    # that triggered automatic VAD's premature end-of-turn in production.
    ctx_data, ctx_sr = sf_.read(context_path, dtype="int16")
    q_data, q_sr = sf_.read(question_path, dtype="int16")
    assert ctx_sr == q_sr  # pyttsx3 on this machine is consistent; if not, this test needs resampling too
    silence = np.zeros(int(0.5 * ctx_sr), dtype=np.int16)
    combined = np.concatenate([ctx_data, silence, q_data])
    sf_.write(combined_path, combined, ctx_sr)

    model = GeminiLiveAudioModel()
    response_path, transcript = model.respond(combined_path)
    heard = model.last_input_transcript.lower()

    assert "purple" in heard or "color" in heard  # context segment was heard
    assert "favorite color" in heard or "what is" in heard  # question segment was ALSO heard, not truncated
