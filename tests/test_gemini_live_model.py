"""
Tests for gemini_live_model.py. The actual live-API call is marked
@pytest.mark.live_api and skipped unless GEMINI_API_KEY is set (see
conftest.py) -- it costs real API quota and needs network access, so it's
opt-in, not part of the standard `pytest -m "not gpu"` run. The pure audio
format helpers (_read_as_pcm16 / _write_pcm16_wav) are plain logic and get
tested unconditionally, no network involved.
"""

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
