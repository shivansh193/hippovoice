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
