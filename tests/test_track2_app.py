"""
Tests for demo/track2_app.py's core logic (build_pipeline, seed_memory,
handle_turn, format_retrieved) -- deliberately never imports gradio, so
these run in the normal test suite without that dependency installed.
The Gradio UI wiring in _launch() is left unexercised here on purpose
(same reasoning as pipeline_audio2audio.py's own real-pyttsx3 test being
isolated to one call: UI wiring is a thin, visually-inspectable layer over
logic that's already fully covered by calling these functions directly).

whisper's real transcribe() is mocked throughout -- real STT isn't the
thing under test here, and downloading/running whisper-tiny for every
test run would be slow and non-deterministic for what these tests
actually check (does the pipeline get wired together correctly).
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from demo.track2_app import build_pipeline, seed_memory, handle_turn, format_retrieved
from pipeline_audio2audio import MockAudioToAudioModel


def _make_llm(memory_type="event"):
    """Mirrors tests/test_pipeline_audio2audio.py::_make_llm exactly."""
    mock = MagicMock()

    def per_turn(system, messages, max_tokens=512):
        sys_l = system.lower()
        user_content = messages[-1]["content"] if messages else ""
        if "extract" in sys_l or "memory" in sys_l:
            turn_text = user_content.split("Turn: ", 1)[-1].strip()
            return json.dumps([{"content": turn_text, "entity": "unknown", "type": memory_type}])
        return "ok"

    mock.generate.side_effect = per_turn
    return mock


def _make_silent_wav(path: str, duration_s: float = 0.5, sr: int = 16000):
    samples = np.zeros(int(duration_s * sr), dtype=np.float32)
    sf.write(path, samples, sr)


def test_build_pipeline_mock_backend_returns_loaded_pipeline():
    pipeline = build_pipeline("mock", llm_client=_make_llm())
    assert isinstance(pipeline.audio_model, MockAudioToAudioModel)
    assert pipeline.audio_model.loaded


def test_build_pipeline_rejects_unknown_backend():
    with pytest.raises(ValueError, match="mock.*qwen"):
        build_pipeline("not_a_real_backend", llm_client=_make_llm())


def test_seed_memory_stores_a_fact_without_a_spoken_turn():
    pipeline = build_pipeline("mock", llm_client=_make_llm(memory_type="fact"))
    status = seed_memory(pipeline, "The user's dog is named Max.")

    assert "Max" in status
    assert pipeline.current_turn == 1
    retrieved = pipeline.retrieve("What is the user's dog's name?", top_k=5)
    assert any("Max" in r.get("content", "") for r in retrieved)


def test_seed_memory_ignores_blank_input():
    pipeline = build_pipeline("mock", llm_client=_make_llm())
    status = seed_memory(pipeline, "   ")
    assert pipeline.current_turn == 0
    assert "nothing" in status.lower()


def test_handle_turn_transcribes_retrieves_and_responds(tmp_path):
    pipeline = build_pipeline("mock", llm_client=_make_llm(memory_type="fact"))
    seed_memory(pipeline, "The user's favorite color is teal.")

    audio_path = str(tmp_path / "turn.wav")
    _make_silent_wav(audio_path)

    with patch("stt.transcribe.transcribe", return_value="What is my favorite color?"):
        result = handle_turn(pipeline, stt_model=object(), user_audio_path=audio_path)

    assert result["user_text"] == "What is my favorite color?"
    assert any("teal" in r.get("content", "") for r in result["retrieved"])
    # MockAudioToAudioModel.respond() is deterministic -- see its own
    # docstring: echoes the (possibly context-prefixed) input path back.
    assert result["response_audio_path"].endswith(".mock_response.wav")
    assert "mock reply" in result["response_transcript"]
    # The turn's own text should now be stored too (process_turn calls
    # ingest_text_turn internally after generation).
    assert pipeline.current_turn == 2


def test_format_retrieved_handles_empty_and_nonempty():
    assert "nothing retrieved" in format_retrieved([]).lower()

    formatted = format_retrieved([{"content": "likes hiking", "_score": 0.842}])
    assert "likes hiking" in formatted
    assert "0.842" in formatted
