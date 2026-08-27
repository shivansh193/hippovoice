"""
Dry-run validation for pipeline_audio2audio.py -- exercises memory
extraction, storage, retrieval, and context-audio construction against
MockAudioToAudioModel (no model weights, no GPU, no download) before any
real audio-to-audio model gets chosen or any AWS GPU instance time gets
spent. See pipeline_audio2audio.py's module docstring for why.
"""

import json
import os
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from pipeline_audio2audio import HippoAudioPipeline, MockAudioToAudioModel, _concatenate_audio


def _make_llm(memory_type="event"):
    """Mirrors tests/test_pipeline.py::_make_llm exactly -- same reasoning:
    batching isn't used here, but generate()'s extraction-detection logic
    needs to match how extract_memories() actually prompts it."""
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
    """Tiny synthetic fixture -- avoids needing a real recorded audio file
    (tests/fixtures/ has none, see BUGS.md) for tests that only care about
    plumbing, not actual audio content."""
    samples = np.zeros(int(duration_s * sr), dtype=np.float32)
    sf.write(path, samples, sr)


def test_process_turn_calls_model_and_stores_memory(tmp_path):
    audio_in = str(tmp_path / "turn1.wav")
    _make_silent_wav(audio_in)

    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm(memory_type="fact"))

    response_audio, transcript = pipe.process_turn(audio_in, "my dog's name is max")

    assert model.loaded is False  # pipeline doesn't call load() itself -- caller's responsibility, matches AudioToAudioModel's separation of load() vs respond()
    assert len(model.calls) == 1
    assert "mock reply" in transcript
    assert (pipe.semantic_memory.count() + pipe.episodic_memory.count()) == 1


def test_no_context_audio_on_first_turn(tmp_path):
    """Nothing stored yet -- _build_context_audio should pass the raw input
    straight through rather than synthesizing/concatenating for no reason."""
    audio_in = str(tmp_path / "turn1.wav")
    _make_silent_wav(audio_in)

    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm())

    pipe.process_turn(audio_in, "the weather was nice today")

    assert model.calls[0] == audio_in  # unchanged -- no context to inject yet


def test_context_audio_injected_when_relevant_memory_exists(tmp_path):
    """The actual point of this pipeline over the Mini-Omni exploratory pass:
    a fact stated in turn 1 should get synthesized and prepended to turn 2's
    audio when turn 2 is actually relevant to it -- real conditioning, not
    just capture."""
    turn1_audio = str(tmp_path / "turn1.wav")
    turn2_audio = str(tmp_path / "turn2.wav")
    _make_silent_wav(turn1_audio)
    _make_silent_wav(turn2_audio)

    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm(memory_type="fact"))

    pipe.process_turn(turn1_audio, "my dog's name is max")
    pipe.process_turn(turn2_audio, "what's my dog's name again")

    assert model.calls[0] == turn1_audio  # first turn: nothing to inject
    assert model.calls[1] != turn2_audio  # second turn: context was prepended
    assert os.path.exists(model.calls[1])

    # The injected audio should be longer than the raw input alone (context
    # clip + original), not just a copy of it.
    combined_duration = len(sf.read(model.calls[1])[0])
    raw_duration = len(sf.read(turn2_audio)[0])
    assert combined_duration > raw_duration


def test_concatenate_audio_produces_combined_duration(tmp_path):
    first = str(tmp_path / "a.wav")
    second = str(tmp_path / "b.wav")
    out = str(tmp_path / "combined.wav")
    _make_silent_wav(first, duration_s=0.3)
    _make_silent_wav(second, duration_s=0.5)

    _concatenate_audio(first, second, out)

    data, sr = sf.read(out)
    assert sr == 16000
    assert len(data) == int(0.3 * 16000) + int(0.5 * 16000)


def test_concatenate_audio_resamples_mismatched_sample_rates(tmp_path):
    """Confirmed on a real local run that this mismatch actually happens
    (pyttsx3 output vs. a model's expected input rate aren't guaranteed to
    match) -- output should end up at second_path's rate (the real audio
    going to the model), with first_path resampled to fit, not rejected."""
    first = str(tmp_path / "a.wav")   # synthesized context clip, 22050Hz
    second = str(tmp_path / "b.wav")  # real user audio, 16000Hz
    out = str(tmp_path / "combined.wav")
    _make_silent_wav(first, duration_s=0.3, sr=22050)
    _make_silent_wav(second, duration_s=0.5, sr=16000)

    _concatenate_audio(first, second, out)

    data, sr = sf.read(out)
    assert sr == 16000  # matches second_path, not first_path
    assert len(data) == int(0.3 * 16000) + int(0.5 * 16000)


def test_fact_and_event_route_to_correct_stores(tmp_path):
    audio_in = str(tmp_path / "turn1.wav")
    _make_silent_wav(audio_in)

    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm(memory_type="fact"))
    pipe.process_turn(audio_in, "user is a vegetarian")
    assert pipe.semantic_memory.count() == 1
    assert pipe.episodic_memory.count() == 0

    model2 = MockAudioToAudioModel()
    pipe2 = HippoAudioPipeline(audio_model=model2, llm_client=_make_llm(memory_type="event"))
    pipe2.process_turn(audio_in, "user went hiking yesterday")
    assert pipe2.episodic_memory.count() == 1
    assert pipe2.semantic_memory.count() == 0
