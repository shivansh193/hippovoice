"""
Dry-run validation for pipeline_audio2audio.py -- exercises memory
extraction, storage, retrieval, and context-audio construction against
MockAudioToAudioModel (no model weights, no GPU, no download) before any
real audio-to-audio model gets chosen or any AWS GPU instance time gets
spent. See pipeline_audio2audio.py's module docstring for why.

Real pyttsx3 gets exercised exactly once, in
test_real_tts_produces_valid_context_audio below, not repeatedly across
every test in this file. Confirmed on a real local run: calling
tts.synthesize.synthesize() more than once or twice within a single test
process reliably deadlocks pyttsx3's underlying SAPI5 COM loop on
Windows -- one test with 3 sequential real synthesis calls hung
indefinitely and had to be force-killed. All other tests here patch
load_tts/synthesize with a fast, deterministic stand-in instead; the
audio-mechanics logic itself (concatenation, resampling) is already fully
covered separately below without touching pyttsx3 at all.
"""

import json
import os
from unittest.mock import MagicMock, patch

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


def _make_llm_extracts_nothing():
    """Every turn produces zero memories -- isolates STM (recent_turns) from
    LTM (retrieved), since with this mock the memory stores stay genuinely
    empty regardless of how many turns are processed."""
    mock = MagicMock()
    mock.generate.side_effect = lambda system, messages, max_tokens=512: "[]"
    return mock


def _make_silent_wav(path: str, duration_s: float = 0.5, sr: int = 16000):
    """Tiny synthetic fixture -- avoids needing a real recorded audio file
    (tests/fixtures/ has none, see BUGS.md) for tests that only care about
    plumbing, not actual audio content."""
    samples = np.zeros(int(duration_s * sr), dtype=np.float32)
    sf.write(path, samples, sr)


def _fake_synthesize(engine, text, output_path):
    """Stand-in for tts.synthesize.synthesize -- see module docstring for
    why real pyttsx3 isn't used here. 22050Hz matches this machine's actual
    pyttsx3 output rate (confirmed earlier), so tests exercising the
    resampling path still reflect a realistic mismatch against 16000Hz
    user audio."""
    _make_silent_wav(output_path, duration_s=0.2, sr=22050)


class _TtsPatch:
    """Small helper bundling the two patches _build_context_audio needs
    mocked together, since it imports both load_tts and synthesize
    lazily/locally inside the method rather than at module level."""

    def __enter__(self):
        self._p1 = patch("tts.model.load_tts", return_value=MagicMock())
        self._p2 = patch("tts.synthesize.synthesize", side_effect=_fake_synthesize)
        self._p1.__enter__()
        self._p2.__enter__()
        return self

    def __exit__(self, *exc):
        self._p2.__exit__(*exc)
        self._p1.__exit__(*exc)


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
    straight through rather than synthesizing/concatenating for no reason.
    No TTS mocking needed here precisely because nothing should be
    synthesized at all -- a real pyttsx3 call happening here would itself
    be a test failure worth catching."""
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

    with _TtsPatch():
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


def test_stm_alone_triggers_context_even_with_nothing_stored(tmp_path):
    """LTM (retrieved) stays empty the whole time -- extraction produces
    nothing to store. STM should still inject context on turn 2 purely from
    the raw recent-turn buffer, proving STM and LTM are actually independent
    mechanisms, not STM riding along only because LTM also fires."""
    turn1_audio = str(tmp_path / "turn1.wav")
    turn2_audio = str(tmp_path / "turn2.wav")
    _make_silent_wav(turn1_audio)
    _make_silent_wav(turn2_audio)

    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm_extracts_nothing())

    with _TtsPatch():
        pipe.process_turn(turn1_audio, "just saying hello")
        pipe.process_turn(turn2_audio, "anyway, how's it going")

    assert pipe.semantic_memory.count() == 0
    assert pipe.episodic_memory.count() == 0  # confirms LTM genuinely never fired
    assert model.calls[0] == turn1_audio        # turn 1: no STM yet either
    assert model.calls[1] != turn2_audio        # turn 2: STM alone injected context


def test_stm_window_ages_out_oldest_turn(tmp_path):
    """recent_turns is a maxlen deque -- confirms it actually behaves like a
    sliding window (oldest turn dropped) rather than growing unbounded."""
    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm_extracts_nothing(), stm_window=2)

    with _TtsPatch():
        for i in range(4):
            audio = str(tmp_path / f"turn{i}.wav")
            _make_silent_wav(audio)
            pipe.process_turn(audio, f"turn number {i}")

    assert list(pipe.recent_turns) == ["turn number 2", "turn number 3"]


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


def test_ingest_text_turn_stores_without_calling_model(tmp_path):
    """The store-only path benchmarks/locomo/evaluate_audio.py needs: a
    conversation's history should populate memory/STM/turn count exactly
    like process_turn's own storage steps, but never touch the audio
    model at all -- no respond() call, so ingesting a 600-turn transcript
    doesn't burn 600 unwanted spoken replies."""
    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm(memory_type="fact"))

    pipe.ingest_text_turn("my dog's name is max")

    assert model.calls == []  # never called respond()
    assert (pipe.semantic_memory.count() + pipe.episodic_memory.count()) == 1
    assert list(pipe.recent_turns) == ["my dog's name is max"]
    assert pipe.current_turn == 1


def test_process_turn_and_ingest_text_turn_agree_on_storage(tmp_path):
    """Refactor safety net: process_turn now delegates its storage steps to
    ingest_text_turn rather than duplicating them -- two pipelines fed the
    same turn one via each path should end up with identical stored state."""
    audio_in = str(tmp_path / "turn1.wav")
    _make_silent_wav(audio_in)

    model_a = MockAudioToAudioModel()
    pipe_a = HippoAudioPipeline(audio_model=model_a, llm_client=_make_llm(memory_type="fact"))
    pipe_a.process_turn(audio_in, "my dog's name is max")

    model_b = MockAudioToAudioModel()
    pipe_b = HippoAudioPipeline(audio_model=model_b, llm_client=_make_llm(memory_type="fact"))
    pipe_b.ingest_text_turn("my dog's name is max")

    assert list(pipe_a.recent_turns) == list(pipe_b.recent_turns)
    assert pipe_a.current_turn == pipe_b.current_turn
    assert (pipe_a.semantic_memory.count(), pipe_a.episodic_memory.count()) == \
           (pipe_b.semantic_memory.count(), pipe_b.episodic_memory.count())


def test_answer_question_reads_without_storing_or_advancing_state(tmp_path):
    """The QA-only path benchmarks/locomo/evaluate_audio.py needs: asking a
    held-out question should retrieve + condition generation like a normal
    turn, but never store the question itself, never touch STM, and never
    advance the turn counter -- mirrors run_locomo's text-pipeline QA step,
    which calls retrieve()+generate() without also ingesting the question."""
    fact_audio = str(tmp_path / "fact.wav")
    question_audio = str(tmp_path / "question.wav")
    _make_silent_wav(fact_audio)
    _make_silent_wav(question_audio)

    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm(memory_type="fact"))

    with _TtsPatch():
        pipe.process_turn(fact_audio, "my dog's name is max")
        turn_after_fact = pipe.current_turn
        recent_after_fact = list(pipe.recent_turns)
        memory_count_after_fact = pipe.semantic_memory.count() + pipe.episodic_memory.count()

        response_audio, transcript = pipe.answer_question("what's my dog's name", question_audio)

    assert "mock reply" in transcript
    assert len(model.calls) == 2  # the fact turn's respond() call, plus this one
    # No side effects from asking the question:
    assert pipe.current_turn == turn_after_fact
    assert list(pipe.recent_turns) == recent_after_fact
    assert (pipe.semantic_memory.count() + pipe.episodic_memory.count()) == memory_count_after_fact


def test_real_tts_produces_valid_context_audio(tmp_path):
    """The one test in this file that touches real pyttsx3 -- confirms the
    actual dependency wires up correctly end to end, exactly once, rather
    than never being exercised for real at all."""
    user_audio = str(tmp_path / "turn.wav")
    _make_silent_wav(user_audio, duration_s=0.5, sr=16000)

    model = MockAudioToAudioModel()
    pipe = HippoAudioPipeline(audio_model=model, llm_client=_make_llm())

    combined = pipe._build_context_audio(
        [{"content": "user's dog is named max"}], ["earlier the user said hi"], user_audio
    )

    assert combined != user_audio
    data, sr = sf.read(combined)
    assert len(data) > 0
    assert sr == 16000  # resampled to match the real user audio's rate
