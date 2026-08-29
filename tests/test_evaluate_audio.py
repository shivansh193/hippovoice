"""
Dry-run validation for benchmarks/locomo/evaluate_audio.py -- exercises
the ingest/QA/scoring/checkpoint plumbing against MockAudioToAudioModel and
a fake LoCoMo conversation (no network, no GPU, no real API). Mirrors
tests/test_locomo_evaluate.py's monkeypatching style for run_locomo; real
Gemini Live behavior is validated separately (test_gemini_live_model.py's
live_api-marked tests) and on Kaggle, not here -- see this module's own
docstring for why a mock can't meaningfully validate "does audio
conditioning actually work," only "is everything wired correctly."
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import benchmarks.locomo.evaluate_audio as evaluate_audio_mod
from pipeline_audio2audio import MockAudioToAudioModel


def _make_extraction_llm(memory_type="fact"):
    """Mirrors tests/test_pipeline_audio2audio.py::_make_llm -- extracts
    exactly one memory per turn, tagged memory_type, so ingestion always
    produces something for retrieval/STM to work with."""
    mock = MagicMock()

    def per_turn(system, messages, max_tokens=512):
        sys_l = system.lower()
        user_content = messages[-1]["content"] if messages else ""
        if "extract" in sys_l or "memory" in sys_l:
            turn_text = user_content.split("Turn: ", 1)[-1].strip()
            return json.dumps([{"content": turn_text, "entity": "unknown", "type": memory_type}])
        return "ok"

    mock.generate.side_effect = per_turn
    mock.model_name = "mock-extraction-model"
    return mock


def _fake_synthesize(engine, text, output_path):
    samples = np.zeros(int(0.2 * 22050), dtype=np.float32)
    sf.write(output_path, samples, 22050)


class _TtsPatch:
    """Same bundling as test_pipeline_audio2audio.py's _TtsPatch -- covers
    both call sites that need it here: HippoAudioPipeline's own
    _build_context_audio, and evaluate_audio._synthesize_question_audio."""

    def __enter__(self):
        self._p1 = patch("tts.model.load_tts", return_value=MagicMock())
        self._p2 = patch("tts.synthesize.synthesize", side_effect=_fake_synthesize)
        self._p1.__enter__()
        self._p2.__enter__()
        return self

    def __exit__(self, *exc):
        self._p2.__exit__(*exc)
        self._p1.__exit__(*exc)


def _fake_conv():
    return {
        "conversation": {
            "session_1_date_time": "1 May, 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Alex", "text": "My dog's name is Max and he loves swimming."},
                {"dia_id": "D1:2", "speaker": "Alex", "text": "The weather has been nice this week."},
            ],
        },
        "qa": [{"question": "What is Alex's dog's name?", "answer": "Max", "category": 2}],
    }


def test_ingests_turns_as_text_without_calling_audio_model(monkeypatch):
    """Ingestion should never touch the audio model -- confirms the whole
    point of ingest_text_turn over process_turn for this benchmark's
    ingestion phase (see module docstring: no reason to "speak" all 369-689
    turns just to store them). Uses a conversation with zero QA pairs so
    any respond() call recorded can only have come from ingestion."""
    conv = _fake_conv()
    conv["qa"] = []
    monkeypatch.setattr(evaluate_audio_mod, "load_locomo", lambda data_path=None: [conv])
    model = MockAudioToAudioModel()

    with _TtsPatch():
        evaluate_audio_mod.run_locomo_audio(
            llm_client=_make_extraction_llm(), audio_model=model,
            num_conversations=1, max_qa_per_conversation=5, verbose=False,
        )

    assert model.calls == []  # no QA pairs -> zero respond() calls anywhere


def test_max_turns_per_conversation_truncates_ingestion(monkeypatch):
    """Same lever as run_locomo's own max_turns_per_conversation, same real
    reason: an API-backed llm_client can hit the provider's quota partway
    through ingesting one long conversation, since ingestion is one
    extraction call per turn with no way to combine turns into fewer calls."""
    conv = _fake_conv()
    conv["qa"] = []  # isolate to ingestion only, mirrors the test above
    monkeypatch.setattr(evaluate_audio_mod, "load_locomo", lambda data_path=None: [conv])

    ingested_calls = []
    llm = _make_extraction_llm()
    orig_generate = llm.generate
    def _spy(system, messages, max_tokens=512):
        if "extract" in system.lower() or "memory" in system.lower():
            ingested_calls.append(messages[-1]["content"])
        return orig_generate(system=system, messages=messages, max_tokens=max_tokens)
    llm.generate = MagicMock(side_effect=_spy)

    model = MockAudioToAudioModel()
    with _TtsPatch():
        evaluate_audio_mod.run_locomo_audio(
            llm_client=llm, audio_model=model,
            num_conversations=1, max_qa_per_conversation=5,
            max_turns_per_conversation=1, verbose=False,
        )

    # _fake_conv() has 2 turns -- capped to 1 means only 1 extraction call.
    assert len(ingested_calls) == 1


def test_qa_step_calls_audio_model_and_scores_transcript(monkeypatch):
    """The actual point of this harness: a QA question gets synthesized,
    sent through the audio model, and the returned transcript gets scored
    against gold with the same score_answer used everywhere else."""
    monkeypatch.setattr(evaluate_audio_mod, "load_locomo", lambda data_path=None: [_fake_conv()])

    model = MockAudioToAudioModel()
    # MockAudioToAudioModel's canned reply won't match "Max" -- this test is
    # about plumbing (did the call happen, did scoring run), not accuracy.
    with _TtsPatch():
        result = evaluate_audio_mod.run_locomo_audio(
            llm_client=_make_extraction_llm(), audio_model=model,
            num_conversations=1, max_qa_per_conversation=5, verbose=False,
        )

    assert len(model.calls) == 1  # exactly one QA pair in the fake conversation
    assert result["total"] == 1
    assert result["details"][0]["gold"] == "max"
    assert 0.0 <= result["details"][0]["f1"] <= 1.0


def test_qa_step_scores_correctly_when_transcript_matches_gold(monkeypatch):
    """Confirms the scoring path actually produces a high F1 for a correct
    answer, not just that it runs without crashing."""
    monkeypatch.setattr(evaluate_audio_mod, "load_locomo", lambda data_path=None: [_fake_conv()])

    class _CorrectAnswerModel(MockAudioToAudioModel):
        def respond(self, audio_path):
            self.calls.append(audio_path)
            return (f"{audio_path}.reply.wav", "Max")

    model = _CorrectAnswerModel()
    with _TtsPatch():
        result = evaluate_audio_mod.run_locomo_audio(
            llm_client=_make_extraction_llm(), audio_model=model,
            num_conversations=1, max_qa_per_conversation=5, verbose=False,
        )

    assert result["details"][0]["f1"] >= 0.7
    assert result["avg_f1"] >= 0.7


def test_qa_pair_does_not_pollute_stm_or_memory(monkeypatch):
    """The whole reason answer_question exists instead of reusing
    process_turn for QA: asking a benchmark question shouldn't itself
    become "recent history" that leaks into how later questions get
    answered."""
    conv = _fake_conv()
    conv["qa"].append({"question": "What is Alex's dog's name?", "answer": "Max", "category": 2})
    monkeypatch.setattr(evaluate_audio_mod, "load_locomo", lambda data_path=None: [conv])

    seen_recent_turns = []

    class _RecordingModel(MockAudioToAudioModel):
        pass

    orig_build_context = None

    model = _RecordingModel()
    with _TtsPatch():
        import pipeline_audio2audio as pa2a_mod
        orig_build_context = pa2a_mod.HippoAudioPipeline._build_context_audio

        def spy_build_context(self, retrieved, recent_turns, user_audio_path):
            seen_recent_turns.append(list(recent_turns))
            return orig_build_context(self, retrieved, recent_turns, user_audio_path)

        with patch.object(pa2a_mod.HippoAudioPipeline, "_build_context_audio", spy_build_context):
            evaluate_audio_mod.run_locomo_audio(
                llm_client=_make_extraction_llm(), audio_model=model,
                num_conversations=1, max_qa_per_conversation=5, verbose=False,
            )

    # Both QA calls should see the SAME recent_turns snapshot (from ingestion
    # only) -- if the first question had polluted STM, the second call's
    # snapshot would differ (grown by one).
    assert len(seen_recent_turns) == 2
    assert seen_recent_turns[0] == seen_recent_turns[1]


def test_checkpoint_records_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_audio_mod, "load_locomo", lambda data_path=None: [_fake_conv()])
    model = MockAudioToAudioModel()
    checkpoint_path = str(tmp_path / "checkpoint.json")

    with _TtsPatch():
        evaluate_audio_mod.run_locomo_audio(
            llm_client=_make_extraction_llm(), audio_model=model,
            num_conversations=1, max_qa_per_conversation=5,
            checkpoint_path=checkpoint_path, system_name="HippoAudio-Test", verbose=False,
        )

    with open(checkpoint_path) as f:
        state = json.load(f)
    assert state["fingerprint"]["system_name"] == "HippoAudio-Test"
    assert state["next_conversation_index"] == 1
