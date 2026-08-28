"""
Dry-run validation for baselines/weight_edit_baseline.py -- exercises
extraction reuse, edit-request conversion, per-conversation reset, and
harness wiring against MockWeightEditor (no model weights, no GPU) before
any real ROME/MEMIT implementation gets built on Kaggle. See that module's
docstring for why the actual edit mechanics can't be meaningfully mocked
and are deliberately out of scope here.
"""

import json
from unittest.mock import MagicMock

from baselines.weight_edit_baseline import WeightEditBaseline, MockWeightEditor


def _make_llm(turn_to_memories: dict[str, list[dict]], content_to_edit: dict[str, dict]):
    """
    turn_to_memories: raw turn text -> what extract_memories() should return
    content_to_edit: a memory's content -> the edit-request dict (or
        {"skip": True}) the conversion step should return for it
    """
    mock = MagicMock()

    def dispatch(system, messages, max_tokens=512):
        user_content = messages[-1]["content"] if messages else ""
        if "knowledge-edit" in system:
            # EDIT_EXTRACTION_PROMPT always ends with "Memory: {content}"
            content = user_content.split("Memory: ", 1)[-1].strip()
            return json.dumps(content_to_edit.get(content, {"skip": True}))
        if "memory extraction" in system:
            turn_text = user_content.split("Turn: ", 1)[-1].strip()
            return json.dumps(turn_to_memories.get(turn_text, []))
        return "ok"

    mock.generate.side_effect = dispatch
    return mock


def test_ingest_converts_editable_memory_and_applies_edit():
    llm = _make_llm(
        turn_to_memories={"my dog's name is max": [{"content": "user's dog is named max", "type": "fact"}]},
        content_to_edit={"user's dog is named max": {
            "prompt": "user's dog is named", "subject": "user's dog", "target_new": "max",
        }},
    )
    editor = MockWeightEditor()
    pipe = WeightEditBaseline(llm_client=llm, editor=editor)

    pipe.ingest_text_turn("my dog's name is max")

    assert editor.edits == {"user's dog": "max"}
    assert len(editor.edit_log) == 1


def test_non_editable_memory_is_skipped_not_forced():
    """An event/feeling has no clean single-answer shape -- conversion
    should return skip, and no edit should ever reach the editor."""
    llm = _make_llm(
        turn_to_memories={"i went hiking and felt amazing": [
            {"content": "user went hiking and felt amazing", "type": "event"},
        ]},
        content_to_edit={"user went hiking and felt amazing": {"skip": True}},
    )
    editor = MockWeightEditor()
    pipe = WeightEditBaseline(llm_client=llm, editor=editor)

    pipe.ingest_text_turn("i went hiking and felt amazing")

    assert editor.edits == {}


def test_malformed_conversion_output_does_not_crash():
    mock = MagicMock()

    def dispatch(system, messages, max_tokens=512):
        if "knowledge-edit" in system:
            return "not valid json at all"
        if "memory extraction" in system:
            return json.dumps([{"content": "some fact", "type": "fact"}])
        return "ok"

    mock.generate.side_effect = dispatch
    editor = MockWeightEditor()
    pipe = WeightEditBaseline(llm_client=mock, editor=editor)

    pipe.ingest_text_turn("whatever turn text")  # should not raise

    assert editor.edits == {}


def test_new_conversation_resets_editor_before_ingesting():
    """Simulates run_locomo's per-conversation pattern: a fresh
    WeightEditBaseline gets constructed for each conversation via
    pipeline_factory. The editor must not carry over edits from whatever
    conversation used it last."""
    llm = _make_llm(
        turn_to_memories={"turn one": [{"content": "fact one", "type": "fact"}]},
        content_to_edit={"fact one": {"prompt": "the answer is", "subject": "fact one", "target_new": "yes"}},
    )
    editor = MockWeightEditor()

    conv1 = WeightEditBaseline(llm_client=llm, editor=editor)
    conv1.ingest_text_turn("turn one")
    assert editor.edits == {"fact one": "yes"}

    # New conversation, same editor instance (as run_locomo would reuse the
    # underlying real model across conversations rather than reloading it) --
    # constructing WeightEditBaseline again must reset it.
    conv2 = WeightEditBaseline(llm_client=llm, editor=editor)
    assert editor.edits == {}


def test_qa_generation_reads_from_edited_model_not_extraction_llm():
    """The actual point of this baseline: QA answers come from the edited
    model's own weights, not from a separate retrieval+context step."""
    llm = _make_llm(
        turn_to_memories={"my dog's name is max": [{"content": "user's dog is named max", "type": "fact"}]},
        content_to_edit={"user's dog is named max": {
            "prompt": "user's dog is named", "subject": "user's dog", "target_new": "max",
        }},
    )
    editor = MockWeightEditor()
    pipe = WeightEditBaseline(llm_client=llm, editor=editor)
    pipe.ingest_text_turn("my dog's name is max")

    answer = pipe.llm.generate(
        system="Answer the question.",
        messages=[{"role": "user", "content": "user's dog is named"}],
        max_tokens=10,
    )
    assert answer == "max"

    # A question about something never edited in should get the
    # editor's honest "don't know" rather than a hallucinated guess.
    unrelated = pipe.llm.generate(
        system="Answer the question.",
        messages=[{"role": "user", "content": "what is the capital of France"}],
        max_tokens=10,
    )
    assert unrelated == "[no edited fact matches this prompt]"
