import json
from unittest.mock import MagicMock

from pipeline import HippoVoicePipeline


def _make_llm():
    """LLM mock supporting both generate() and generate_batch() with
    identical per-item behavior -- batching one call per item must produce
    the same output as issuing the calls separately."""
    mock = MagicMock()

    def per_turn(system, messages, max_tokens=512):
        sys_l = system.lower()
        user_content = messages[-1]["content"] if messages else ""
        if "extract" in sys_l or "memory" in sys_l:
            turn_text = user_content.split("Turn: ", 1)[-1].strip()
            return json.dumps([{"content": turn_text, "entity": "unknown", "type": "fact"}])
        if "summarise" in sys_l or "summary" in sys_l:
            return "summary"
        return "ok"

    mock.generate.side_effect = per_turn
    mock.generate_batch.side_effect = lambda system, messages_list, max_tokens=512: [
        per_turn(system, messages, max_tokens) for messages in messages_list
    ]
    return mock


def test_batch_ingestion_matches_sequential_ingestion():
    texts = [f"turn number {i}" for i in range(12)]

    seq_pipeline = HippoVoicePipeline(llm_client=_make_llm(), text_only=True)
    for t in texts:
        seq_pipeline.ingest_text_turn(t)

    batch_pipeline = HippoVoicePipeline(llm_client=_make_llm(), text_only=True)
    batch_pipeline.ingest_text_turns_batch(texts)

    assert batch_pipeline.current_turn == seq_pipeline.current_turn == len(texts)
    assert batch_pipeline.memory.count() == seq_pipeline.memory.count()

    seq_contents = sorted(m["content"] for m in seq_pipeline.memory.get_all())
    batch_contents = sorted(m["content"] for m in batch_pipeline.memory.get_all())
    assert seq_contents == batch_contents


def test_batch_ingestion_in_chunks_matches_single_batch():
    """Chunking a batch (as the benchmarks do) must give the same result as
    one big batch call -- current_turn/decay must not depend on chunk size."""
    texts = [f"turn number {i}" for i in range(25)]

    one_shot = HippoVoicePipeline(llm_client=_make_llm(), text_only=True)
    one_shot.ingest_text_turns_batch(texts)

    chunked = HippoVoicePipeline(llm_client=_make_llm(), text_only=True)
    for b in range(0, len(texts), 10):
        chunked.ingest_text_turns_batch(texts[b:b + 10])

    assert one_shot.current_turn == chunked.current_turn == len(texts)
    assert one_shot.memory.count() == chunked.memory.count()


def test_batch_ingestion_calls_generate_batch_not_generate_per_turn():
    llm = _make_llm()
    pipe = HippoVoicePipeline(llm_client=llm, text_only=True)
    pipe.ingest_text_turns_batch(["a", "b", "c", "d"])

    llm.generate_batch.assert_called_once()
    # generate() may still be called by the periodic compress cycle, but
    # never once per turn for extraction -- with only 4 turns (< DECAY_EVERY)
    # no decay cycle fires, so generate() shouldn't be called at all here.
    assert llm.generate.call_count == 0


def test_batch_ingestion_empty_list():
    pipe = HippoVoicePipeline(llm_client=_make_llm(), text_only=True)
    pipe.ingest_text_turns_batch([])
    assert pipe.current_turn == 0
    assert pipe.memory.count() == 0


# ── _maybe_decay actually mutates the store ─────────────────────────────────────
# Regression coverage: get_all() previously returned memory dicts with no "id"
# field, so _maybe_decay()'s `m.get("id")` was always None and forgotten
# memories were never actually deleted -- the store grew unbounded forever.

def test_forgotten_memories_are_actually_deleted_from_store():
    pipe = HippoVoicePipeline(llm_client=_make_llm(), text_only=True)
    # Plain, low-intensity neutral turns -- no emotion keywords, no intensity
    # boosters -- decay fast enough to be forgotten well within 90 turns
    # (matches tests/test_decay.py's neutral/low-intensity forgetting math).
    texts = [f"the weather was mild on day {i}" for i in range(90)]
    for t in texts:
        pipe.ingest_text_turn(t)

    assert pipe.memory.count() < len(texts), (
        "store should have actually forgotten old low-salience memories, "
        "not accumulated every single one forever"
    )


def test_compress_cycle_persists_compressed_entry_and_removes_originals():
    pipe = HippoVoicePipeline(llm_client=_make_llm(), text_only=True)
    # Seed memories directly in the compress band at turn 60 (neutral/0.3 ->
    # salience ~0.13, between FORGET_THRESHOLD 0.08 and COMPRESS_THRESHOLD
    # 0.25 -- same arithmetic as tests/test_decay.py's compress-threshold test).
    original_ids = [
        pipe.memory.add({
            "content": f"low salience memory {i}",
            "emotion": {"label": "neutral", "intensity": 0.3},
            "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
        })
        for i in range(3)
    ]

    pipe.current_turn = 60  # multiple of DECAY_EVERY so _maybe_decay actually fires
    pipe._maybe_decay()

    remaining = pipe.memory.get_all()
    remaining_ids = {m["id"] for m in remaining}

    assert not (set(original_ids) & remaining_ids), (
        "original low-salience memories should be gone once compressed away"
    )
    compressed = [m for m in remaining if m.get("compressed_from")]
    assert len(compressed) == 1
    assert compressed[0]["compressed_from"] == 3
