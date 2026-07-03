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
