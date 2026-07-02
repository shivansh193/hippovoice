"""
Tests for the Mem0-style and A-MEM-style baseline reimplementations.

These are CPU-only (mock_llm fixture) -- no GPU or real model download needed.
"""

import json
from unittest.mock import MagicMock

from baselines.mem0_baseline import Mem0Baseline
from baselines.a_mem_baseline import AMemBaseline


# ── Mem0Baseline ────────────────────────────────────────────────────────────

def test_mem0_ingest_and_retrieve(mock_llm):
    baseline = Mem0Baseline(llm_client=mock_llm)
    baseline.ingest_text_turn("I have a golden retriever named Max who loves swimming")
    baseline.ingest_text_turn("The weather was cloudy today")

    results = baseline.retrieve("tell me about the user's dog", top_k=5)
    contents = [r["content"].lower() for r in results]
    assert any("dog" in c or "retriever" in c or "max" in c for c in contents)


def test_mem0_no_forgetting(mock_llm):
    """Unlike HippoVoice, Mem0-style has no decay -- old memories stay forever."""
    baseline = Mem0Baseline(llm_client=mock_llm)
    baseline.ingest_text_turn("I have a golden retriever named Max who loves swimming")
    for i in range(50):
        baseline.ingest_text_turn(f"Neutral filler turn number {i}.")

    all_memories = baseline.store.get_all()
    assert any("dog" in m["content"].lower() or "max" in m["content"].lower() for m in all_memories)


def test_mem0_decision_update_replaces_memory():
    """Directly exercise the ADD/UPDATE/DELETE decision path with a purpose-built mock."""
    llm = MagicMock()

    call_count = {"n": 0}

    def side_effect(system, messages, max_tokens=512):
        sys_l = system.lower()
        user_content = messages[-1]["content"]
        if "memory database" in sys_l:
            call_count["n"] += 1
            # First candidate in the prompt is always the one we "update"
            first_id = user_content.split("id=", 1)[1].split(":", 1)[0].strip()
            return json.dumps({"action": "UPDATE", "target_id": first_id})
        # Extraction: return the turn content as a single fact
        turn_text = user_content.split("Turn: ", 1)[-1].strip()
        return json.dumps([{"content": turn_text, "entity": "user", "type": "fact"}])

    llm.generate.side_effect = side_effect

    baseline = Mem0Baseline(llm_client=llm)
    baseline.ingest_text_turn("user's favorite color is blue")
    baseline.ingest_text_turn("user's favorite color is blue")  # near-duplicate -> triggers decision

    assert call_count["n"] >= 1
    # UPDATE should keep exactly one memory for this fact, not two
    matching = [m for m in baseline.store.get_all() if "favorite color" in m["content"]]
    assert len(matching) == 1


def test_mem0_decision_delete_removes_memory():
    llm = MagicMock()

    def side_effect(system, messages, max_tokens=512):
        sys_l = system.lower()
        user_content = messages[-1]["content"]
        if "memory database" in sys_l:
            first_id = user_content.split("id=", 1)[1].split(":", 1)[0].strip()
            return json.dumps({"action": "DELETE", "target_id": first_id})
        turn_text = user_content.split("Turn: ", 1)[-1].strip()
        return json.dumps([{"content": turn_text, "entity": "user", "type": "fact"}])

    llm.generate.side_effect = side_effect

    baseline = Mem0Baseline(llm_client=llm)
    baseline.ingest_text_turn("user likes hiking")
    baseline.ingest_text_turn("user no longer likes hiking")  # contradicts -> DELETE

    remaining = [m for m in baseline.store.get_all() if "hiking" in m["content"]]
    assert len(remaining) == 0


# ── AMemBaseline ─────────────────────────────────────────────────────────────

def test_a_mem_ingest_and_retrieve(mock_llm):
    baseline = AMemBaseline(llm_client=mock_llm)
    baseline.ingest_text_turn("user has a golden retriever named Max")
    baseline.ingest_text_turn("user likes hiking on weekends")

    results = baseline.retrieve("dog", top_k=5)
    assert len(results) > 0


def test_a_mem_links_similar_notes(mock_llm):
    baseline = AMemBaseline(llm_client=mock_llm)
    baseline.ingest_text_turn("user has a golden retriever")
    baseline.ingest_text_turn("user's dog is named Max")
    baseline.ingest_text_turn("user likes hiking in the mountains")

    results = baseline.retrieve("tell me about the dog", top_k=10)
    contents = [r["content"].lower() for r in results]
    assert any("retriever" in c for c in contents)
    assert any("max" in c for c in contents)


def test_a_mem_no_forgetting(mock_llm):
    baseline = AMemBaseline(llm_client=mock_llm)
    baseline.ingest_text_turn("user has a golden retriever named Max")
    for i in range(50):
        baseline.ingest_text_turn(f"Neutral filler turn number {i}.")

    all_memories = baseline.store.get_all()
    assert any("max" in m["content"].lower() for m in all_memories)
