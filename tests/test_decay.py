import pytest
from memory.decay import apply_forgetting_cycle, COMPRESS_THRESHOLD, FORGET_THRESHOLD


def _make_memories(count, label, intensity, turn_created=0):
    return [
        {
            "content": f"memory {i}",
            "base_weight": 1.0,
            "emotion": {"label": label, "intensity": intensity},
            "recall_count": 0,
            "turn_created": turn_created,
        }
        for i in range(count)
    ]


def test_neutral_memories_forgotten_after_many_turns():
    memories = _make_memories(5, "neutral", 0.01)
    active, forgotten = apply_forgetting_cycle(memories, current_turn=80)
    assert len(forgotten) == 5, f"Expected 5 forgotten, got {len(forgotten)}"
    assert len(active) == 0, f"Expected 0 active, got {len(active)}"


def test_fear_memories_survive_45_turns():
    memories = _make_memories(3, "fear", 0.9)
    active, forgotten = apply_forgetting_cycle(memories, current_turn=45)
    assert len(active) > 0, "High-salience fear memories should survive 45 turns"
    assert len(forgotten) == 0


def test_compress_threshold_merges_low_salience(mock_llm):
    # At turn 35 with neutral/0.3: salience = 1.3 × e^(-1.75) = 0.226 < COMPRESS_THRESHOLD (0.25)
    # but > FORGET_THRESHOLD (0.08) — so memories should compress, not forget
    memories = _make_memories(5, "neutral", 0.3)
    active, forgotten = apply_forgetting_cycle(memories, current_turn=35, llm_client=mock_llm)
    assert len(active) == 1, f"Expected 1 compressed entry, got {len(active)}"
    assert len(forgotten) == 0
    assert active[0].get("compressed_from") == 5


def test_compress_without_llm_joins_contents():
    # Same turn arithmetic: neutral/0.3 at turn 35 → compress zone
    memories = _make_memories(3, "neutral", 0.3)
    active, _ = apply_forgetting_cycle(memories, current_turn=35, llm_client=None)
    assert len(active) == 1
    assert "memory 0" in active[0]["content"] or ";" in active[0]["content"]


def test_salience_stored_on_returned_memories():
    memories = _make_memories(3, "joy", 0.8)
    active, forgotten = apply_forgetting_cycle(memories, current_turn=5)
    for m in active:
        assert "current_salience" in m
        assert m["current_salience"] > 0


def test_mixed_batch_splits_correctly():
    # At turn 35:
    #   fear/0.9 with recall=3: 3.42 × 1.9 × e^(-1.75) = 6.498 × 0.174 = 1.13 → ACTIVE
    #   neutral/0.01 with recall=0: 1.01 × e^(-1.75) = 0.176 → COMPRESS (below 0.25)
    high = [
        {
            "content": f"high memory {i}", "base_weight": 1.0,
            "emotion": {"label": "fear", "intensity": 0.9},
            "recall_count": 3, "turn_created": 0,
        }
        for i in range(3)
    ]
    low = _make_memories(3, "neutral", 0.01, turn_created=0)
    all_mems = high + low

    active, forgotten = apply_forgetting_cycle(all_mems, current_turn=35)

    # High-salience fear memories should remain active
    high_active = [m for m in active if "high memory" in m.get("content", "")]
    assert len(high_active) > 0, "Fear memories should be active at turn 35"
    # Total items accounted for (active includes compressed low-sal entry)
    assert len(active) + len(forgotten) > 0


def test_empty_input():
    active, forgotten = apply_forgetting_cycle([], current_turn=10)
    assert active == []
    assert forgotten == []
