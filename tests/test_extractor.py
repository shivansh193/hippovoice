import numpy as np
import pytest
from memory.extractor import (
    extract_memories,
    tag_emotion_text,
    tag_emotion_audio,
    extract_turn,
)


# ── tag_emotion_text ──────────────────────────────────────────────────────────

def test_joy_detection():
    result = tag_emotion_text("I'm so excited about my promotion!")
    assert result["label"] == "joy"
    assert result["intensity"] > 0.6


def test_neutral_detection():
    result = tag_emotion_text("The weather is okay.")
    assert result["label"] == "neutral"
    assert result["intensity"] < 0.3


def test_sadness_detection():
    result = tag_emotion_text("My dog of twelve years passed away yesterday. I'm heartbroken.")
    assert result["label"] in ["sadness", "fear"]
    assert result["intensity"] > 0.4


def test_fear_detection():
    result = tag_emotion_text("I feel absolutely terrified about the biopsy results.")
    assert result["label"] == "fear"
    assert result["intensity"] > 0.5


def test_anger_detection():
    result = tag_emotion_text("I'm furious about how they treated me.")
    assert result["label"] == "anger"


def test_intensity_in_range():
    for text in ["Hello", "I'm devastated", "The sky is blue", "I'm so happy!"]:
        result = tag_emotion_text(text)
        assert 0.0 <= result["intensity"] <= 1.0, f"Intensity out of range for: '{text}'"


def test_label_is_valid():
    valid = {"neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust"}
    for text in ["Hello", "I'm scared", "I'm thrilled", "That's disgusting"]:
        result = tag_emotion_text(text)
        assert result["label"] in valid


# ── tag_emotion_audio ─────────────────────────────────────────────────────────

def test_prosody_boosts_neutral_intensity():
    flat_emb = np.ones(1280, dtype=np.float32) * 0.5
    loud_emb = np.random.default_rng(1).normal(0, 3.0, 1280).astype(np.float32)

    result_flat = tag_emotion_audio("I see", flat_emb)
    result_loud = tag_emotion_audio("I see", loud_emb)

    assert result_loud["intensity"] >= result_flat["intensity"]


def test_prosody_boost_flag_set_when_neutral_and_energetic():
    loud_emb = np.random.default_rng(2).normal(0, 4.0, 1280).astype(np.float32)
    result = tag_emotion_audio("That's fine I suppose", loud_emb)
    if result.get("prosody_boosted"):
        assert result["intensity"] > 0.2


def test_prosody_does_not_affect_already_emotional_text():
    loud_emb = np.random.default_rng(3).normal(0, 4.0, 1280).astype(np.float32)
    result = tag_emotion_audio("I'm absolutely terrified and scared", loud_emb)
    # Should still be fear, not switched to neutral
    assert result["label"] in ["fear", "sadness", "anger"]


def test_audio_intensity_stays_in_range():
    rng = np.random.default_rng(99)
    for _ in range(10):
        emb = rng.normal(0, rng.uniform(0, 10), 1280).astype(np.float32)
        result = tag_emotion_audio("something neutral", emb)
        assert 0.0 <= result["intensity"] <= 1.0


# ── extract_memories ──────────────────────────────────────────────────────────

def test_extracts_dog_facts(mock_llm):
    memories = extract_memories(
        "I have a golden retriever named Max who loves swimming", mock_llm
    )
    assert len(memories) > 0
    contents = [m["content"].lower() for m in memories]
    assert any("dog" in c or "golden retriever" in c or "retriever" in c for c in contents)
    assert any("max" in c for c in contents)


def test_memory_types_are_valid(mock_llm):
    memories = extract_memories("I love hiking on weekends", mock_llm)
    valid_types = {"fact", "preference", "event", "person"}
    for m in memories:
        assert m.get("type") in valid_types, f"Invalid type: {m.get('type')}"


def test_invalid_llm_response_returns_empty():
    from unittest.mock import MagicMock
    bad_llm = MagicMock()
    bad_llm.generate.return_value = "this is not json at all %%%"
    memories = extract_memories("something", bad_llm)
    assert memories == []


def test_empty_turn_returns_empty_or_list(mock_llm):
    mock_llm.generate.return_value = "[]"
    result = extract_memories("", mock_llm)
    assert isinstance(result, list)


# ── extract_turn ──────────────────────────────────────────────────────────────

def test_fearful_turn_emotion(mock_llm, sad_audio_embedding):
    memories = extract_turn(
        "My dog Max got hit by a car today",
        sad_audio_embedding,
        mock_llm,
    )
    assert len(memories) > 0
    for m in memories:
        assert "emotion" in m
        assert m["emotion"]["label"] in {"fear", "sadness", "anger", "neutral"}
        assert 0.0 <= m["emotion"]["intensity"] <= 1.0


def test_all_memories_carry_emotion(mock_llm, flat_audio_embedding):
    memories = extract_turn(
        "I have a golden retriever named Max who loves swimming",
        flat_audio_embedding,
        mock_llm,
    )
    for m in memories:
        assert "emotion" in m
        assert "label" in m["emotion"]
        assert "intensity" in m["emotion"]
