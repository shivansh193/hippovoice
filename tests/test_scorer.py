import math
import pytest
from memory.scorer import compute_salience, EMOTION_MULTIPLIERS


def test_fear_high_intensity_no_decay():
    score = compute_salience(1.0, {"label": "fear", "intensity": 0.95}, 0, 0)
    # 1.0 × 1.8 × (1+0.95) × 1.0 × e^0 = 1.8 × 1.95 = 3.51
    assert score > 3.0, f"Expected > 3.0, got {score:.4f}"


def test_neutral_decays_below_compress_threshold():
    score = compute_salience(1.0, {"label": "neutral", "intensity": 0.01}, 0, 45)
    assert score < 0.25, f"Expected < 0.25 (compress threshold), got {score:.4f}"


def test_neutral_decays_below_forget_threshold():
    score = compute_salience(1.0, {"label": "neutral", "intensity": 0.01}, 0, 80)
    assert score < 0.08, f"Expected < 0.08 (forget threshold), got {score:.4f}"


def test_fear_survives_45_turns():
    score = compute_salience(1.0, {"label": "fear", "intensity": 0.9}, 0, 45)
    assert score > 0.25, f"Fear memory should survive 45 turns, got {score:.4f}"


def test_recall_boosts_salience():
    low = compute_salience(1.0, {"label": "joy", "intensity": 0.5}, recall_count=0, turns_elapsed=10)
    high = compute_salience(1.0, {"label": "joy", "intensity": 0.5}, recall_count=5, turns_elapsed=10)
    assert high > low, f"Recall should boost salience: {low:.4f} → {high:.4f}"


def test_decay_is_exponential():
    s10 = compute_salience(1.0, {"label": "neutral", "intensity": 0.5}, 0, 10)
    s20 = compute_salience(1.0, {"label": "neutral", "intensity": 0.5}, 0, 20)
    s30 = compute_salience(1.0, {"label": "neutral", "intensity": 0.5}, 0, 30)
    ratio_1 = s10 / s20
    ratio_2 = s20 / s30
    assert abs(ratio_1 - ratio_2) < 0.15, (
        f"Ratios differ too much: {ratio_1:.4f} vs {ratio_2:.4f} — decay may not be exponential"
    )


def test_all_emotion_labels_accepted():
    labels = list(EMOTION_MULTIPLIERS.keys())
    for label in labels:
        score = compute_salience(1.0, {"label": label, "intensity": 0.5}, 0, 0)
        assert score > 0, f"Label '{label}' produced zero/negative score"


def test_unknown_label_defaults_to_neutral():
    score_unknown = compute_salience(1.0, {"label": "boredom", "intensity": 0.5}, 0, 0)
    score_neutral = compute_salience(1.0, {"label": "neutral", "intensity": 0.5}, 0, 0)
    assert score_unknown == score_neutral


def test_intensity_zero_still_positive():
    score = compute_salience(1.0, {"label": "fear", "intensity": 0.0}, 0, 0)
    assert score > 0


def test_salience_non_negative():
    score = compute_salience(1.0, {"label": "neutral", "intensity": 0.0}, 0, 1000)
    assert score >= 0
