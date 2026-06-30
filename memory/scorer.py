import math

EMOTION_MULTIPLIERS = {
    "neutral":  1.0,
    "joy":      1.4,
    "sadness":  1.6,
    "fear":     1.8,
    "anger":    1.5,
    "surprise": 1.3,
    "disgust":  1.4,
}

DEFAULT_DECAY_LAMBDA = 0.05


def compute_salience(
    base_weight: float,
    emotion: dict,
    recall_count: int,
    turns_elapsed: int,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
) -> float:
    """
    salience = base_weight × emotion_multiplier × intensity_factor × recall_boost × e^(−λ·turns)

    emotion_multiplier: label-based weight from EMOTION_MULTIPLIERS
    intensity_factor:   scales 1.0–2.0 based on emotion["intensity"] in [0, 1]
    recall_boost:       each past retrieval adds 0.3 to the multiplier
    decay:              exponential with λ=0.05 per conversational turn
    """
    em = EMOTION_MULTIPLIERS.get(emotion.get("label", "neutral"), 1.0)
    intensity_factor = 1.0 + float(emotion.get("intensity", 0.0))
    recall_boost = 1.0 + (0.3 * recall_count)
    decay = math.exp(-decay_lambda * turns_elapsed)
    return base_weight * em * intensity_factor * recall_boost * decay
