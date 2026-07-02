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
    salience = base_weight × emotion_multiplier × intensity_factor × recall_boost × e^(−λ_eff·turns)

    emotion_multiplier: label-based weight from EMOTION_MULTIPLIERS
    intensity_factor:   scales 1.0–2.0 based on emotion["intensity"] in [0, 1]
    recall_boost:       each past retrieval adds 0.3 to the multiplier
    λ_eff:              decay_lambda / (emotion_multiplier × intensity_factor)

    Emotionally salient memories don't just start higher, they decay slower —
    this mirrors amygdala-mediated consolidation of emotional memories (see
    Kensinger 2009, McGaugh 2004: "flashbulb" memories are consolidated more
    strongly, not just recalled more vividly at time of encoding). Without
    this, a uniform λ means a highly emotional memory from 40 turns ago can
    lose a retrieval competition to a completely mundane memory from 2 turns
    ago purely on recency, which defeats the point of salience-weighted
    retrieval over plain similarity search.
    """
    em = EMOTION_MULTIPLIERS.get(emotion.get("label", "neutral"), 1.0)
    intensity_factor = 1.0 + float(emotion.get("intensity", 0.0))
    recall_boost = 1.0 + (0.3 * recall_count)
    salience_weight = em * intensity_factor
    effective_lambda = decay_lambda / salience_weight
    decay = math.exp(-effective_lambda * turns_elapsed)
    return base_weight * salience_weight * recall_boost * decay
