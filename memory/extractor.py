import json
import numpy as np

# Emotion labels the system recognises
VALID_LABELS = {"neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust"}

EXTRACTION_PROMPT = """\
Extract distinct, self-contained memory fragments from this conversation turn.
Each fragment must be a single fact, preference, or event worth remembering long-term.
Return ONLY a JSON array — no prose, no markdown fences.

Schema: [{{"content": "...", "entity": "...", "type": "fact|preference|event|person"}}]

Turn: {turn}"""


def extract_memories(turn_text: str, llm_client) -> list[dict]:
    """Call the LLM to pull discrete memory fragments from a single turn."""
    prompt = EXTRACTION_PROMPT.format(turn=turn_text)
    raw = llm_client.generate(
        system="You are a memory extraction assistant. Output only valid JSON.",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    try:
        # Strip any accidental markdown fences the model may add
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        memories = json.loads(cleaned)
        if not isinstance(memories, list):
            return []
        valid_types = {"fact", "preference", "event", "person"}
        for m in memories:
            if m.get("type") not in valid_types:
                m["type"] = "fact"
        return memories
    except (json.JSONDecodeError, TypeError):
        return []


def tag_emotion_text(text: str) -> dict:
    """
    Classify emotion from text using a lightweight zero-shot approach.

    Returns {"label": str, "intensity": float}.

    Uses a simple keyword heuristic for now; swap in distilroberta-base-emotion
    or an LLM call when running on GPU.
    """
    text_lower = text.lower()

    keyword_map = {
        "fear":     ["terrified", "scared", "afraid", "fear", "terrifying", "horror", "panic"],
        "sadness":  ["sad", "devastated", "heartbroken", "grief", "crying", "miss", "lost", "died", "death", "mourning"],
        "anger":    ["angry", "furious", "rage", "hate", "mad", "outraged", "infuriated"],
        "joy":      ["excited", "happy", "thrilled", "wonderful", "amazing", "love", "great", "fantastic", "promotion"],
        "surprise": ["surprised", "shocked", "unbelievable", "unexpected", "wow", "sudden"],
        "disgust":  ["disgusting", "gross", "revolting", "nasty", "awful", "horrible"],
    }

    intensity_boosters = ["so", "very", "extremely", "really", "incredibly", "absolutely", "deeply"]
    booster_present = any(b in text_lower for b in intensity_boosters)

    best_label = "neutral"
    best_score = 0

    for label, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == "neutral":
        intensity = 0.1 + (0.1 if booster_present else 0.0)
    else:
        base_intensity = min(0.4 + (best_score * 0.15), 0.95)
        intensity = min(base_intensity + (0.1 if booster_present else 0.0), 1.0)

    return {"label": best_label, "intensity": round(intensity, 3)}


def tag_emotion_audio(text: str, audio_embedding: np.ndarray) -> dict:
    """
    Fuse text-based emotion tag with prosodic signal from the audio embedding.

    High variance in the encoder embedding is a proxy for emotional speech energy.
    If text reads as neutral but the audio is energetic, boost the intensity.
    """
    text_emotion = tag_emotion_text(text)

    audio_std = float(np.std(audio_embedding))
    # Empirically, encoder embeddings for calm speech cluster around std ~1.0–2.0;
    # emotionally charged speech pushes to ~3.0+.
    normalized_signal = min(audio_std / 5.0, 1.0)

    if text_emotion["label"] == "neutral" and normalized_signal > 0.4:
        text_emotion["intensity"] = round(min(text_emotion["intensity"] + 0.2, 1.0), 3)
        text_emotion["prosody_boosted"] = True

    return text_emotion


def extract_turn(turn_text: str, audio_embedding: np.ndarray, llm_client) -> list[dict]:
    """
    Full extraction for a single conversation turn.

    Returns a list of memory dicts, each carrying:
      content, entity, type, emotion (label + intensity)
    """
    memories = extract_memories(turn_text, llm_client)
    emotion = tag_emotion_audio(turn_text, audio_embedding)
    for m in memories:
        m["emotion"] = emotion
    return memories
