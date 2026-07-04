import json
import numpy as np

# Emotion labels the system recognises
VALID_LABELS = {"neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust"}

# Deliberately simple, minimal few-shot set. Confirmed on a real run that a
# 0.6B model can't reliably execute nuanced conditional extraction rules
# regardless of wording -- three rounds of increasingly elaborate prompts
# each fixed one failure mode (junk pollution, then general under-extraction,
# then emotional-turn under-extraction) while introducing another, all while
# greedy decoding collapsed to one canned answer across wildly different
# inputs. Qwen3-4B got every test case right on this exact simple prompt,
# with no extra scaffolding needed -- the reliability problem was model
# capacity, not prompt engineering. This prompt is validated against
# Qwen3-4B specifically; a smaller model may need the extra scaffolding
# again (see BUGS.md for the full trace of what didn't work and why).
EXTRACTION_PROMPT = """\
Extract any noteworthy facts, preferences, plans, or events mentioned in this turn.
If it's just a greeting, thanks, or acknowledgment with nothing else in it, return an empty array.

Return ONLY a JSON array — no prose, no markdown fences.
Schema: [{{"content": "...", "entity": "...", "type": "fact|preference|event|person"}}]

Example input: Thanks, Mel!
Example output: []

Example input: My best friend told me she's moving across the country permanently.
Example output: [{{"content": "Best friend is moving across the country permanently", "entity": "user", "type": "event"}}]

Turn: {turn}"""


EXTRACTION_SYSTEM_PROMPT = "You are a memory extraction assistant. Output only valid JSON."

# A few short JSON fragments never need anywhere near 512 tokens -- this call
# runs once per conversation turn, so on models that don't emit a stop token
# quickly (e.g. residual "thinking" behavior even with enable_thinking=False),
# a generous cap here is the single biggest per-turn latency cost in the
# whole ingestion pipeline.
EXTRACTION_MAX_TOKENS = 200


def _parse_extraction_response(raw: str) -> list[dict]:
    try:
        # Strip any accidental markdown fences the model may add
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        memories = json.loads(cleaned)
        if not isinstance(memories, list):
            return []
        valid_types = {"fact", "preference", "event", "person"}
        cleaned_memories = []
        for m in memories:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                # LLM occasionally omits/misnames the content field on real
                # (non-mocked) inference; drop these rather than storing a
                # dict that will blow up every downstream m["content"] access.
                continue
            if len(content.split()) < 2:
                # On a low-content turn (a greeting, a short reply), a real
                # LLM sometimes lazily emits content that's just the
                # speaker/subject's bare name ("Caroline") instead of
                # correctly returning nothing -- confirmed directly on a
                # real run: a "person"-type memory with content="Caroline"
                # never decays (semantic store), and a bare proper noun is a
                # near-perfect cosine match for any query mentioning that
                # same name, so these accumulate over a long conversation
                # and systematically crowd out genuinely informative facts
                # about the same person under pure-relevance ranking. A
                # single bare word can't be a self-contained "fact,
                # preference, or event" per the extraction prompt's own
                # definition, so require at least two.
                continue
            if m.get("type") not in valid_types:
                m["type"] = "fact"
            cleaned_memories.append(m)
        return cleaned_memories
    except (json.JSONDecodeError, TypeError):
        return []


def extract_memories(turn_text: str, llm_client) -> list[dict]:
    """Call the LLM to pull discrete memory fragments from a single turn."""
    prompt = EXTRACTION_PROMPT.format(turn=turn_text)
    raw = llm_client.generate(
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=EXTRACTION_MAX_TOKENS,
    )
    return _parse_extraction_response(raw)


def extract_memories_batch(turn_texts: list[str], llm_client) -> list[list[dict]]:
    """
    Batched version of extract_memories -- one forward pass covering many
    turns instead of one generate() call each. Each turn's extraction is
    independent of every other turn's, so this only changes wall-clock cost,
    not the result: same prompt/parsing per item, just issued as a single
    batch via llm_client.generate_batch().
    """
    if not turn_texts:
        return []
    messages_list = [
        [{"role": "user", "content": EXTRACTION_PROMPT.format(turn=t)}] for t in turn_texts
    ]
    raws = llm_client.generate_batch(
        system=EXTRACTION_SYSTEM_PROMPT,
        messages_list=messages_list,
        max_tokens=EXTRACTION_MAX_TOKENS,
    )
    return [_parse_extraction_response(raw) for raw in raws]


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
