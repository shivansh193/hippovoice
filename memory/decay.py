from memory.scorer import compute_salience

COMPRESS_THRESHOLD = 0.25
FORGET_THRESHOLD = 0.08


def apply_forgetting_cycle(
    memories: list[dict],
    current_turn: int,
    llm_client=None,
) -> tuple[list[dict], list[dict]]:
    """
    Partition memories into active and forgotten based on current salience.

    Salience < FORGET_THRESHOLD   → forgotten (removed from store)
    Salience < COMPRESS_THRESHOLD → compressed (merged into one summary entry)
    Otherwise                     → active (unchanged)

    Returns (active_memories, forgotten_memories).
    The returned active list may include a synthetic compressed entry.
    """
    active = []
    compress_candidates = []
    forgotten = []

    for m in memories:
        turns_elapsed = current_turn - m.get("turn_created", 0)
        score = compute_salience(
            base_weight=m.get("base_weight", 1.0),
            emotion=m.get("emotion", {"label": "neutral", "intensity": 0.0}),
            recall_count=m.get("recall_count", 0),
            turns_elapsed=turns_elapsed,
        )
        m = {**m, "current_salience": round(score, 4)}

        if score < FORGET_THRESHOLD:
            forgotten.append(m)
        elif score < COMPRESS_THRESHOLD:
            compress_candidates.append(m)
        else:
            active.append(m)

    if compress_candidates:
        compressed = _compress(compress_candidates, current_turn, llm_client)
        active.append(compressed)

    return active, forgotten


def _compress(memories: list[dict], current_turn: int, llm_client=None) -> dict:
    """
    Merge a list of low-salience memories into a single summary entry.

    If an LLM client is available, ask it for a one-sentence summary.
    Otherwise, concatenate contents separated by '; '.
    """
    contents = [m["content"] for m in memories]

    if llm_client is not None:
        joined = "; ".join(contents)
        summary = llm_client.generate(
            system="Summarise these facts about a person into one concise sentence.",
            messages=[{"role": "user", "content": joined}],
            max_tokens=80,
        ).strip()
    else:
        summary = "; ".join(contents)

    avg_intensity = sum(m.get("emotion", {}).get("intensity", 0.0) for m in memories) / len(memories)

    return {
        "content": summary,
        "entity": "compressed",
        "type": "fact",
        "emotion": {"label": "neutral", "intensity": round(avg_intensity, 3)},
        "base_weight": 0.5,
        "recall_count": 0,
        "turn_created": current_turn,
        "current_salience": COMPRESS_THRESHOLD,
        "compressed_from": len(memories),
    }
