from memory.scorer import compute_salience, DEFAULT_DECAY_LAMBDA

COMPRESS_THRESHOLD = 0.25
FORGET_THRESHOLD = 0.08


def apply_forgetting_cycle(
    memories: list[dict],
    current_turn: int,
    llm_client=None,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
) -> tuple[list[dict], list[dict]]:
    """
    Partition memories into active and forgotten based on current salience.

    Salience < FORGET_THRESHOLD   → forgotten (removed from store)
    Salience < COMPRESS_THRESHOLD → compressed (merged into one summary entry)
    Otherwise                     → active (unchanged)

    `decay_lambda` defaults to the value tuned for ~90-100 turn conversations
    (see memory/scorer.py). Confirmed on a real LoCoMo run that this default
    is wrong for 369-663 turn conversations: even a strongly emotion-boosted
    memory crosses FORGET_THRESHOLD by ~turn 173, so almost every episodic
    memory is physically deleted well before a question about it is ever
    asked, independent of extraction or retrieval quality. Callers ingesting
    much longer conversations should pass a smaller decay_lambda explicitly
    rather than relying on this default.

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
            decay_lambda=decay_lambda,
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
    contents = [m.get("content", "") for m in memories]

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
    # Inherit the oldest source memory's turn_created rather than stamping
    # "now". Resetting the clock here would let a batch of long-decayed,
    # low-value noise reappear as if freshly created, letting it outrank
    # properly-aged high-salience memories on pure recency at the next
    # retrieval -- consolidation should preserve age, not erase it.
    earliest_turn = min(
        (m.get("turn_created", current_turn) for m in memories), default=current_turn
    )

    return {
        "content": summary,
        "entity": "compressed",
        "type": "fact",
        "emotion": {"label": "neutral", "intensity": round(avg_intensity, 3)},
        "base_weight": 0.5,
        "recall_count": 0,
        "turn_created": earliest_turn,
        "current_salience": COMPRESS_THRESHOLD,
        "compressed_from": len(memories),
    }
