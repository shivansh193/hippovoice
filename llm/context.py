BASE_COMPANION_PROMPT = """\
You are a warm, attentive voice companion. You remember details about the person
you speak with and refer to them naturally — never recite a list of facts.
Respond conversationally in 2–4 sentences unless asked something specific."""

# Rough token budget for memory block (leaves room for the conversation itself)
MAX_MEMORY_TOKENS = 800


def build_system_prompt(
    retrieved_memories: list[dict],
    base_prompt: str = BASE_COMPANION_PROMPT,
    tokenizer=None,
) -> str:
    """
    Inject retrieved memories into the system prompt, ordered by salience descending.

    If a tokenizer is supplied, trims the memory block so the total stays under
    MAX_MEMORY_TOKENS. Without a tokenizer, limits by character count (≈ 4 chars/token).
    """
    if not retrieved_memories:
        return base_prompt

    sorted_memories = sorted(
        retrieved_memories,
        key=lambda m: m.get("current_salience", 0.0),
        reverse=True,
    )

    lines = []
    char_budget = MAX_MEMORY_TOKENS * 4  # rough char estimate

    for m in sorted_memories:
        content = m.get("content", "").strip()
        if not content:
            continue
        line = f"- {content}"
        if tokenizer is not None:
            # Precise token counting when tokenizer is available
            used = len(tokenizer.encode("\n".join(lines + [line])))
            if used > MAX_MEMORY_TOKENS:
                break
        else:
            if sum(len(l) for l in lines) + len(line) > char_budget:
                break
        lines.append(line)

    if not lines:
        return base_prompt

    memory_block = "\n".join(lines)
    return (
        f"{base_prompt}\n\n"
        f"## What you remember about this person:\n"
        f"{memory_block}\n\n"
        f"Use these naturally in conversation. Do not recite them verbatim."
    )
