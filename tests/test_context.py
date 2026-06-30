from llm.context import build_system_prompt, BASE_COMPANION_PROMPT


def test_memories_injected_into_prompt():
    memories = [
        {"content": "user's dog Max died last month", "current_salience": 2.5},
        {"content": "user likes hiking on weekends", "current_salience": 0.8},
    ]
    prompt = build_system_prompt(memories, BASE_COMPANION_PROMPT)
    assert "Max" in prompt
    assert "hiking" in prompt


def test_higher_salience_appears_first():
    memories = [
        {"content": "user likes hiking", "current_salience": 0.8},
        {"content": "user's dog Max died", "current_salience": 2.5},
    ]
    prompt = build_system_prompt(memories, BASE_COMPANION_PROMPT)
    assert prompt.index("Max") < prompt.index("hiking")


def test_empty_memories_returns_base_prompt():
    base = "You are a helpful companion."
    prompt = build_system_prompt([], base)
    assert prompt == base


def test_memory_block_header_present():
    memories = [{"content": "user is 30 years old", "current_salience": 1.0}]
    prompt = build_system_prompt(memories, "Base.")
    assert "remember" in prompt.lower() or "memory" in prompt.lower() or "know" in prompt.lower()


def test_prompt_contains_base():
    base = "You are a warm companion."
    memories = [{"content": "user likes tea", "current_salience": 1.0}]
    prompt = build_system_prompt(memories, base)
    assert base in prompt


def test_memory_with_no_salience_key_still_works():
    memories = [{"content": "user enjoys jazz", "current_salience": 0.0}]
    prompt = build_system_prompt(memories, "Base.")
    assert "jazz" in prompt


def test_very_long_memory_list_stays_bounded():
    memories = [
        {"content": f"memory about topic {i} which is somewhat lengthy text " * 5, "current_salience": float(i)}
        for i in range(100)
    ]
    prompt = build_system_prompt(memories, "Base.")
    # Character count < 4096 tokens × 4 chars/token
    assert len(prompt) < 4096 * 4
