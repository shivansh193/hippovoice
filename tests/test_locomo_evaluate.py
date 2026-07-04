from benchmarks.locomo.evaluate import (
    _answer_matches,
    build_qa_context,
    _flatten_conversation,
    rescore_details,
    _current_commit_hash,
    debug_extraction_for_turns,
)


def test_exact_substring_match():
    assert _answer_matches("the answer is transgender woman for sure", "transgender woman")


def test_fuzzy_match_all_words_present():
    assert _answer_matches("caroline researched adoption agencies last year", "adoption agencies")


def test_trailing_punctuation_does_not_break_fuzzy_match():
    # Regression: "adoption." (with trailing period) must still match "adoption"
    assert _answer_matches("caroline researched adoption agencies.", "adoption agencies")


def test_fuzzy_match_below_threshold_fails():
    assert not _answer_matches("melanie painted a sunrise in the morning.", "2022")


def test_partial_overlap_below_threshold_fails():
    # Only 1 of 3 gold words present -- below the 0.7 overlap threshold
    assert not _answer_matches("it happened in 2023 sometime", "7 may 2023")


def test_empty_gold_is_trivial_substring_match():
    # "" is a substring of everything in Python -- this is harmless in
    # practice since run_locomo already skips QA pairs with an empty gold
    # answer before ever calling _answer_matches.
    assert _answer_matches("anything at all", "")


def test_build_qa_context_joins_contents():
    memories = [{"content": "fact one"}, {"content": "fact two"}]
    context = build_qa_context(memories)
    assert "fact one" in context
    assert "fact two" in context


def test_flatten_conversation_orders_sessions_numerically():
    conv = {
        "session_2": [{"speaker": "A", "text": "second"}],
        "session_10": [{"speaker": "A", "text": "tenth"}],
        "session_1": [{"speaker": "A", "text": "first"}],
    }
    turns = _flatten_conversation(conv)
    assert turns == ["A: first", "A: second", "A: tenth"]


def test_flatten_conversation_prefixes_each_turn_with_session_date():
    # Regression: relative date language ("yesterday", "last Saturday") in a
    # turn is only resolvable against its session's actual calendar date --
    # previously discarded entirely during flattening.
    conv = {
        "session_1_date_time": "8 May, 2023",
        "session_1": [{"speaker": "Caroline", "text": "I went to a support group yesterday."}],
    }
    turns = _flatten_conversation(conv)
    assert turns == ["[8 May, 2023] Caroline: I went to a support group yesterday."]


def test_flatten_conversation_handles_missing_date():
    conv = {"session_1": [{"speaker": "A", "text": "no date available"}]}
    turns = _flatten_conversation(conv)
    assert turns == ["A: no date available"]


# ── rescore_details ────────────────────────────────────────────────────────────

def test_rescore_details_recomputes_accuracy_without_rerunning():
    details = [
        {"question": "q1", "gold": "adoption agencies", "predicted": "researched adoption.", "correct": False},
        {"question": "q2", "gold": "2022", "predicted": "no year mentioned here", "correct": False},
    ]
    result = rescore_details(details)
    assert result["total"] == 2
    # q1 now flips: 1/2 gold words present is still < 0.7, so it stays False --
    # this checks rescoring runs the real matcher, not that it always improves.
    assert result["details"][0]["correct"] == _answer_matches("researched adoption.", "adoption agencies")
    assert result["details"][1]["correct"] is False
    assert result["correct"] == sum(d["correct"] for d in result["details"])


def test_rescore_details_empty_list():
    result = rescore_details([])
    assert result == {"accuracy": 0.0, "total": 0, "correct": 0, "details": []}


def test_rescore_details_flips_previously_wrong_answer():
    # Full gold phrase now present verbatim -- should flip to correct.
    details = [
        {"question": "q", "gold": "transgender woman", "predicted": "she is a transgender woman", "correct": False},
    ]
    result = rescore_details(details)
    assert result["details"][0]["correct"] is True
    assert result["correct"] == 1


# ── checkpoint fingerprint / commit hash ────────────────────────────────────────

def test_current_commit_hash_returns_nonempty_string_in_this_repo():
    # This test file lives inside the hippovoice git repo, so a real commit
    # hash should always be resolvable here (falls back to "unknown" only
    # outside a git checkout).
    commit = _current_commit_hash()
    assert isinstance(commit, str)
    assert commit != ""


# ── debug_extraction_for_turns ──────────────────────────────────────────────────

def _make_extraction_llm():
    import json
    from unittest.mock import MagicMock

    mock = MagicMock()

    def per_turn(system, messages, max_tokens=512):
        user_content = messages[-1]["content"] if messages else ""
        turn_text = user_content.split("Turn: ", 1)[-1].strip()
        return json.dumps([{"content": turn_text, "entity": "unknown", "type": "event"}])

    mock.generate.side_effect = per_turn
    mock.generate_batch.side_effect = lambda system, messages_list, max_tokens=512: [
        per_turn(system, m, max_tokens) for m in messages_list
    ]
    return mock


def test_debug_extraction_for_turns_finds_turn_by_dia_id_anywhere_in_conversation():
    # D2:14 is in session 2, not session 1 -- this must work by id lookup,
    # not by assuming the target is near the start of the flattened list.
    conv = {
        "conversation": {
            "session_1_date_time": "8 May, 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "hello"},
            ],
            "session_2_date_time": "25 May, 2023",
            "session_2": [
                {"speaker": "Caroline", "dia_id": "D2:14", "text": "it'll be tough as a single parent"},
            ],
        }
    }
    results = debug_extraction_for_turns(conv, ["D2:14"], _make_extraction_llm())
    assert len(results) == 1
    assert results[0]["dia_id"] == "D2:14"
    assert "[25 May, 2023]" in results[0]["turn"]
    assert "single parent" in results[0]["turn"]
    assert results[0]["extracted"][0]["content"] == results[0]["turn"]


def test_debug_extraction_for_turns_skips_missing_dia_ids():
    conv = {
        "conversation": {
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}],
        }
    }
    results = debug_extraction_for_turns(conv, ["D1:1", "D9:99"], _make_extraction_llm())
    assert len(results) == 1
    assert results[0]["dia_id"] == "D1:1"


def test_debug_extraction_for_turns_empty_when_no_ids_found():
    conv = {"conversation": {"session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}]}}
    assert debug_extraction_for_turns(conv, ["D9:99"], _make_extraction_llm()) == []
