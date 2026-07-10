import pytest

from benchmarks.locomo.evaluate import (
    normalize_answer,
    f1_score,
    multi_hop_f1,
    score_answer,
    bin_f1_scores,
    build_qa_context,
    _flatten_conversation,
    rescore_details,
    _current_commit_hash,
    debug_extraction_for_turns,
    print_qa_trace,
    run_locomo,
    MULTI_HOP_CATEGORY,
    ADVERSARIAL_CATEGORY,
)


# ── hand-verified ground truth ───────────────────────────────────────────────────
# These four values were hand-derived against LoCoMo's actual published
# evaluation.py source (fetched and quoted verbatim, not approximated) before
# writing any code here -- per the principle that a reimplementation is only
# trustworthy once validated against known-correct values, not just "looks
# right". If the coded scorer doesn't reproduce these exactly, the scorer is
# wrong and every downstream conclusion drawn from it is contaminated.

def test_hand_verified_partial_credit_missing_specific_noun():
    # "researched adoption" vs "adoption agencies" -- category 1 (multi-hop),
    # but neither string contains a comma so multi_hop_f1 reduces to plain
    # f1_score. Overlap = {adopt} (stemmed) -> precision=1/2, recall=1/2, F1=0.5.
    assert multi_hop_f1("researched adoption", "adoption agencies") == 0.5


def test_hand_verified_multi_hop_credits_matched_subanswer_only():
    # gold "Kickboxing, Taekwondo" vs predicted "yoga, kickboxing, and
    # circuit training". Splits to predictions=[yoga, kickboxing, and
    # circuit training] and ground_truths=[Kickboxing, Taekwondo].
    # "Kickboxing" sub-answer: max F1 against all 3 prediction fragments ->
    #   f1_score("kickboxing","Kickboxing")=1.0 (exact after normalize+stem)
    #   -- the other two garbage fragments simply don't hurt this pairing.
    # "Taekwondo" sub-answer: no fragment matches -> 0.
    # mean(1.0, 0) = 0.5 -- NOT 0.33, which was an earlier hand-estimate
    # made before fetching LoCoMo's actual verbatim multi-hop algorithm;
    # this test locks in the corrected, verified value.
    assert multi_hop_f1("yoga, kickboxing, and circuit training", "Kickboxing, Taekwondo") == 0.5


def test_hand_verified_partial_credit_from_shared_pronoun():
    # "her family" vs "her mother" -- category 2 (non-multi-hop), plain
    # f1_score. Overlap = {her} only -> precision=1/2, recall=1/2, F1=0.5.
    # (Credit here comes from a shared pronoun, not real content overlap --
    # illustrates the metric isn't free of its own noise, just less blind
    # than a boolean matcher.)
    assert f1_score("her family", "her mother") == 0.5


def test_hand_verified_zero_overlap_stays_zero():
    # An honest "not in context" hedge shares no lexical content with the
    # gold answer at all -- zero overlap, zero F1, same result a boolean
    # matcher would give. The metric fix doesn't manufacture credit where
    # none exists.
    assert f1_score("caroline's identity is not provided in the context.", "transgender woman") == 0.0


# ── normalize_answer ──────────────────────────────────────────────────────────

def test_normalize_answer_strips_commas_articles_and_punctuation():
    assert normalize_answer("The Adoption, Agencies.") == "adoption agencies"


def test_normalize_answer_removes_and_as_well_as_articles():
    assert normalize_answer("yoga, kickboxing, and circuit training") == "yoga kickboxing circuit training"


# ── f1_score ──────────────────────────────────────────────────────────────────

def test_f1_score_exact_match_after_normalization():
    assert f1_score("Kickboxing", "kickboxing.") == 1.0


def test_f1_score_no_overlap_is_zero():
    assert f1_score("completely unrelated text", "sweden") == 0.0


# ── multi_hop_f1 ──────────────────────────────────────────────────────────────

def test_multi_hop_f1_reduces_to_f1_score_for_non_comma_answers():
    pred, gold = "she moved from sweden", "sweden"
    assert multi_hop_f1(pred, gold) == f1_score(pred, gold)


# ── score_answer (category dispatch) ─────────────────────────────────────────

def test_score_answer_uses_multi_hop_for_category_1():
    f1 = score_answer("yoga, kickboxing, and circuit training", "Kickboxing, Taekwondo", MULTI_HOP_CATEGORY)
    assert f1 == 0.5


def test_score_answer_does_not_split_commas_in_dates_for_other_categories():
    # "19 January, 2023" must NOT be treated as two sub-answers ["19
    # January", "2023"] for a non-multi-hop category -- that comma is
    # date punctuation, not a list separator.
    f1 = score_answer("19 january 2023", "19 January, 2023", category=2)
    assert f1 == 1.0  # exact match once the comma is just stripped, not split on


def test_score_answer_adversarial_category_checks_for_hedge_phrases():
    assert score_answer("there is no information available", "adversarial answer", ADVERSARIAL_CATEGORY) == 1.0
    assert score_answer("i think it is x", "adversarial answer", ADVERSARIAL_CATEGORY) == 0.0


# ── bin_f1_scores ─────────────────────────────────────────────────────────────

def test_bin_f1_scores_buckets_correctly():
    details = [
        {"f1": 0.0}, {"f1": 0.15},
        {"f1": 0.2}, {"f1": 0.5}, {"f1": 0.69},
        {"f1": 0.7}, {"f1": 1.0},
    ]
    bins = bin_f1_scores(details)
    assert bins == {"near_zero": 2, "partial": 3, "high": 2}


# ── build_qa_context / _flatten_conversation ─────────────────────────────────

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

def test_rescore_details_recomputes_f1_without_rerunning():
    details = [
        {"question": "q1", "gold": "adoption agencies", "predicted": "researched adoption.",
         "category": MULTI_HOP_CATEGORY},
        {"question": "q2", "gold": "2022", "predicted": "no year mentioned here", "category": 2},
    ]
    result = rescore_details(details)
    assert result["total"] == 2
    assert result["details"][0]["f1"] == 0.5
    assert result["details"][1]["f1"] == 0.0
    assert result["total_f1"] == 0.5
    assert result["avg_f1"] == 0.25
    assert result["bins"] == {"near_zero": 1, "partial": 1, "high": 0}


def test_rescore_details_empty_list():
    result = rescore_details([])
    assert result == {
        "avg_f1": 0.0, "total": 0, "total_f1": 0.0,
        "bins": {"near_zero": 0, "partial": 0, "high": 0}, "details": [],
    }


def test_rescore_details_backfills_missing_category_from_dataset():
    # Old checkpoint format has no "category" field at all -- must look it
    # up from the real dataset by matching question text, rather than
    # crashing or silently mis-scoring a multi-hop question as non-multi-hop.
    details = [
        {"question": "What did Caroline research?", "gold": "adoption agencies",
         "predicted": "researched adoption."},
    ]
    result = rescore_details(details)
    assert result["details"][0]["category"] == MULTI_HOP_CATEGORY
    assert result["details"][0]["f1"] == 0.5


# ── checkpoint fingerprint / commit hash ────────────────────────────────────────

def test_current_commit_hash_returns_nonempty_string_in_this_repo():
    # This test file lives inside the hippovoice git repo, so a real commit
    # hash should always be resolvable here (falls back to "unknown" only
    # outside a git checkout).
    commit = _current_commit_hash()
    assert isinstance(commit, str)
    assert commit != ""


# ── pipeline_factory (fair comparison against baselines) ─────────────────────────
# run_locomo() used to hardcode HippoVoicePipeline construction per conversation,
# meaning baselines (Mem0Baseline/AMemBaseline/NaiveRAG) could never be run
# through the real, F1-scored LoCoMo QA benchmark -- only the separate
# synthetic signal/noise one. pipeline_factory generalizes this so any
# baseline can be evaluated under the identical LLM/data/scoring.

def test_custom_pipeline_factory_requires_system_name():
    with pytest.raises(ValueError):
        run_locomo(llm_client=object(), pipeline_factory=lambda llm: None)


def test_system_name_requires_custom_pipeline_factory():
    with pytest.raises(ValueError):
        run_locomo(llm_client=object(), system_name="not really custom")


def test_pipeline_factory_is_used_instead_of_hippovoice(monkeypatch):
    import benchmarks.locomo.evaluate as evaluate_mod

    fake_conv = {
        "conversation": {
            "session_1_date_time": "1 May, 2023",
            "session_1": [{"dia_id": "D1:1", "speaker": "Alex", "text": "I got a new job."}],
        },
        "qa": [{"question": "What did Alex get?", "answer": "a new job", "category": 2}],
    }
    monkeypatch.setattr(evaluate_mod, "load_locomo", lambda data_path=None: [fake_conv])

    constructed = []

    class FakePipeline:
        def __init__(self, llm):
            self.llm = llm
            constructed.append(self)

        def ingest_text_turn(self, text):
            pass

        def retrieve(self, query, top_k=5):
            return []

    llm = _make_extraction_llm()
    llm.generate.side_effect = None
    llm.generate.return_value = "a new job"

    result = evaluate_mod.run_locomo(
        llm_client=llm,
        num_conversations=1,
        pipeline_factory=lambda llm_client: FakePipeline(llm_client),
        system_name="FakeSystem",
        verbose=False,
    )

    assert len(constructed) == 1 and isinstance(constructed[0], FakePipeline), (
        "pipeline_factory should be used for the per-conversation pipeline "
        "instead of hardcoding HippoVoicePipeline"
    )
    assert result["total"] == 1


def test_fingerprint_records_system_name(monkeypatch, tmp_path):
    import benchmarks.locomo.evaluate as evaluate_mod

    fake_conv = {
        "conversation": {
            "session_1_date_time": "1 May, 2023",
            "session_1": [{"dia_id": "D1:1", "speaker": "Alex", "text": "I got a new job."}],
        },
        "qa": [{"question": "What did Alex get?", "answer": "a new job", "category": 2}],
    }
    monkeypatch.setattr(evaluate_mod, "load_locomo", lambda data_path=None: [fake_conv])

    class FakePipeline:
        def __init__(self, llm):
            self.llm = llm

        def ingest_text_turn(self, text):
            pass

        def retrieve(self, query, top_k=5):
            return []

    llm = _make_extraction_llm()
    llm.generate.side_effect = None
    llm.generate.return_value = "a new job"
    # Bare MagicMock attribute access auto-creates a nested MagicMock, which
    # isn't JSON serializable -- the fingerprint dict includes these two
    # fields, so they need real (string) values before the checkpoint write.
    llm.model_name = "mock-model"
    llm._backend = "mock"

    checkpoint_path = str(tmp_path / "checkpoint.json")
    evaluate_mod.run_locomo(
        llm_client=llm,
        num_conversations=1,
        pipeline_factory=lambda llm_client: FakePipeline(llm_client),
        system_name="Mem0-style",
        checkpoint_path=checkpoint_path,
        verbose=False,
    )

    import json
    with open(checkpoint_path) as f:
        state = json.load(f)
    assert state["fingerprint"]["system_name"] == "Mem0-style", (
        "checkpoints for different systems against the same checkpoint_path "
        "must never be silently confused for each other"
    )


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


# ── print_qa_trace ────────────────────────────────────────────────────────────

def test_print_qa_trace_finds_matching_question_and_prints_context(capsys):
    details = [
        {
            "question": "Which city have both Jean and John visited?",
            "gold": "rome", "predicted": "downtown", "f1": 0.0, "category": 1,
            "context": "- John went to Rome in 2019\n- Jean loves Italian food",
        },
        {"question": "Unrelated question", "gold": "x", "predicted": "y", "f1": 0.0, "category": 2, "context": ""},
    ]
    print_qa_trace(details, "Jean and John")
    out = capsys.readouterr().out
    assert "Which city have both Jean and John visited?" in out
    assert "John went to Rome in 2019" in out
    assert "Unrelated question" not in out


def test_print_qa_trace_no_match_prints_message(capsys):
    details = [{"question": "some question", "gold": "x", "predicted": "y", "f1": 0.0, "category": 2, "context": ""}]
    print_qa_trace(details, "nonexistent")
    out = capsys.readouterr().out
    assert "No question matching" in out


def test_print_qa_trace_handles_missing_context_field_gracefully(capsys):
    # Older checkpoints (written before context logging was added) won't
    # have this key -- must say so explicitly, not crash or print nothing.
    details = [{"question": "old question", "gold": "x", "predicted": "y", "f1": 0.0, "category": 2}]
    print_qa_trace(details, "old question")
    out = capsys.readouterr().out
    assert "context not logged" in out
