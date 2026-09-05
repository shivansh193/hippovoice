"""
Tests for the Zep-style baseline reimplementation.

The shared `mock_llm` fixture in conftest.py doesn't know this baseline's
extraction shape (its system prompt contains "extract", which routes into
the generic memory/extractor.py-shaped branch and returns a list, not the
{"entities": [...], "facts": [...]} dict this baseline expects) -- so, like
test_mem0_decision_update_replaces_memory/test_mem0_decision_delete_removes_memory
in test_baselines.py, these use a purpose-built mock instead of the shared one.

CPU-only -- no GPU or real model download needed (MemoryStore's embedder
runs on CPU fine for a handful of short strings).
"""

import json

from unittest.mock import MagicMock

from baselines.zep_baseline import ZepBaseline


def _turn_text(user_content: str) -> str:
    return user_content.split("Turn: ", 1)[-1].strip()


def _extraction_llm(turn_responses: dict[str, dict]) -> MagicMock:
    """
    Builds a mock whose extraction calls return exactly the given
    {"entities": [...], "facts": [...]} dict for a matching turn substring,
    and an empty extraction otherwise.
    """
    llm = MagicMock()

    def side_effect(system, messages, max_tokens=512):
        turn = _turn_text(messages[-1]["content"])
        for substring, response in turn_responses.items():
            if substring in turn:
                return json.dumps(response)
        return json.dumps({"entities": [], "facts": []})

    llm.generate.side_effect = side_effect
    return llm


def test_zep_ingest_and_retrieve():
    llm = _extraction_llm({
        "golden retriever": {
            "entities": [{"name": "Max", "type": "concept"}, {"name": "User", "type": "person"}],
            "facts": [{"subject": "User", "predicate": "has a dog named", "object": "Max", "time": None}],
        },
    })
    baseline = ZepBaseline(llm_client=llm)
    baseline.ingest_text_turn("I have a golden retriever named Max")

    results = baseline.retrieve("tell me about the dog", top_k=5)
    assert len(results) == 1
    assert "max" in results[0]["content"].lower()


def test_zep_entity_resolution_dedupes_same_entity():
    """Two turns mentioning the same entity by name shouldn't create two
    separate entity nodes -- confirmed indirectly via invalidation below,
    which only works at all if both facts share the same resolved subject id."""
    llm = _extraction_llm({
        "lives in Seattle": {
            "entities": [{"name": "Caroline", "type": "person"}],
            "facts": [{"subject": "Caroline", "predicate": "lives in", "object": "Seattle", "time": None}],
        },
        "moved to Portland": {
            "entities": [{"name": "Caroline", "type": "person"}],
            "facts": [{"subject": "Caroline", "predicate": "lives in", "object": "Portland", "time": None}],
        },
    })
    baseline = ZepBaseline(llm_client=llm)
    baseline.ingest_text_turn("Caroline lives in Seattle")
    baseline.ingest_text_turn("Caroline moved to Portland")

    # Only one Caroline entity should ever have been created.
    caroline_ids = [eid for eid, name in baseline._entity_names.items() if name == "Caroline"]
    assert len(caroline_ids) == 1


def test_zep_edge_invalidation_on_contradiction():
    """A new (subject, predicate) fact with a different object supersedes
    the old one -- the old fact should never surface in retrieve() again."""
    llm = _extraction_llm({
        "lives in Seattle": {
            "entities": [{"name": "Caroline", "type": "person"}],
            "facts": [{"subject": "Caroline", "predicate": "lives in", "object": "Seattle", "time": None}],
        },
        "moved to Portland": {
            "entities": [{"name": "Caroline", "type": "person"}],
            "facts": [{"subject": "Caroline", "predicate": "lives in", "object": "Portland", "time": None}],
        },
    })
    baseline = ZepBaseline(llm_client=llm)
    baseline.ingest_text_turn("Caroline lives in Seattle")
    baseline.ingest_text_turn("Caroline moved to Portland")

    live_facts = [f for f in baseline._facts.values() if f["invalid_at"] is None]
    assert len(live_facts) == 1
    assert "portland" in live_facts[0]["content"].lower()

    results = baseline.retrieve("where does Caroline live", top_k=10)
    contents = [r["content"].lower() for r in results]
    assert not any("seattle" in c for c in contents)
    assert any("portland" in c for c in contents)


def test_zep_retrieve_excludes_invalidated_facts_even_with_exact_keyword_match():
    """Even a query that lexically matches the OLD (invalidated) fact's
    wording exactly should not surface it -- retrieve() must filter by
    invalid_at, not just rank everything and hope the new fact wins."""
    llm = _extraction_llm({
        "lives in Seattle": {
            "entities": [{"name": "Caroline", "type": "person"}],
            "facts": [{"subject": "Caroline", "predicate": "lives in", "object": "Seattle", "time": None}],
        },
        "moved to Portland": {
            "entities": [{"name": "Caroline", "type": "person"}],
            "facts": [{"subject": "Caroline", "predicate": "lives in", "object": "Portland", "time": None}],
        },
    })
    baseline = ZepBaseline(llm_client=llm)
    baseline.ingest_text_turn("Caroline lives in Seattle")
    baseline.ingest_text_turn("Caroline moved to Portland")

    results = baseline.retrieve("Caroline lives in Seattle", top_k=10)
    contents = [r["content"].lower() for r in results]
    assert not any("seattle" in c for c in contents)


def test_zep_graph_distance_pulls_in_connected_facts():
    """A fact sharing an entity with the top semantic hit should be pulled
    into results via the graph-distance signal, even if it wouldn't rank
    highly on semantic similarity or BM25 alone."""
    llm = _extraction_llm({
        "golden retriever": {
            "entities": [{"name": "Max", "type": "concept"}, {"name": "Caroline", "type": "person"}],
            "facts": [{"subject": "Caroline", "predicate": "has a dog named", "object": "Max", "time": None}],
        },
        "vet appointment": {
            "entities": [{"name": "Max", "type": "concept"}],
            "facts": [{"subject": "Max", "predicate": "has an appointment at", "object": "the vet", "time": None}],
        },
    })
    baseline = ZepBaseline(llm_client=llm)
    baseline.ingest_text_turn("Caroline has a golden retriever named Max")
    baseline.ingest_text_turn("Max has a vet appointment next week")

    # Query semantically matches the dog-ownership fact directly; the vet
    # fact should still surface via 1-hop graph expansion through "Max".
    results = baseline.retrieve("tell me about Caroline's dog", top_k=10)
    contents = [r["content"].lower() for r in results]
    assert any("vet" in c for c in contents)


def test_zep_ignores_turns_with_nothing_to_extract():
    llm = _extraction_llm({})  # every turn falls through to empty extraction
    baseline = ZepBaseline(llm_client=llm)
    baseline.ingest_text_turn("Thanks, see you later!")

    assert baseline.retrieve("anything", top_k=5) == []
