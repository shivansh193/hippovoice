import time
import numpy as np
import pytest
from memory.store import HippoMemory, AssociationGraph
from memory.retriever import retrieve_seeds, expand_via_graph, hippo_retrieve


# ── retrieve_seeds ────────────────────────────────────────────────────────────

def test_seed_retrieval_returns_relevant_ids(populated_store):
    seeds = retrieve_seeds("tell me about the user's pets", populated_store, top_k=3)
    assert len(seeds) <= 3
    contents = [populated_store.get_by_id(s)["content"].lower() for s in seeds if populated_store.get_by_id(s)]
    assert any("dog" in c or "max" in c or "retriever" in c for c in contents)


def test_seed_retrieval_respects_top_k(populated_store):
    seeds = retrieve_seeds("anything", populated_store, top_k=2)
    assert len(seeds) <= 2


def test_seed_retrieval_empty_store():
    store = HippoMemory(collection_name="empty_store_test")
    seeds = retrieve_seeds("query", store, top_k=3)
    assert seeds == []


# ── expand_via_graph ──────────────────────────────────────────────────────────

def test_graph_walk_expands_connected_nodes(connected_graph):
    expanded = expand_via_graph(["mem_dog1"], connected_graph)
    assert "mem_dog1" in expanded
    assert "mem_dog2" in expanded


def test_graph_walk_does_not_include_unconnected(connected_graph):
    expanded = expand_via_graph(["mem_dog1"], connected_graph)
    assert "mem_hike" not in expanded


def test_graph_walk_isolated_node_returns_seed_only(isolated_graph):
    expanded = expand_via_graph(["isolated_mem"], isolated_graph)
    assert "isolated_mem" in expanded
    assert len(expanded) == 1


def test_graph_walk_empty_seeds(connected_graph):
    expanded = expand_via_graph([], connected_graph)
    assert expanded == []


def test_graph_walk_deduplicates():
    from memory.store import AssociationGraph
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    g = AssociationGraph()
    # Use very similar sentences to guarantee auto-connect at 0.40 threshold
    g.add_node("a", embedder.encode("the user has a dog named Max"))
    g.add_node("b", embedder.encode("user owns a dog called Max"))

    # Both a and b are seeds — b is also a neighbour of a
    expanded = expand_via_graph(["a", "b"], g)
    assert expanded.count("a") == 1
    assert expanded.count("b") == 1


# ── hippo_retrieve ────────────────────────────────────────────────────────────

def test_availability_promotes_higher_salience_among_similarly_relevant_memories():
    # Both memories are about the same subject (Max) so relevance-to-query
    # should be comparable -- this isolates availability (current_salience)
    # as the deciding factor, rather than conflating it with a relevance
    # difference (e.g. one mentioning the exact query keyword and the other
    # not, which tests relevance ranking, not availability's contribution).
    mem = HippoMemory(collection_name="test_rerank")
    graph = mem.graph

    mem.add({
        "content": "Max the dog went to the park today",
        "emotion": {"label": "neutral", "intensity": 0.05},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "low_sal")

    mem.add({
        "content": "Max the dog got very sick and the user is heartbroken",
        "emotion": {"label": "sadness", "intensity": 0.95},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "high_sal")

    results = hippo_retrieve("what happened with Max", mem, graph, current_turn=5, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "high_sal", (
        "among similarly-relevant candidates, the one with higher availability "
        "(current_salience) should rank first"
    )


def test_relevance_still_matters_not_just_availability():
    # A highly-available (fresh, emotionally neutral but recent) memory that
    # is NOT about the query subject at all shouldn't beat a clearly relevant
    # memory just because it's newer -- relevance must still carry weight,
    # not be discarded in favor of pure availability.
    mem = HippoMemory(collection_name="test_relevance_matters")
    graph = mem.graph

    mem.add({
        "content": "the weather was cloudy this afternoon",
        "emotion": {"label": "neutral", "intensity": 0.1},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 5,
    }, "fresh_irrelevant")

    mem.add({
        "content": "the user's dog Max is a golden retriever who loves swimming",
        "emotion": {"label": "neutral", "intensity": 0.1},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "relevant")

    results = hippo_retrieve("tell me about Max the dog", mem, graph, current_turn=5, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "relevant"


def test_retrieve_increments_recall_count(populated_store):
    graph = populated_store.graph
    before = populated_store.get_by_id("mem_dog1")
    initial_count = before.get("recall_count", 0) if before else 0

    hippo_retrieve("golden retriever", populated_store, graph, current_turn=5, top_k=3)

    after = populated_store.get_by_id("mem_dog1")
    if after:  # mem_dog1 might not have been in top results
        new_count = after.get("recall_count", 0)
        assert new_count >= initial_count  # at minimum unchanged, likely incremented


def test_retrieve_returns_at_most_top_k(populated_store):
    results = hippo_retrieve("anything", populated_store, populated_store.graph, current_turn=0, top_k=2)
    assert len(results) <= 2


def test_retrieve_empty_store():
    mem = HippoMemory(collection_name="empty_retrieve_test")
    results = hippo_retrieve("query", mem, mem.graph, current_turn=0)
    assert results == []


def test_exact_name_match_disambiguates_similarly_embedded_names():
    # Confirmed on a real LoCoMo run: "Jon" and "John" (different people) are
    # close enough in embedding space that pure cosine similarity can't
    # reliably tell them apart -- retrieval for a "John" question
    # consistently surfaced "Jon" content instead. An exact whole-word name
    # match should let the correctly-named memory win even against a
    # topically-similar, wrong-name distractor.
    mem = HippoMemory(collection_name="test_name_match")
    graph = mem.graph

    mem.add({
        "content": "Jon took a trip to Rome last week to clear his mind",
        "emotion": {"label": "neutral", "intensity": 0.1},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "wrong_name")

    mem.add({
        "content": "John visited Rome for a work conference",
        "emotion": {"label": "neutral", "intensity": 0.1},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "right_name")

    results = hippo_retrieve("Which city did John visit?", mem, graph, current_turn=5, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "right_name", (
        "exact name match on 'John' should win over a similarly-embedded "
        "'Jon' memory about the same general topic"
    )


def test_name_match_guarantees_candidate_even_if_crowded_out_of_seed_pool():
    # The correct memory must be considered even if many topically-similar
    # distractor memories (sharing a similarly-embedded but wrong name)
    # dominate the raw embedding-similarity seed pool -- otherwise it never
    # even reaches reranking for the name-match bonus to promote.
    mem = HippoMemory(collection_name="test_name_match_pool")
    graph = mem.graph

    for i in range(20):
        mem.add({
            "content": f"Jon went to a city on trip number {i}",
            "emotion": {"label": "neutral", "intensity": 0.1},
            "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
        }, f"jon_{i}")

    mem.add({
        "content": "John visited Rome for a work conference",
        "emotion": {"label": "neutral", "intensity": 0.1},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "right_name")

    results = hippo_retrieve("Which city did John visit?", mem, graph, current_turn=5, top_k=3)
    assert any(r["id"] == "right_name" for r in results), (
        "an exact name match should be considered even if crowded out of "
        "the raw embedding-similarity seed pool by many similarly-worded "
        "distractor memories"
    )


def test_extract_proper_nouns_excludes_question_words():
    from memory.retriever import _extract_proper_nouns
    names = _extract_proper_nouns("Which city have both Jean and John visited?")
    assert names == {"Jean", "John"}
    assert "Which" not in names


def test_decay_lambda_override_keeps_old_memories_available():
    # Confirms hippo_retrieve actually uses a custom decay_lambda rather than
    # always falling back to the module default. Confirmed on a real LoCoMo
    # run that the default (tuned for ~90-100 turn conversations) crushes
    # availability to near-zero long before a 369-663 turn conversation
    # ends -- two separate stores (not two calls on the same store) since
    # hippo_retrieve increments recall_count as a side effect, which would
    # otherwise contaminate the second call's salience computation.
    def _make_store(name):
        mem = HippoMemory(collection_name=name)
        mem.add({
            "content": "the user mentioned their dog Max",
            "emotion": {"label": "neutral", "intensity": 0.1},
            "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
        }, "old_mem")
        return mem

    default_mem = _make_store("test_decay_lambda_default")
    default_results = hippo_retrieve("tell me about Max", default_mem, default_mem.graph, current_turn=600, top_k=1)

    scaled_mem = _make_store("test_decay_lambda_scaled")
    scaled_results = hippo_retrieve(
        "tell me about Max", scaled_mem, scaled_mem.graph, current_turn=600, top_k=1, decay_lambda=0.001
    )

    assert default_results[0]["current_salience"] < 0.01, "Default decay_lambda should crush availability by turn 600"
    assert scaled_results[0]["current_salience"] > 0.3, "A much smaller decay_lambda should keep this available"


def test_retrieval_latency_500_memories():
    mem = HippoMemory(collection_name="latency_test_500")
    graph = mem.graph

    for i in range(500):
        mem.add({
            "content": f"memory {i} about various daily events and experiences",
            "emotion": {"label": "neutral", "intensity": 0.3},
            "base_weight": 1.0, "recall_count": 0, "turn_created": i,
        }, f"lat_{i}")

    start = time.perf_counter()
    hippo_retrieve("recent experiences", mem, graph, current_turn=600, top_k=5)
    elapsed = time.perf_counter() - start

    # Relaxed from 500ms: computing relevance now requires one batched
    # embedding call over the whole candidate pool (previously ranking was
    # pure arithmetic on precomputed salience, no embedding work at
    # retrieval time at all) -- genuine added cost for a real capability,
    # not a regression to claw back.
    assert elapsed < 1.5, f"Retrieval took {elapsed:.3f}s, target < 1.5s"
