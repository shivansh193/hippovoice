import pickle
import numpy as np
import pytest
from memory.store import HippoMemory, AssociationGraph, MemoryStore


# ── MemoryStore ───────────────────────────────────────────────────────────────

def test_semantic_search_ranking():
    store = MemoryStore("test_search_ranking")
    store.add({"content": "user has a golden retriever named Max"}, "mem_dog")
    store.add({"content": "user likes hiking on weekends"}, "mem_hike")
    store.add({"content": "user's sister lives in Seattle"}, "mem_sister")

    results = store.search("what kind of pet does the user have?", top_k=3)
    assert len(results) > 0
    top = results[0]["content"].lower()
    assert "max" in top or "retriever" in top or "dog" in top


def test_add_and_retrieve_count():
    store = MemoryStore("test_count_store")
    for i in range(10):
        store.add({"content": f"fact number {i}"}, f"id_{i}")
    results = store.search("fact", top_k=5)
    assert len(results) == 5


def test_get_by_id_returns_correct_content():
    store = MemoryStore("test_get_by_id")
    store.add({"content": "user loves jazz music"}, "jazz_mem")
    m = store.get_by_id("jazz_mem")
    assert m is not None
    assert "jazz" in m["content"].lower()


def test_get_by_id_missing_returns_none():
    store = MemoryStore("test_missing_id")
    assert store.get_by_id("nonexistent_id_xyz") is None


def test_delete_removes_memory():
    store = MemoryStore("test_delete")
    store.add({"content": "to be deleted"}, "del_me")
    store.delete("del_me")
    assert store.get_by_id("del_me") is None
    assert store.count() == 0


def test_count_reflects_additions():
    store = MemoryStore("test_count")
    assert store.count() == 0
    store.add({"content": "one"}, "m1")
    assert store.count() == 1
    store.add({"content": "two"}, "m2")
    assert store.count() == 2


# ── AssociationGraph ──────────────────────────────────────────────────────────

def test_similar_memories_auto_connect(connected_graph):
    assert "mem_dog2" in connected_graph.get_neighbors("mem_dog1")


def test_dissimilar_memories_not_connected(connected_graph):
    assert "mem_hike" not in connected_graph.get_neighbors("mem_dog1")


def test_isolated_node_has_no_neighbours(isolated_graph):
    assert isolated_graph.get_neighbors("isolated_mem") == []


def test_get_neighbors_missing_node():
    from memory.store import AssociationGraph
    g = AssociationGraph()
    assert g.get_neighbors("ghost_node") == []


def test_graph_serialisation():
    g = AssociationGraph()
    g.add_node("a", np.array([1.0] + [0.0] * 383, dtype=np.float32))
    blob = pickle.dumps(g)
    restored = pickle.loads(blob)
    assert "a" in restored.graph.nodes


def test_remove_node():
    g = AssociationGraph()
    g.add_node("x", np.ones(384, dtype=np.float32))
    g.remove_node("x")
    assert "x" not in g.graph.nodes


# ── HippoMemory (unified facade) ──────────────────────────────────────────────

def test_hippomemory_add_and_search():
    mem = HippoMemory(collection_name="test_hippo_add")
    mem.add({
        "content": "user has a cat named Luna",
        "emotion": {"label": "joy", "intensity": 0.6},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "luna_mem")
    results = mem.search("user's cat", top_k=1)
    assert len(results) == 1
    assert "luna" in results[0]["content"].lower()


def test_hippomemory_graph_auto_connects():
    mem = HippoMemory(collection_name="test_hippo_graph")
    mem.add({"content": "user has a golden retriever"}, "d1")
    mem.add({"content": "user's dog is named Max"}, "d2")
    mem.add({"content": "user enjoys mountain climbing"}, "c1")

    dog_neighbors = mem.graph.get_neighbors("d1")
    assert "d2" in dog_neighbors
    assert "c1" not in dog_neighbors


def test_hippomemory_persistence(tmp_path):
    mem = HippoMemory(collection_name="test_hippo_persist")
    mem.add({
        "content": "user's dog is named Max",
        "emotion": {"label": "joy", "intensity": 0.7},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0,
    }, "persist_test")

    save_path = str(tmp_path / "hippo_state")
    mem.save(save_path)

    restored = HippoMemory(collection_name="test_hippo_persist_reload")
    restored.load(save_path)

    # Graph should be restored
    assert "persist_test" in restored.graph.graph.nodes
