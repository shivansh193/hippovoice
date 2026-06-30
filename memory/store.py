import pickle
import uuid
from pathlib import Path

import networkx as nx
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class AssociationGraph:
    """
    NetworkX graph where nodes are memory IDs and edges are cosine similarity
    between their sentence embeddings. Edges are only created when similarity
    exceeds AUTO_CONNECT_THRESHOLD.
    """

    AUTO_CONNECT_THRESHOLD = 0.40

    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, memory_id: str, embedding: np.ndarray):
        self.graph.add_node(memory_id, embedding=embedding)
        self._auto_connect(memory_id, embedding)

    def _auto_connect(self, new_id: str, new_emb: np.ndarray):
        for node_id, data in list(self.graph.nodes(data=True)):
            if node_id == new_id:
                continue
            node_emb = data.get("embedding")
            if node_emb is None:
                continue
            sim = _cosine_similarity(new_emb, node_emb)
            if sim >= self.AUTO_CONNECT_THRESHOLD:
                self.graph.add_edge(new_id, node_id, weight=round(sim, 4))

    def get_neighbors(self, memory_id: str, min_weight: float = 0.40) -> list[str]:
        if memory_id not in self.graph:
            return []
        return [
            nbr
            for nbr, attrs in self.graph[memory_id].items()
            if attrs.get("weight", 0.0) >= min_weight
        ]

    def remove_node(self, memory_id: str):
        if memory_id in self.graph:
            self.graph.remove_node(memory_id)


class MemoryStore:
    """
    ChromaDB-backed vector store for memory content.
    Uses all-MiniLM-L6-v2 (384-dim) for semantic embeddings.
    """

    EMBEDDER_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, collection_name: str = "hippovoice", persist_path: str | None = None):
        self.collection_name = collection_name
        self._id_to_meta: dict[str, dict] = {}

        if persist_path:
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(self.EMBEDDER_MODEL)

    def add(self, memory: dict, memory_id: str | None = None) -> str:
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        content = memory.get("content", "")
        embedding = self.embedder.encode(content).tolist()

        # Store all metadata except content (stored as document)
        meta = {k: v for k, v in memory.items() if k != "content"}
        # ChromaDB metadata values must be str/int/float/bool — flatten emotion dict
        flat_meta = _flatten_meta(meta)
        # ChromaDB rejects empty metadata dicts — ensure at least one field exists
        if not flat_meta:
            flat_meta = {"_type": "memory"}

        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[flat_meta],
        )
        self._id_to_meta[memory_id] = memory
        return memory_id

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        n = min(top_k, self._collection.count())
        if n == 0:
            return []
        query_emb = self.embedder.encode(query).tolist()
        results = self._collection.query(query_embeddings=[query_emb], n_results=n)
        return self._format_results(results)

    def get_by_id(self, memory_id: str) -> dict | None:
        if memory_id in self._id_to_meta:
            return self._id_to_meta[memory_id]
        try:
            result = self._collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if result["ids"]:
                doc = result["documents"][0]
                meta = result["metadatas"][0]
                return {"content": doc, **meta}
        except Exception:
            pass
        return None

    def update_meta(self, memory_id: str, updates: dict):
        if memory_id in self._id_to_meta:
            self._id_to_meta[memory_id].update(updates)
        flat = _flatten_meta(updates)
        self._collection.update(ids=[memory_id], metadatas=[flat])

    def delete(self, memory_id: str):
        self._collection.delete(ids=[memory_id])
        self._id_to_meta.pop(memory_id, None)

    def get_all(self) -> list[dict]:
        return list(self._id_to_meta.values())

    def count(self) -> int:
        return self._collection.count()

    def _format_results(self, results: dict) -> list[dict]:
        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for mid, doc, meta in zip(ids, docs, metas):
            entry = {"id": mid, "content": doc, **meta}
            if mid in self._id_to_meta:
                entry = {**self._id_to_meta[mid], "id": mid, **entry}
            out.append(entry)
        return out

    def get_embedding(self, memory_id: str) -> np.ndarray | None:
        """Return the stored embedding for a memory ID."""
        mem = self.get_by_id(memory_id)
        if mem is None:
            return None
        return self.embedder.encode(mem.get("content", ""))


class HippoMemory:
    """
    Unified memory facade: MemoryStore (ChromaDB) + AssociationGraph (NetworkX).
    This is the object the pipeline and retriever interact with.
    """

    def __init__(self, collection_name: str = "hippovoice", persist_path: str | None = None):
        self.store = MemoryStore(collection_name, persist_path)
        self.graph = AssociationGraph()

    @property
    def embedder(self):
        return self.store.embedder

    def add(self, memory: dict, memory_id: str | None = None) -> str:
        mid = self.store.add(memory, memory_id)
        emb = self.store.embedder.encode(memory.get("content", ""))
        self.graph.add_node(mid, emb)
        return mid

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.store.search(query, top_k)

    def get_by_id(self, memory_id: str) -> dict | None:
        return self.store.get_by_id(memory_id)

    def update_meta(self, memory_id: str, updates: dict):
        self.store.update_meta(memory_id, updates)

    def delete(self, memory_id: str):
        self.store.delete(memory_id)
        self.graph.remove_node(memory_id)

    def get_all(self) -> list[dict]:
        return self.store.get_all()

    def count(self) -> int:
        return self.store.count()

    def save(self, path: str):
        """Persist graph state. ChromaDB PersistentClient auto-saves."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)
        with open(p / "id_meta.pkl", "wb") as f:
            pickle.dump(self.store._id_to_meta, f)

    def load(self, path: str):
        """Restore graph state. Assumes ChromaDB PersistentClient points to same path."""
        p = Path(path)
        with open(p / "graph.pkl", "rb") as f:
            self.graph = pickle.load(f)
        with open(p / "id_meta.pkl", "rb") as f:
            self.store._id_to_meta = pickle.load(f)


def _flatten_meta(meta: dict) -> dict:
    """
    ChromaDB only accepts str/int/float/bool metadata values.
    Flatten nested dicts with dot notation and drop non-serialisable values.
    """
    flat = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_val = sub_v
                if isinstance(flat_val, (str, int, float, bool)):
                    flat[f"{k}.{sub_k}"] = flat_val
                else:
                    flat[f"{k}.{sub_k}"] = str(flat_val)
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat
