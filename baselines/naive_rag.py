"""
Naive RAG baseline — flat vector retrieval, no forgetting, no salience scoring.

Everything goes in; top-k by cosine similarity comes out.
This is the strawman that hippocampal memory is measured against.
"""

from memory.store import MemoryStore


class NaiveRAG:
    def __init__(self):
        self.store = MemoryStore("naive_rag_baseline")
        self.current_turn = 0

    def ingest_text_turn(self, text: str):
        self.store.add(
            {
                "content": text,
                "turn_created": self.current_turn,
                "emotion": {"label": "neutral", "intensity": 0.0},
            }
        )
        self.current_turn += 1

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        return self.store.search(query, top_k=top_k)
