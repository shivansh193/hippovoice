"""
A-MEM-style baseline.

Reimplements A-MEM's Zettelkasten note-linking algorithm
(https://arxiv.org/abs/2502.12110) locally rather than vendoring the upstream
research repos (WujiangXu/A-mem, agiresearch/A-mem). Those repos pin their
own dependency/LLM-provider stack (assume OpenAI-style API keys, specific
chromadb/litellm versions) that don't fit cleanly into this pinned Colab
environment, so we reimplement the published algorithm against our own
LLMClient + MemoryStore/AssociationGraph instead.

A-MEM's real pipeline, per the paper:
  1. Each new memory becomes a "note": an LLM generates keywords, tags, and a
     short contextual description for it (structured attributes).
  2. The note is embedded and auto-linked to existing notes above a
     similarity threshold (dynamic indexing + link generation).
  3. Retrieval: vector search seeds + 1-hop expansion via links.

Deliberately NOT included, to isolate what HippoVoice's memory adds on top:
  - no salience/decay reranking (this baseline ranks purely by similarity +
    link adjacency, same as HippoRAG's seed+walk shape but without the
    Ebbinghaus weighting)
  - no forgetting (matches the goal doc's "A-MEM — Zettelkasten linking, no
    forgetting")

Wire-compatible with NaiveRAG / HippoVoicePipeline: ingest_text_turn(text),
retrieve(query, top_k).
"""

from __future__ import annotations

import json

from memory.store import AssociationGraph, MemoryStore

NOTE_PROMPT = """\
Generate a memory note for this piece of conversation. Extract:
- keywords: 3-5 short keywords
- tags: 1-3 broad topic tags
- context: one sentence summarizing why this might matter later

Return ONLY JSON: {{"keywords": [...], "tags": [...], "context": "..."}}

Text: {text}"""

# Same threshold as HippoRAG's AssociationGraph default, so the comparison
# isolates decay/salience rather than link-sensitivity differences.
LINK_THRESHOLD = 0.40


class AMemBaseline:
    def __init__(self, llm_client=None):
        self.store = MemoryStore("amem_baseline")
        self.graph = AssociationGraph()
        self.graph.AUTO_CONNECT_THRESHOLD = LINK_THRESHOLD
        self.llm = llm_client
        self.current_turn = 0

    def ingest_text_turn(self, text: str):
        note = self._make_note(text)
        memory = {
            "content": text,
            "keywords": note.get("keywords", []),
            "tags": note.get("tags", []),
            "context": note.get("context", ""),
            "turn_created": self.current_turn,
        }
        mid = self.store.add(memory)
        emb = self.store.embedder.encode(text)
        self.graph.add_node(mid, emb)
        self.current_turn += 1

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        seed_k = max(1, top_k // 2)
        seeds = self.store.search(query, top_k=seed_k)
        seed_ids = [s["id"] for s in seeds if "id" in s]

        expanded_ids = list(dict.fromkeys(seed_ids))  # preserve order, de-dup
        for sid in seed_ids:
            for nbr in self.graph.get_neighbors(sid, min_weight=LINK_THRESHOLD):
                if nbr not in expanded_ids:
                    expanded_ids.append(nbr)

        results = []
        for mid in expanded_ids[:top_k]:
            mem = self.store.get_by_id(mid)
            if mem:
                mem = {**mem, "id": mid}
                results.append(mem)
        return results

    # ── internal ──────────────────────────────────────────────────────────────

    def _make_note(self, text: str) -> dict:
        raw = self.llm.generate(
            system="You are a Zettelkasten note-taking assistant. Output only valid JSON.",
            messages=[{"role": "user", "content": NOTE_PROMPT.format(text=text)}],
            max_tokens=150,
        )
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            note = json.loads(cleaned)
            return note if isinstance(note, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
