"""
Mem0-style baseline.

Reimplements Mem0's actual memory-update algorithm locally rather than
depending on the `mem0ai` pip package. Reason: mem0ai's local/offline configs
only support OpenAI, Ollama, LiteLLM-server-backed, or LangChain providers --
there's no direct HuggingFace-transformers provider (see LLM.md in
mem0ai/mem0). Running a second inference server (Ollama) alongside our own
already-loaded LLM on a single T4 isn't practical for this benchmark, so we
reimplement the published algorithm against our own LLMClient instead.

Mem0's real pipeline (per their paper/docs):
  1. An LLM extracts a small set of candidate facts from each new turn.
  2. For each candidate fact, search existing memory for similar stored facts.
  3. If nothing similar exists, ADD it outright.
  4. If similar memories exist, an LLM call decides ADD / UPDATE / DELETE /
     NOOP relative to those matches.

No forgetting or decay anywhere in this pipeline -- once a fact is ADDed it
persists until an explicit UPDATE/DELETE decision touches it. That's the
property the goal doc calls out ("Mem0 — flat vector retrieval, no
forgetting").

Wire-compatible with NaiveRAG / HippoVoicePipeline: ingest_text_turn(text),
retrieve(query, top_k).
"""

from __future__ import annotations

import json

from memory.extractor import extract_memories
from memory.store import MemoryStore

# Only called when a candidate fact has a similar existing memory --
# most turns in a synthetic benchmark are semantically distinct, so this
# stays cheap (one extra LLM call only on real overlap).
DECISION_PROMPT = """\
You manage a memory database. A new fact was extracted from conversation.
Compare it against similar existing memories and decide ADD, UPDATE, or DELETE.

Return ONLY JSON: {{"action": "ADD|UPDATE|DELETE", "target_id": "..." or null}}

- ADD: the new fact is distinct information, keep both.
- UPDATE: the new fact supersedes/refines an existing memory (give its id as target_id).
- DELETE: the new fact contradicts an existing memory, which should be removed (give its id as target_id).

New fact: {fact}

Similar existing memories:
{candidates}"""

# Similarity gate for calling the decision LLM at all (cosine distance in
# ChromaDB's "cosine" space is 1 - similarity, so lower = more similar).
SIMILARITY_DISTANCE_THRESHOLD = 0.35


class Mem0Baseline:
    def __init__(self, llm_client=None):
        self.store = MemoryStore("mem0_baseline")
        self.llm = llm_client
        self.current_turn = 0

    def ingest_text_turn(self, text: str):
        facts = extract_memories(text, self.llm)
        for fact in facts:
            content = fact.get("content", "")
            if content.strip():
                self._add_or_update(content)
        self.current_turn += 1

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    # ── internal ──────────────────────────────────────────────────────────────

    def _add_or_update(self, content: str):
        similar = self.store.search(content, top_k=3) if self.store.count() else []
        close_matches = [m for m in similar if m.get("_distance", 1.0) <= SIMILARITY_DISTANCE_THRESHOLD]

        if not close_matches:
            self.store.add({"content": content, "turn_created": self.current_turn})
            return

        decision = self._decide(content, close_matches)
        action = decision.get("action", "ADD")
        target_id = decision.get("target_id")

        if action == "UPDATE" and target_id:
            self.store.delete(target_id)
            self.store.add({"content": content, "turn_created": self.current_turn}, target_id)
        elif action == "DELETE" and target_id:
            self.store.delete(target_id)
        elif action == "NOOP":
            pass
        else:  # ADD, or malformed decision -> default to ADD (matches Mem0's fail-open behavior)
            self.store.add({"content": content, "turn_created": self.current_turn})

    def _decide(self, fact: str, candidates: list[dict]) -> dict:
        candidate_text = "\n".join(f'- id={c["id"]}: {c["content"]}' for c in candidates)
        raw = self.llm.generate(
            system="You are a memory database manager. Output only valid JSON.",
            messages=[{"role": "user", "content": DECISION_PROMPT.format(fact=fact, candidates=candidate_text)}],
            max_tokens=100,
        )
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            decision = json.loads(cleaned)
            return decision if isinstance(decision, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
