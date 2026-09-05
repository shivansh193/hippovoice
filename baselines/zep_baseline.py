"""
Zep-style baseline.

Reimplements the core algorithmic shape of Zep's Graphiti engine
(arXiv:2501.13956, "Zep: A Temporal Knowledge Graph Architecture for Agent
Memory") locally against our own LLMClient + MemoryStore, rather than
vendoring the real Graphiti/Neo4j stack -- same reasoning as
mem0_baseline.py/a_mem_baseline.py: the real system assumes its own
provider/database stack that doesn't fit this project's pinned, single-GPU
Colab/Kaggle environment.

Graphiti's real pipeline, per the paper (Section 3 + appendix, confirmed via
the actual paper text, not guessed): each conversation turn is an "episode";
an LLM pipeline runs Entity Extraction -> Entity Resolution -> Fact
Extraction -> Fact Resolution -> Temporal Extraction as separate stages,
producing entity nodes and (subject, predicate, object) fact edges with
bi-temporal validity (valid_at/invalid_at); entity resolution deduplicates
new entities against existing graph nodes via embedding similarity; edges
get invalidated when a new fact contradicts an existing one; retrieval
combines semantic search, BM25, and graph-distance signals via Reciprocal
Rank Fusion (RRF), which needs no explicit weight tuning between the three
signals.

Deliberately simplified from the real pipeline, documented here rather than
silently diverging:
  - ONE LLM call per turn does entity + fact + temporal extraction together
    (the real pipeline runs these as separate stages/calls). Matches this
    project's own established one-call-per-turn cost discipline (see
    memory/extractor.py's EXTRACTION_PROMPT, same reasoning) -- a 3-5x
    per-turn LLM-call multiplier for a benchmark baseline wasn't a cost
    trade worth making just to mirror Graphiti's internal staging, since
    the OUTPUT shape (entities + typed facts + temporal info) is what
    actually matters for a fair architectural comparison, not how many
    calls it took to produce it.
  - Edge invalidation is a deterministic rule (same subject entity + same
    predicate, different object -> the older edge is marked invalid),
    not an LLM-judged contradiction check. The paper doesn't fully specify
    whether Graphiti's own invalidation is LLM-driven or rule-based either
    (confirmed by direct inspection: the appendix describes the temporal
    comparison step but not conclusively which). A deterministic rule is
    cheap, reproducible, and testable -- the properties that matter for a
    benchmark baseline -- rather than an assumption written on top of an
    already-ambiguous spec.
  - No community-summary/Leiden-clustering tier. That's a higher-level
    aggregation feature on top of the base episodic/entity/fact graph, not
    part of the core retrieval path the LoCoMo benchmark actually exercises
    (a per-question fact lookup, not a "summarize this cluster" query).
  - The graph is a plain in-process structure (entity id -> fact edges),
    not Neo4j -- no different in spirit from HippoRAG's own AssociationGraph
    already being NetworkX rather than a real graph database.

Wire-compatible with NaiveRAG / HippoVoicePipeline: ingest_text_turn(text),
retrieve(query, top_k).
"""

from __future__ import annotations

import json
import math
import re
import uuid

from memory.store import MemoryStore, _cosine_similarity

EXTRACTION_PROMPT = """\
Extract entities and facts from this conversation turn, in the style of a \
knowledge graph: named entities (people, places, organizations, concepts), \
and facts connecting them as (subject, predicate, object) triples. If the \
turn starts with a date/time prefix (e.g. "[1:56 pm on 8 May, 2023]"), \
include that date as the "time" field on any fact it applies to -- most \
turns won't have one, extract those normally with time set to null.

Return ONLY JSON, no prose, no markdown fences:
{{"entities": [{{"name": "...", "type": "person|place|organization|concept"}}],
  "facts": [{{"subject": "...", "predicate": "...", "object": "...", "time": "..." or null}}]}}

If nothing worth extracting (a greeting, thanks, acknowledgment), return \
{{"entities": [], "facts": []}}.

Turn: {turn}"""

EXTRACTION_SYSTEM_PROMPT = "You are a knowledge graph extraction assistant. Output only valid JSON."

# Cosine similarity above which a newly extracted entity is treated as the
# same node as an existing one, rather than creating a duplicate. Same
# threshold family as HippoRAG's own AssociationGraph.AUTO_CONNECT_THRESHOLD
# and A-MEM-style's LINK_THRESHOLD, for the same reason: keeps the
# similarity-sensitivity comparable across baselines rather than introducing
# a fourth arbitrary constant.
ENTITY_RESOLUTION_THRESHOLD = 0.75

# Standard Okapi BM25 constants (Robertson/Sparck Jones) -- not tuned for
# this dataset, since the paper doesn't report tuning them either.
BM25_K1 = 1.5
BM25_B = 0.75

# Standard RRF constant from Cormack et al. 2009 ("Reciprocal Rank Fusion
# Outperforms Condorcet and Individual Rank Learning Methods") -- the paper
# Graphiti itself cites RRF from. Deliberately not tuned.
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _parse_json(raw: str) -> dict:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class ZepBaseline:
    def __init__(self, llm_client=None):
        # Fact content lives in a MemoryStore for its embedder + vector
        # search; entities and edges are tracked separately below since
        # Zep's actual edge semantics (typed, bi-temporal, explicitly
        # invalidated) don't match AssociationGraph's similarity-threshold
        # auto-connection -- that's a real, different kind of graph.
        self.store = MemoryStore("zep_baseline")
        self.llm = llm_client
        self.current_turn = 0

        self._entity_names: dict[str, str] = {}       # entity_id -> display name
        self._entity_embeddings: dict[str, "np.ndarray"] = {}  # entity_id -> name embedding
        # fact_id -> {"subject": id, "predicate": str, "object": id,
        #             "valid_at": int, "invalid_at": int|None}
        self._facts: dict[str, dict] = {}

    def ingest_text_turn(self, text: str) -> None:
        extracted = self._extract(text)
        entity_ids = {}
        for e in extracted.get("entities", []):
            name = e.get("name", "").strip()
            if name:
                entity_ids[name] = self._resolve_entity(name)

        for f in extracted.get("facts", []):
            subj_name = f.get("subject", "").strip()
            pred = f.get("predicate", "").strip()
            obj_name = f.get("object", "").strip()
            if not (subj_name and pred and obj_name):
                continue
            subj_id = entity_ids.get(subj_name) or self._resolve_entity(subj_name)
            obj_id = entity_ids.get(obj_name) or self._resolve_entity(obj_name)
            self._add_fact(subj_id, pred, obj_id, f.get("time"))

        self.current_turn += 1

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        live_ids = [fid for fid, f in self._facts.items() if f["invalid_at"] is None]
        if not live_ids:
            return []

        semantic_ranked = self._rank_semantic(query, live_ids)
        bm25_ranked = self._rank_bm25(query, live_ids)
        graph_ranked = self._rank_graph_distance(semantic_ranked, live_ids)

        rrf_scores: dict[str, float] = {}
        for ranked in (semantic_ranked, bm25_ranked, graph_ranked):
            for rank, fid in enumerate(ranked):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + 1.0 / (RRF_K + rank + 1)

        top_ids = sorted(rrf_scores, key=lambda fid: rrf_scores[fid], reverse=True)[:top_k]
        return [self._fact_as_memory(fid) for fid in top_ids]

    # ── internal: extraction + graph maintenance ────────────────────────────

    def _extract(self, text: str) -> dict:
        raw = self.llm.generate(
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(turn=text)}],
            max_tokens=250,
        )
        return _parse_json(raw)

    def _resolve_entity(self, name: str) -> str:
        emb = self.store.embedder.encode(name)
        best_id, best_sim = None, 0.0
        for eid, existing_emb in self._entity_embeddings.items():
            sim = _cosine_similarity(emb, existing_emb)
            if sim > best_sim:
                best_id, best_sim = eid, sim
        if best_id is not None and best_sim >= ENTITY_RESOLUTION_THRESHOLD:
            return best_id

        new_id = str(uuid.uuid4())
        self._entity_names[new_id] = name
        self._entity_embeddings[new_id] = emb
        return new_id

    def _add_fact(self, subj_id: str, predicate: str, obj_id: str, time: str | None) -> None:
        # Deterministic invalidation rule (see module docstring): a new fact
        # sharing (subject, predicate) with an existing live fact but naming
        # a different object supersedes it.
        for existing in self._facts.values():
            if (existing["invalid_at"] is None
                    and existing["subject"] == subj_id
                    and existing["predicate"] == predicate
                    and existing["object"] != obj_id):
                existing["invalid_at"] = self.current_turn

        fact_id = str(uuid.uuid4())
        content = self._fact_content(subj_id, predicate, obj_id, time)
        self._facts[fact_id] = {
            "subject": subj_id, "predicate": predicate, "object": obj_id,
            "time": time, "content": content,
            "valid_at": self.current_turn, "invalid_at": None,
        }
        self.store.add(
            {"content": content, "turn_created": self.current_turn},
            memory_id=fact_id,
        )

    def _fact_content(self, subj_id: str, predicate: str, obj_id: str, time: str | None) -> str:
        subj = self._entity_names.get(subj_id, subj_id)
        obj = self._entity_names.get(obj_id, obj_id)
        base = f"{subj} {predicate} {obj}"
        return f"{base} ({time})" if time else base

    def _fact_as_memory(self, fact_id: str) -> dict:
        f = self._facts[fact_id]
        return {
            "id": fact_id,
            "content": f["content"],
            "turn_created": f["valid_at"],
        }

    # ── internal: hybrid retrieval signals ──────────────────────────────────

    def _rank_semantic(self, query: str, live_ids: list[str]) -> list[str]:
        results = self.store.search(query, top_k=len(live_ids))
        live_set = set(live_ids)
        return [r["id"] for r in results if r.get("id") in live_set]

    def _rank_bm25(self, query: str, live_ids: list[str]) -> list[str]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        docs = {fid: _tokenize(self._facts[fid]["content"]) for fid in live_ids}
        n_docs = len(docs)
        avgdl = sum(len(d) for d in docs.values()) / n_docs if n_docs else 0.0

        df = {}
        for term in set(query_terms):
            df[term] = sum(1 for d in docs.values() if term in d)

        scores = {}
        for fid, doc in docs.items():
            doc_len = len(doc) or 1
            score = 0.0
            for term in query_terms:
                tf = doc.count(term)
                if tf == 0:
                    continue
                idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
                score += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avgdl))
            scores[fid] = score

        return sorted((fid for fid in live_ids if scores.get(fid, 0.0) > 0), key=lambda f: scores[f], reverse=True)

    def _rank_graph_distance(self, semantic_ranked: list[str], live_ids: list[str]) -> list[str]:
        # 1-hop graph expansion: facts sharing an entity (subject or object)
        # with a top semantic seed, mirroring HippoRAG's own seed-then-walk
        # shape rather than a from-scratch graph-search algorithm.
        seed_entities = set()
        for fid in semantic_ranked[:3]:
            f = self._facts[fid]
            seed_entities.add(f["subject"])
            seed_entities.add(f["object"])

        neighbors = [
            fid for fid in live_ids
            if fid not in semantic_ranked[:3]
            and (self._facts[fid]["subject"] in seed_entities or self._facts[fid]["object"] in seed_entities)
        ]
        return list(semantic_ranked[:3]) + neighbors
