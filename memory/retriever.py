import math

from memory.scorer import compute_salience
from memory.store import HippoMemory, AssociationGraph, _cosine_similarity
from memory.decay import FORGET_THRESHOLD

# Log-space bounds for normalizing availability into [0, 1]. FORGET_THRESHOLD
# is the existing boundary below which a memory is already being deleted by
# the forgetting cycle, so it's the natural floor. ~4.0 is close to the
# highest salience a fresh, maximally emotion-boosted memory can reach
# (base_weight=1 x salience_weight up to ~3.5-4 x recall_boost).
_MIN_AVAILABILITY_LOG = math.log(FORGET_THRESHOLD)
_MAX_AVAILABILITY_LOG = math.log(4.0)


def _availability_score(salience: float) -> float:
    """
    Map a raw current_salience value (which spans dozens of orders of
    magnitude over a long conversation -- from ~1e-13 decayed to ~1+ fresh)
    into a bounded [0, 1] score, log-linear between FORGET_THRESHOLD and a
    near-maximum salience value.

    Availability needs to combine with relevance (already bounded, roughly
    [0, 1] for cosine similarity between related text) without either
    signal trivially dominating the other. A raw additive/multiplicative
    blend fails: a fresh, barely-relevant memory (salience ~1) would still
    swamp an old, highly-relevant one (salience ~1e-13) purely because of
    units, reproducing the exact recency-collapse problem availability was
    meant to fix. This log-normalizes first so magnitude differences are
    preserved (unlike rank-based fusion, which only knows "more" or "less"
    and loses sensitivity entirely once a candidate pool has just a
    handful of items) but bounded so neither term can numerically dominate
    just because of scale.
    """
    if salience <= 0:
        return 0.0
    log_s = math.log(salience)
    normalized = (log_s - _MIN_AVAILABILITY_LOG) / (_MAX_AVAILABILITY_LOG - _MIN_AVAILABILITY_LOG)
    return max(0.0, min(1.0, normalized))


def retrieve_seeds(query: str, memory: HippoMemory, top_k: int = 3) -> list[str]:
    """
    Step 1 of HippoRAG: vector similarity search to get seed memory IDs.
    """
    results = memory.search(query, top_k=top_k)
    return [r["id"] for r in results if "id" in r]


def expand_via_graph(
    seed_ids: list[str],
    graph: AssociationGraph,
    max_hops: int = 1,
) -> list[str]:
    """
    Step 2: walk the association graph from each seed.
    Currently supports max_hops=1 (direct neighbours only).
    Returns deduplicated list of seed IDs + all neighbours.
    """
    expanded = set(seed_ids)
    frontier = set(seed_ids)

    for _ in range(max_hops):
        next_frontier = set()
        for node_id in frontier:
            neighbors = graph.get_neighbors(node_id)
            new = set(neighbors) - expanded
            expanded.update(new)
            next_frontier.update(new)
        frontier = next_frontier

    return list(expanded)


def hippo_retrieve(
    query: str,
    memory: HippoMemory,
    graph: AssociationGraph,
    current_turn: int,
    top_k: int = 5,
    graph_expand_seeds: int = 10,
    relevance_weight: float = 0.65,
) -> list[dict]:
    """
    Full HippoRAG retrieval: seed → graph walk → rerank by relevance × availability.

    Seeds are over-fetched well past top_k (4x, floor 15) before reranking —
    the final rerank can only promote a candidate above a more-similar one
    if it's actually in the pool. With a narrow seed pool (top_k alone),
    reranking has nothing to work with once noise turns outnumber signal
    turns by enough volume to dominate raw vector similarity to the query.

    Graph expansion only walks from the closest `graph_expand_seeds` seeds
    (by raw similarity, not the full over-fetched tail). Walking from a
    marginal, barely-relevant seed pulls in nodes whose only qualification is
    "embedding-similar to something borderline" — which gave low-relevance,
    high-recency noise a backdoor into the reranked pool even though it
    would never have made the cut on similarity to the query itself.

    Final ranking combines two signals per candidate:
      - relevance:    cosine similarity between the query and this memory's
                       content -- what HippoRAG's associative retrieval is
                       actually supposed to answer ("does this fit the cue?").
      - availability: current_salience from the Ebbinghaus/emotion decay
                       model, log-normalized into [0, 1] (see
                       _availability_score) -- "has this been retained?"
                       (a retention prior, not the ranking signal itself).

    score = relevance_weight * relevance + (1 - relevance_weight) * availability_score

    Both terms are bounded to comparable ranges before combining. Two
    simpler approaches were tried and rejected:
      - Raw multiplicative/additive blend on unnormalized values: fails
        outright, since availability can span ~1e-13 to ~1+ over a long
        conversation -- a fresh, irrelevant memory would still swamp an
        old, highly-relevant one purely on units.
      - Reciprocal rank fusion (rank position instead of magnitude): fixes
        the scale problem but overcorrects -- with a small candidate pool
        (a handful of items, common in practice), rank 0 vs rank 1 barely
        differs regardless of whether the actual availability gap is 1% or
        1000000%, so it can't reliably let a truly dominant signal actually
        dominate.
    Log-normalizing availability to a fixed [0, 1] scale (anchored to
    FORGET_THRESHOLD, not the current candidate pool) keeps real magnitude
    differences meaningful while still preventing either signal from
    numerically overwhelming the other purely due to units.

    Side effect: increments recall_count on each retrieved memory so future
    salience calculations reflect the reinforcement.
    """
    seed_ids = retrieve_seeds(query, memory, top_k=max(top_k * 4, 15))
    expanded_ids = expand_via_graph(seed_ids[:graph_expand_seeds], graph)
    candidate_ids = list(dict.fromkeys(seed_ids + expanded_ids))

    found = [(mid, memory.get_by_id(mid)) for mid in candidate_ids]
    found = [(mid, m) for mid, m in found if m is not None]
    if not found:
        return []

    # Batch-encode query + all candidate contents in one call rather than
    # one encode() per candidate -- with a pool of dozens of candidates per
    # retrieval, per-item encoding turned a single-digit-millisecond op into
    # a multi-second one.
    texts = [query] + [m.get("content", "") for _, m in found]
    embeddings = memory.embedder.encode(texts)
    query_emb, mem_embs = embeddings[0], embeddings[1:]

    candidates = []
    for (mid, m), mem_emb in zip(found, mem_embs):
        turns_elapsed = current_turn - m.get("turn_created", 0)
        availability = compute_salience(
            base_weight=m.get("base_weight", 1.0),
            emotion=m.get("emotion", {"label": "neutral", "intensity": 0.0}),
            recall_count=m.get("recall_count", 0),
            turns_elapsed=turns_elapsed,
        )
        relevance = _cosine_similarity(query_emb, mem_emb)
        score = relevance_weight * relevance + (1 - relevance_weight) * _availability_score(availability)
        candidates.append({
            **m,
            "id": mid,
            "current_salience": round(availability, 6),
            "_relevance": round(relevance, 6),
            "_score": round(score, 6),
        })

    ranked = sorted(candidates, key=lambda c: c["_score"], reverse=True)
    top = ranked[:top_k]

    # Increment recall_count for returned memories
    for m in top:
        mid = m["id"]
        new_count = m.get("recall_count", 0) + 1
        memory.update_meta(mid, {"recall_count": new_count})

    return top
