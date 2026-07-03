from memory.scorer import compute_salience
from memory.store import HippoMemory, AssociationGraph


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
) -> list[dict]:
    """
    Full HippoRAG retrieval: seed → graph walk → rerank by current salience.

    Seeds are over-fetched well past top_k (4x, floor 15) before reranking —
    salience can only promote a high-salience/lower-similarity memory above a
    low-salience/higher-similarity one if it's actually in the candidate pool.
    With a narrow seed pool (top_k alone), salience reranking has nothing to
    work with once noise turns outnumber signal turns by enough volume to
    dominate raw vector similarity to the query.

    Graph expansion only walks from the closest `graph_expand_seeds` seeds
    (by raw similarity, not the full over-fetched tail). Walking from a
    marginal, barely-relevant seed pulls in nodes whose only qualification is
    "embedding-similar to something borderline" — which gave low-relevance,
    high-recency noise a backdoor into the salience-reranked pool even though
    it would never have made the cut on similarity to the query itself.

    Side effect: increments recall_count on each retrieved memory so future
    salience calculations reflect the reinforcement.
    """
    seed_ids = retrieve_seeds(query, memory, top_k=max(top_k * 4, 15))
    expanded_ids = expand_via_graph(seed_ids[:graph_expand_seeds], graph)
    candidate_ids = list(dict.fromkeys(seed_ids + expanded_ids))

    candidates = []
    for mid in candidate_ids:
        m = memory.get_by_id(mid)
        if m is None:
            continue
        turns_elapsed = current_turn - m.get("turn_created", 0)
        score = compute_salience(
            base_weight=m.get("base_weight", 1.0),
            emotion=m.get("emotion", {"label": "neutral", "intensity": 0.0}),
            recall_count=m.get("recall_count", 0),
            turns_elapsed=turns_elapsed,
        )
        candidates.append({**m, "id": mid, "current_salience": round(score, 4)})

    # Rerank by salience descending
    ranked = sorted(candidates, key=lambda x: x["current_salience"], reverse=True)
    top = ranked[:top_k]

    # Increment recall_count for returned memories
    for m in top:
        mid = m["id"]
        new_count = m.get("recall_count", 0) + 1
        memory.update_meta(mid, {"recall_count": new_count})

    return top
