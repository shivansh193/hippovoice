"""Token cost benchmark — measures average tokens per retrieval context."""

from llm.context import build_system_prompt


def measure_token_cost(
    pipeline,
    num_queries: int = 50,
    tokenizer=None,
) -> float:
    """
    For num_queries queries against the pipeline's current memory state,
    measure the average token count of the system prompt (context) passed to the LLM.

    Returns average token count as float.
    """
    sample_queries = [
        "What does the user like to do on weekends?",
        "Tell me about the user's family.",
        "What are the user's biggest fears?",
        "What happened to the user recently?",
        "What are the user's hobbies and interests?",
        "How is the user feeling about their career?",
        "What pets does the user have?",
        "Where does the user live?",
        "What is the user's relationship status?",
        "What challenges is the user facing?",
    ]

    # Cycle through sample queries to reach num_queries
    queries = [sample_queries[i % len(sample_queries)] for i in range(num_queries)]

    if tokenizer is None and hasattr(pipeline, "llm"):
        tokenizer = getattr(pipeline.llm, "tokenizer", None)

    total_tokens = 0
    for q in queries:
        retrieved = pipeline.retrieve(q, top_k=5)
        prompt = build_system_prompt(retrieved, tokenizer=tokenizer)
        if tokenizer is not None:
            tokens = len(tokenizer.encode(prompt))
        else:
            tokens = len(prompt) // 4  # rough estimate
        total_tokens += tokens

    return total_tokens / num_queries
