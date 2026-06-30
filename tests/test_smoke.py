def test_imports():
    import chromadb
    import networkx
    import numpy
    import scipy
    assert True


def test_memory_imports():
    try:
        from memory.scorer import compute_salience
        from memory.extractor import tag_emotion_text, extract_memories
        from memory.decay import apply_forgetting_cycle
        from memory.store import HippoMemory, AssociationGraph
        from memory.retriever import hippo_retrieve
    except ValueError as e:
        if "numpy" in str(e).lower():
            import pytest; pytest.skip(f"numpy binary incompatibility (fix on Colab): {e}")
        raise


def test_llm_context_import():
    from llm.context import build_system_prompt, BASE_COMPANION_PROMPT
    assert BASE_COMPANION_PROMPT


def test_baseline_import():
    try:
        from baselines.naive_rag import NaiveRAG
    except ValueError as e:
        if "numpy" in str(e).lower():
            import pytest; pytest.skip(f"numpy binary incompatibility (fix on Colab): {e}")
        raise
