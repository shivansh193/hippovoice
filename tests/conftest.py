"""
Shared pytest fixtures.

GPU-dependent fixtures (canary_model, fish_model, llm_client) are marked @gpu
and skipped automatically when CUDA is not available. Run with:
    pytest -m "not gpu"   # CPU-only (memory / logic tests)
    pytest                # all tests (requires A100)
"""

import json
import numpy as np
import pytest
from unittest.mock import MagicMock


# ── GPU guard ─────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


def pytest_collection_modifyitems(config, items):
    import torch
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ── LLM mock ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """LLM that returns a plausible JSON memory extraction for dog-related input."""
    mock = MagicMock()

    def smart_generate(system, messages, max_tokens=512):
        user_content = messages[-1]["content"] if messages else ""
        user_lower = user_content.lower()

        # Memory extraction responses
        if "extract" in system.lower() or "memory" in system.lower():
            if "dog" in user_lower or "max" in user_lower or "retriever" in user_lower:
                return json.dumps([
                    {"content": "user has a golden retriever", "entity": "dog", "type": "fact"},
                    {"content": "user's dog is named Max", "entity": "Max", "type": "fact"},
                ])
            if "car" in user_lower and ("accident" in user_lower or "hit" in user_lower):
                return json.dumps([
                    {"content": "user's dog Max got hit by a car", "entity": "Max", "type": "event"},
                ])
            return json.dumps([
                {"content": user_content[:120], "entity": "unknown", "type": "fact"}
            ])

        # Summarisation / compression
        if "summarise" in system.lower() or "summary" in system.lower():
            return "User mentioned several minor things including " + user_content[:60]

        # QA answering
        if "context" in user_lower and "question" in user_lower:
            return "Based on the context, the answer is related to the user's experiences."

        return "I understand. Tell me more."

    mock.generate.side_effect = smart_generate
    return mock


# ── Audio embeddings ──────────────────────────────────────────────────────────

@pytest.fixture
def flat_audio_embedding():
    """Low-variance embedding simulating calm/neutral speech."""
    return np.ones(1280, dtype=np.float32) * 0.5


@pytest.fixture
def sad_audio_embedding():
    """High-variance embedding simulating emotional/distressed speech."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 3.0, size=(1280,)).astype(np.float32)


# ── Memory store fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def populated_store():
    """HippoMemory pre-loaded with a small set of test memories."""
    from memory.store import HippoMemory

    store = HippoMemory(collection_name="test_populated")
    entries = [
        ("mem_dog1", "user has a golden retriever", "joy", 0.7, 0),
        ("mem_dog2", "user's dog is named Max", "joy", 0.8, 1),
        ("mem_hike", "user likes hiking on weekends", "neutral", 0.3, 2),
        ("mem_sister", "user's sister lives in Seattle", "neutral", 0.2, 3),
    ]
    for mid, content, label, intensity, turn in entries:
        store.add({
            "content": content,
            "emotion": {"label": label, "intensity": intensity},
            "base_weight": 1.0,
            "recall_count": 0,
            "turn_created": turn,
        }, mid)
    return store


@pytest.fixture
def connected_graph():
    """AssociationGraph with dog memories connected and hiking isolated."""
    from memory.store import AssociationGraph
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    graph = AssociationGraph()
    graph.add_node("mem_dog1", embedder.encode("user has a golden retriever"))
    graph.add_node("mem_dog2", embedder.encode("user's dog is named Max"))
    graph.add_node("mem_hike", embedder.encode("user loves hiking in the mountains"))
    return graph


@pytest.fixture
def isolated_graph():
    """AssociationGraph where the single node has no neighbours."""
    from memory.store import AssociationGraph
    import numpy as np

    graph = AssociationGraph()
    # Use orthogonal vectors so cosine similarity = 0 (no auto-connect possible)
    graph.add_node("isolated_mem", np.array([1.0] + [0.0] * 383))
    return graph


# ── GPU-dependent fixtures (skipped on CPU) ───────────────────────────────────

@pytest.fixture(scope="session")
def canary_model():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from stt.model import load_canary
    return load_canary()


@pytest.fixture(scope="session")
def fish_model():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tts.model import load_fish_tts
    return load_fish_tts()


@pytest.fixture(scope="session")
def llm_client():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from llm.client import LLMClient
    return LLMClient()
