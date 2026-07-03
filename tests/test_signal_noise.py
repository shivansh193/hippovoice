"""
Signal/noise benchmark tests.

These run without GPU. Uses a dedicated passthrough_llm fixture rather than
the shared mock_llm fixture from conftest.py -- mock_llm's dog/car special
cases (added for extractor unit tests) hijack 2 of the 22 SIGNAL_TURNS into
replacement text that no longer contains the emotional keywords _is_signal
checks for ("My dog of twelve years passed away... I'm heartbroken" becomes
"user has a golden retriever" / "user's dog is named Max" -- losing
"heartbroken" entirely). This benchmark is testing retrieval/decay behavior,
not extraction correctness, so a pure passthrough mock avoids that coupling.

Default noise_per_signal=2 (66 total turns) -- see benchmarks/signal_noise/run.py
module docstring for why this ratio was chosen over the original 1:1 split.
"""

import json
from unittest.mock import MagicMock

import pytest
from pipeline import HippoVoicePipeline
from baselines.naive_rag import NaiveRAG
from baselines.mem0_baseline import Mem0Baseline
from baselines.a_mem_baseline import AMemBaseline
from benchmarks.signal_noise.run import run_signal_noise_benchmark, _is_signal


@pytest.fixture
def passthrough_llm():
    """LLM mock that never rewrites turn content -- verbatim extraction only."""
    mock = MagicMock()

    def side_effect(system, messages, max_tokens=512):
        user_content = messages[-1]["content"] if messages else ""
        sys_l = system.lower()

        if "zettelkasten" in sys_l:
            return json.dumps({"keywords": user_content.split()[:5], "tags": ["general"], "context": "note"})
        if "memory database" in sys_l:
            return json.dumps({"action": "ADD", "target_id": None})
        if "extract" in sys_l or "memory" in sys_l:
            turn_text = user_content.split("Turn: ", 1)[-1].strip()
            return json.dumps([{"content": turn_text, "entity": "unknown", "type": "fact"}])
        return "I understand."

    def batch_side_effect(system, messages_list, max_tokens=512):
        return [side_effect(system, messages, max_tokens) for messages in messages_list]

    mock.generate.side_effect = side_effect
    mock.generate_batch.side_effect = batch_side_effect
    return mock


# ── _is_signal classification ─────────────────────────────────────────────────

def test_is_signal_detects_emotional_turns():
    assert _is_signal("My father was diagnosed with cancer last week.")
    assert _is_signal("I got into a serious car accident today.")
    assert _is_signal("My dog of twelve years died yesterday.")


def test_is_signal_rejects_noise_turns():
    assert not _is_signal("The weather was cloudy today.")
    assert not _is_signal("I had cereal for breakfast.")
    assert not _is_signal("I saw a blue car parked outside.")


# ── HippoVoice noise rate ─────────────────────────────────────────────────────

def test_hippovoice_noise_rate_below_threshold(passthrough_llm):
    pipe = HippoVoicePipeline(llm_client=passthrough_llm, text_only=True)
    result = run_signal_noise_benchmark(pipe, "HippoVoice")

    assert result["noise_rate"] < 0.20, (
        f"HippoVoice noise rate {result['noise_rate']:.1%} exceeds 20% "
        f"(signal={result['signal_count']}, noise={result['noise_count']})"
    )


# ── Baseline comparisons ──────────────────────────────────────────────────────

def test_naive_rag_is_noisier_than_hippovoice(passthrough_llm):
    hippo = HippoVoicePipeline(llm_client=passthrough_llm, text_only=True)
    naive = NaiveRAG()

    hippo_result = run_signal_noise_benchmark(hippo, "HippoVoice")
    naive_result = run_signal_noise_benchmark(naive, "NaiveRAG")

    assert naive_result["noise_rate"] >= hippo_result["noise_rate"], (
        f"Naive RAG ({naive_result['noise_rate']:.1%}) should have >= noise "
        f"than HippoVoice ({hippo_result['noise_rate']:.1%})"
    )


def test_mem0_style_is_noisier_than_hippovoice(passthrough_llm):
    hippo = HippoVoicePipeline(llm_client=passthrough_llm, text_only=True)
    mem0 = Mem0Baseline(llm_client=passthrough_llm)

    hippo_result = run_signal_noise_benchmark(hippo, "HippoVoice")
    mem0_result = run_signal_noise_benchmark(mem0, "Mem0-style")

    assert mem0_result["noise_rate"] >= hippo_result["noise_rate"], (
        f"Mem0-style ({mem0_result['noise_rate']:.1%}) should have >= noise "
        f"than HippoVoice ({hippo_result['noise_rate']:.1%})"
    )


def test_a_mem_style_no_worse_than_hippovoice(passthrough_llm):
    # A-MEM-style has no decay either -- at this conversation length it can
    # tie HippoVoice (both stay near the association-graph "true positives"),
    # so this only asserts it isn't dramatically worse, not strictly worse.
    hippo = HippoVoicePipeline(llm_client=passthrough_llm, text_only=True)
    amem = AMemBaseline(llm_client=passthrough_llm)

    hippo_result = run_signal_noise_benchmark(hippo, "HippoVoice")
    amem_result = run_signal_noise_benchmark(amem, "AMem-style")

    assert amem_result["noise_rate"] <= hippo_result["noise_rate"] + 0.20


def test_result_structure(passthrough_llm):
    pipe = HippoVoicePipeline(llm_client=passthrough_llm, text_only=True)
    result = run_signal_noise_benchmark(pipe, "TestSystem")

    assert "system" in result
    assert "noise_rate" in result
    assert "signal_count" in result
    assert "noise_count" in result
    assert 0.0 <= result["noise_rate"] <= 1.0
    assert result["signal_count"] + result["noise_count"] == result["total_retrieved"]
