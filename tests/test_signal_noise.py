"""
Signal/noise benchmark tests.

These run without GPU — HippoVoicePipeline in text_only mode.
The mock_llm fixture handles memory extraction so LLM calls don't require a real model.
"""

import pytest
from pipeline import HippoVoicePipeline
from baselines.naive_rag import NaiveRAG
from benchmarks.signal_noise.run import run_signal_noise_benchmark, _is_signal


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

def test_hippovoice_noise_rate_below_threshold(mock_llm):
    pipe = HippoVoicePipeline(llm_client=mock_llm, text_only=True)
    result = run_signal_noise_benchmark(pipe, "HippoVoice")

    assert result["noise_rate"] < 0.20, (
        f"HippoVoice noise rate {result['noise_rate']:.1%} exceeds 20% "
        f"(signal={result['signal_count']}, noise={result['noise_count']})"
    )


# ── Naive RAG baseline ────────────────────────────────────────────────────────

def test_naive_rag_is_noisier_than_hippovoice(mock_llm):
    hippo = HippoVoicePipeline(llm_client=mock_llm, text_only=True)
    naive = NaiveRAG()

    hippo_result = run_signal_noise_benchmark(hippo, "HippoVoice")
    naive_result = run_signal_noise_benchmark(naive, "NaiveRAG")

    assert naive_result["noise_rate"] >= hippo_result["noise_rate"], (
        f"Naive RAG ({naive_result['noise_rate']:.1%}) should have >= noise "
        f"than HippoVoice ({hippo_result['noise_rate']:.1%})"
    )


def test_result_structure(mock_llm):
    pipe = HippoVoicePipeline(llm_client=mock_llm, text_only=True)
    result = run_signal_noise_benchmark(pipe, "TestSystem")

    assert "system" in result
    assert "noise_rate" in result
    assert "signal_count" in result
    assert "noise_count" in result
    assert 0.0 <= result["noise_rate"] <= 1.0
    assert result["signal_count"] + result["noise_count"] == result["total_retrieved"]
