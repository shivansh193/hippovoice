"""
LoCoMo benchmark evaluation.

LoCoMo: 30 long-term conversations (15–40 turns each, ~1,500 candidate memories)
with ground-truth QA pairs. Every major memory paper (Mem0, A-MEM, MemoryBank,
Supermemory) reports on this dataset.

Dataset: https://github.com/snap-research/locomo
Download: automatic via HuggingFace datasets if available, otherwise manual.
"""

from __future__ import annotations
import json
from pathlib import Path


def load_locomo(data_path: str | None = None) -> list[dict]:
    """
    Load LoCoMo conversations.

    Tries HuggingFace datasets first, falls back to local JSON.
    Each item: {"conversation": [...], "qa_pairs": [{"question": ..., "answer": ...}]}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("snap-research/locomo", split="test")
        return list(ds)
    except Exception:
        pass

    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return json.load(f)

    raise RuntimeError(
        "LoCoMo dataset not found.\n"
        "Install: pip install datasets\n"
        "Or download manually from https://github.com/snap-research/locomo\n"
        "and pass data_path='path/to/locomo.json'"
    )


def run_locomo(
    pipeline=None,
    llm_client=None,
    num_conversations: int = 30,
    data_path: str | None = None,
) -> dict:
    """
    Run LoCoMo evaluation.

    For each conversation:
      1. Ingest all turns into the pipeline memory
      2. For each QA pair, retrieve context and ask the LLM to answer
      3. Compare against ground truth (exact match + fuzzy)

    Returns {"accuracy": float, "details": list[dict]}
    """
    from pipeline import HippoVoicePipeline

    if pipeline is None:
        pipeline = HippoVoicePipeline(llm_client=llm_client, text_only=True)

    conversations = load_locomo(data_path)[:num_conversations]

    correct = 0
    total = 0
    details = []

    for conv in conversations:
        # Fresh pipeline per conversation
        from pipeline import HippoVoicePipeline
        conv_pipeline = HippoVoicePipeline(llm_client=pipeline.llm, text_only=True)

        # Ingest conversation turns
        for turn in conv.get("conversation", []):
            text = turn.get("text") or turn.get("content") or ""
            if text.strip():
                conv_pipeline.ingest_text_turn(text)

        # Answer QA pairs
        for qa in conv.get("qa_pairs", []):
            question = qa.get("question", "")
            gold_answer = qa.get("answer", "").lower().strip()

            retrieved = conv_pipeline.retrieve(question, top_k=5)
            context = build_qa_context(retrieved)

            predicted = conv_pipeline.llm.generate(
                system=(
                    "Answer the question using only the provided context. "
                    "Be concise — one sentence or less."
                ),
                messages=[
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                max_tokens=80,
            ).lower().strip()

            is_correct = _answer_matches(predicted, gold_answer)
            correct += int(is_correct)
            total += 1
            details.append({
                "question": question,
                "gold": gold_answer,
                "predicted": predicted,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": round(accuracy, 4), "total": total, "correct": correct, "details": details}


def build_qa_context(retrieved_memories: list[dict]) -> str:
    return "\n".join(f"- {m.get('content', '')}" for m in retrieved_memories)


def _answer_matches(predicted: str, gold: str) -> bool:
    if gold in predicted:
        return True
    # Fuzzy: all words in gold answer appear in predicted
    gold_words = set(gold.split())
    pred_words = set(predicted.split())
    overlap = gold_words & pred_words
    return len(overlap) / len(gold_words) >= 0.7 if gold_words else False
