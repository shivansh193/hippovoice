"""
LoCoMo benchmark evaluation.

LoCoMo: 10 long-term conversations (19-32 sessions each, hundreds of turns per
conversation) with ground-truth QA pairs. Every major memory paper (Mem0,
A-MEM, MemoryBank, Supermemory) reports on this dataset.

Dataset: https://github.com/snap-research/locomo (data/locomo10.json)

Note: there is no "snap-research/locomo" (or similar) dataset on the
HuggingFace Hub -- the only real distribution is the JSON file in the GitHub
repo above. We download and cache it directly rather than going through
`datasets.load_dataset`.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_CACHE_PATH = Path(__file__).parent / "locomo10.json"

# category 5 = adversarial/unanswerable questions (ground truth is
# "adversarial_answer", not "answer" -- the correct model behavior is to
# recognize the question isn't answerable from the conversation). Excluded
# by default since they need different scoring logic than exact/fuzzy match.
ADVERSARIAL_CATEGORY = 5


def load_locomo(data_path: str | None = None) -> list[dict]:
    """
    Load the 10 LoCoMo conversations, downloading + caching on first use.

    Each item: {
        "qa": [{"question", "answer", "evidence", "category"}, ...],
        "conversation": {
            "speaker_a", "speaker_b",
            "session_1_date_time", "session_1": [{"speaker", "dia_id", "text"}, ...],
            "session_2_date_time", "session_2": [...], ...
        },
        "sample_id": ...,
    }
    """
    cache_path = Path(data_path) if data_path else DEFAULT_CACHE_PATH

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(LOCOMO_URL, cache_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not download LoCoMo dataset from {LOCOMO_URL}: {e}\n"
                f"Download manually and pass data_path='path/to/locomo10.json'"
            ) from e

    with open(cache_path) as f:
        return json.load(f)


def _flatten_conversation(conv: dict) -> list[str]:
    """Flatten all session turns, in session order, into 'Speaker: text' strings."""
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    turns = []
    for key in session_keys:
        for turn in conv[key]:
            text = (turn.get("text") or "").strip()
            if text:
                turns.append(f"{turn.get('speaker', '')}: {text}")
    return turns


def run_locomo(
    pipeline=None,
    llm_client=None,
    num_conversations: int = 10,
    max_qa_per_conversation: int | None = None,
    include_adversarial: bool = False,
    data_path: str | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run LoCoMo evaluation.

    For each conversation:
      1. Ingest all turns (flattened across sessions, in order) into a fresh
         pipeline's memory.
      2. For each QA pair, retrieve context and ask the LLM to answer.
      3. Compare against ground truth (exact match + fuzzy).

    `max_qa_per_conversation` caps QA pairs per conversation -- useful for a
    cheap smoke test (a full run is ~150-250 QA pairs x 10 conversations,
    each needing an LLM call, which is slow on a single T4).

    `checkpoint_path`, if given, saves progress to that JSON file after each
    conversation and resumes from it if the file already exists. Every turn
    and QA pair is a real LLM call, so a full run is tens of minutes to a
    couple hours -- long enough that a Colab free-tier disconnect losing all
    progress is a real, painful risk without this.

    The checkpoint records a fingerprint (LLM model name/backend + run
    parameters) and refuses to resume from one written under a different
    setup -- e.g. a dry-run mock LLM, or a different num_conversations --
    since silently trusting a stale file there would replay whatever
    (possibly garbage) results it already has instead of actually running.
    `checkpoint_path` itself does NOT persist across a Colab "Restart
    session": that only resets the Python process, not /content/'s disk, so
    a leftover checkpoint from an earlier dry-run attempt survives restarts.

    `verbose` prints per-conversation and per-50-turns progress, since this
    otherwise gives zero output until fully done.

    Returns {"accuracy": float, "total": int, "correct": int, "details": [...]}
    """
    from pipeline import HippoVoicePipeline

    if pipeline is None:
        pipeline = HippoVoicePipeline(llm_client=llm_client, text_only=True)

    conversations = load_locomo(data_path)[:num_conversations]

    fingerprint = {
        "model_name": getattr(pipeline.llm, "model_name", None),
        "backend": getattr(pipeline.llm, "_backend", None),
        "num_conversations": num_conversations,
        "max_qa_per_conversation": max_qa_per_conversation,
        "include_adversarial": include_adversarial,
    }

    correct = 0
    total = 0
    details = []
    start_index = 0

    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            state = json.load(f)
        if state.get("fingerprint") != fingerprint:
            print(
                f"WARNING: ignoring checkpoint at {checkpoint_path} -- it was written "
                f"under a different setup ({state.get('fingerprint')}) than this run "
                f"({fingerprint}). Starting fresh. Delete the file to silence this."
            )
        else:
            correct = state["correct"]
            total = state["total"]
            details = state["details"]
            start_index = state["next_conversation_index"]
            if verbose:
                print(
                    f"Resuming from checkpoint: {start_index}/{len(conversations)} "
                    f"conversations already done ({correct}/{total} correct so far)"
                )

    for i, conv in enumerate(conversations):
        if i < start_index:
            continue

        # Fresh pipeline per conversation -- memory shouldn't leak across conversations
        conv_pipeline = HippoVoicePipeline(llm_client=pipeline.llm, text_only=True)

        turns = _flatten_conversation(conv["conversation"])
        if verbose:
            print(f"Conversation {i + 1}/{len(conversations)}: ingesting {len(turns)} turns...")
        for t, turn_text in enumerate(turns):
            conv_pipeline.ingest_text_turn(turn_text)
            if verbose and (t + 1) % 50 == 0:
                print(f"  ingested {t + 1}/{len(turns)} turns")

        qa_pairs = conv.get("qa", [])
        if not include_adversarial:
            qa_pairs = [qa for qa in qa_pairs if qa.get("category") != ADVERSARIAL_CATEGORY]
        if max_qa_per_conversation:
            qa_pairs = qa_pairs[:max_qa_per_conversation]

        for qa in qa_pairs:
            question = qa.get("question", "")
            gold_raw = qa.get("answer", qa.get("adversarial_answer", ""))
            gold_answer = str(gold_raw).lower().strip()
            if not question or not gold_answer:
                continue

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

        if verbose:
            print(f"  conversation {i + 1} done -- {correct}/{total} correct so far")

        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "fingerprint": fingerprint,
                    "correct": correct,
                    "total": total,
                    "details": details,
                    "next_conversation_index": i + 1,
                }, f)

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
