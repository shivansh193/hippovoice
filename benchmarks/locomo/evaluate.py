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
import re
import time
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


def _current_commit_hash() -> str:
    """
    Short git commit hash of the working tree, or "unknown" outside a repo.

    Included in the checkpoint fingerprint because model_name/backend/run
    params staying the same across a `git pull` is the common case --
    without this, resuming from an existing checkpoint after pulling a real
    code change (e.g. a retrieval fix) would silently skip re-running
    entirely and just replay stale pre-fix results, since nothing else in
    the fingerprint would have changed.
    """
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _flatten_conversation(conv: dict) -> list[str]:
    """
    Flatten all session turns, in session order, into 'Speaker: text' strings.

    Each turn is prefixed with its session's date. LoCoMo turns routinely use
    relative date language ("yesterday", "last Saturday", "next month") that
    can only be resolved against the session's actual calendar date -- stored
    separately in a session_N_date_time key, previously discarded entirely
    here. Without it, a perfectly-retrieved memory like "I went to a support
    group yesterday" is fundamentally unanswerable for a "when did X happen"
    question: the specific date was never in the text to begin with. Roughly
    half of LoCoMo's QA pairs are date questions, so this alone plausibly
    accounts for a large share of any "correct fact, wrong/missing date"
    failures, independent of retrieval or decay behavior.
    """
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    turns = []
    for key in session_keys:
        session_num = key.split("_")[1]
        date = conv.get(f"session_{session_num}_date_time", "")
        date_prefix = f"[{date}] " if date else ""
        for turn in conv[key]:
            text = (turn.get("text") or "").strip()
            if text:
                turns.append(f"{date_prefix}{turn.get('speaker', '')}: {text}")
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
    batch_size: int = 50,
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

    The checkpoint records a fingerprint (LLM model name/backend, run
    parameters, and current git commit) and refuses to resume from one
    written under a different setup -- e.g. a dry-run mock LLM, a different
    num_conversations, or code pulled after the checkpoint was written --
    since silently trusting a stale file there would replay whatever
    (possibly garbage or pre-fix) results it already has instead of
    actually running.
    `checkpoint_path` itself does NOT persist across a Colab "Restart
    session": that only resets the Python process, not /content/'s disk, so
    a leftover checkpoint from an earlier dry-run attempt survives restarts.

    `verbose` prints per-conversation progress, since this otherwise gives
    zero output until fully done.

    `batch_size`, when the pipeline supports `ingest_text_turns_batch`
    (HippoVoicePipeline does), controls how many independent turns get
    extracted in a single forward pass -- and how often progress prints
    during ingestion. Pipelines without that method fall back to one
    ingest_text_turn() call per turn (progress printed every 50 turns).

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
        "commit": _current_commit_hash(),
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
        ingest_start = time.perf_counter()
        if hasattr(conv_pipeline, "ingest_text_turns_batch"):
            # Each turn's extraction is independent of every other turn's --
            # batching them into one forward pass avoids per-turn GPU idle
            # overhead. Storage/decay/turn-order stay fully sequential inside
            # ingest_text_turns_batch, so results are identical either way.
            for b in range(0, len(turns), batch_size):
                chunk = turns[b:b + batch_size]
                conv_pipeline.ingest_text_turns_batch(chunk)
                done = b + len(chunk)
                if verbose:
                    elapsed = time.perf_counter() - ingest_start
                    print(
                        f"  ingested {done}/{len(turns)} turns "
                        f"({elapsed:.0f}s elapsed, {elapsed / done:.2f}s/turn)"
                    )
        else:
            for t, turn_text in enumerate(turns):
                conv_pipeline.ingest_text_turn(turn_text)
                if verbose and (t + 1) % 50 == 0:
                    elapsed = time.perf_counter() - ingest_start
                    print(
                        f"  ingested {t + 1}/{len(turns)} turns "
                        f"({elapsed:.0f}s elapsed, {elapsed / (t + 1):.2f}s/turn)"
                    )

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


def rescore_details(details: list[dict]) -> dict:
    """
    Recompute accuracy from an already-run details list (e.g. loaded from a
    checkpoint file written by a previous run_locomo() call) using the
    current _answer_matches logic, without re-running any ingestion or LLM
    calls. Useful after a scoring-only change (e.g. a matching/tokenization
    fix) when the predictions themselves haven't changed and re-running the
    full (slow, GPU-hungry) benchmark would be wasteful.
    """
    correct = 0
    rescored = []
    for d in details:
        is_correct = _answer_matches(d["predicted"], d["gold"])
        rescored.append({**d, "correct": is_correct})
        correct += int(is_correct)
    total = len(rescored)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total": total,
        "correct": correct,
        "details": rescored,
    }


_WORD_RE = re.compile(r"\w+")


def _answer_matches(predicted: str, gold: str) -> bool:
    if gold in predicted:
        return True
    # Fuzzy: all words in gold answer appear in predicted. Tokenize with a
    # word regex rather than a plain whitespace split -- otherwise a trailing
    # period ("adoption." vs "adoption") silently breaks an otherwise-correct
    # match, since "adoption." and "adoption" are different set members.
    gold_words = set(_WORD_RE.findall(gold))
    pred_words = set(_WORD_RE.findall(predicted))
    overlap = gold_words & pred_words
    return len(overlap) / len(gold_words) >= 0.7 if gold_words else False
