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
import string
import time
import urllib.request
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_CACHE_PATH = Path(__file__).parent / "locomo10.json"

# category 5 = adversarial/unanswerable questions (ground truth is
# "adversarial_answer", not "answer" -- the correct model behavior is to
# recognize the question isn't answerable from the conversation). Excluded
# by default since they need different scoring logic than exact/fuzzy match.
ADVERSARIAL_CATEGORY = 5
# category 1 = multi-hop -- LoCoMo's official scorer splits both prediction
# and gold on commas and takes a mean-of-max F1 across sub-answers for these
# specifically (see multi_hop_f1 below). Every other category is scored as
# one undecomposed string. Splitting on comma for non-multi-hop questions
# would be wrong: many category 2 (temporal) gold answers are dates that
# happen to contain a comma as punctuation ("19 January, 2023"), not a list
# separator -- naively splitting those would fabricate two fake sub-answers.
MULTI_HOP_CATEGORY = 1

_ps = PorterStemmer()


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


def debug_extraction_for_turns(conv: dict, dia_ids: list[str], llm_client) -> list[dict]:
    """
    Run real extraction on specific dia_id turns from a conversation (by
    id, not "first N turns" -- a turn evidencing a QA pair can be anywhere,
    e.g. deep into session 2), and return exactly what the LLM extracted --
    content and type -- without ingesting into any pipeline/store.

    Useful for directly answering "did the model even extract this fact,
    and how did it classify it?" for a specific known evidence turn, e.g.
    checking whether an implicitly-stated identity fact ("the transgender
    stories were so inspiring") gets extracted at all, and whether it's
    tagged as a durable fact/person/preference (semantic store, never
    decays) or an event (episodic store, subject to forgetting) --
    resolving that with real data instead of guessing from final QA
    accuracy alone, which conflates extraction quality with everything
    downstream of it.
    """
    from memory.extractor import extract_memories_batch

    turns_by_id = {}
    for key in conv["conversation"]:
        if key.startswith("session_") and not key.endswith("_date_time"):
            session_num = key.split("_")[1]
            date = conv["conversation"].get(f"session_{session_num}_date_time", "")
            date_prefix = f"[{date}] " if date else ""
            for turn in conv["conversation"][key]:
                text = (turn.get("text") or "").strip()
                if text:
                    turns_by_id[turn["dia_id"]] = f"{date_prefix}{turn.get('speaker', '')}: {text}"

    found_ids = [d for d in dia_ids if d in turns_by_id]
    selected_turns = [turns_by_id[d] for d in found_ids]
    if not selected_turns:
        return []

    batch_memories = extract_memories_batch(selected_turns, llm_client)
    return [
        {"dia_id": d, "turn": t, "extracted": m}
        for d, t, m in zip(found_ids, selected_turns, batch_memories)
    ]


def run_locomo(
    pipeline=None,
    llm_client=None,
    num_conversations: int = 10,
    max_qa_per_conversation: int | None = None,
    max_turns_per_conversation: int | None = None,
    include_adversarial: bool = False,
    data_path: str | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = True,
    batch_size: int = 50,
    decay_lambda: float = 0.001,
    relevance_weight: float = 0.85,
    top_k: int = 5,
    pipeline_factory=None,
    system_name: str | None = None,
) -> dict:
    """
    Run LoCoMo evaluation.

    For each conversation:
      1. Ingest all turns (flattened across sessions, in order) into a fresh
         pipeline's memory.
      2. For each QA pair, retrieve context and ask the LLM to answer.
      3. Compare against ground truth (exact match + fuzzy).

    `decay_lambda`/`relevance_weight` default to values tuned for LoCoMo's
    conversation length (369-663 turns), not HippoVoicePipeline's own
    defaults (tuned for ~90-100 turn conversations). Confirmed directly:
    with the pipeline default decay rate, even a strongly emotion-boosted
    episodic memory crosses FORGET_THRESHOLD by ~turn 173 and gets
    physically deleted -- so almost nothing survives to the end of a real
    LoCoMo conversation regardless of extraction or retrieval quality.
    decay_lambda=0.001 keeps a plain neutral memory above COMPRESS_THRESHOLD
    even at 663 turns elapsed; relevance_weight=0.85 (up from the pipeline
    default of 0.65) leans on relevance to do more of the ranking work now
    that availability barely discriminates between memories any more.
    `top_k` controls how many memories are retrieved per question in the QA
    loop below -- previously hardcoded to 5 inline with no way to override
    it for tuning. Applies regardless of `pipeline_factory`, since every
    system's QA loop calls `.retrieve(question, top_k=...)` the same way.

    `decay_lambda`/`relevance_weight` only apply to the default
    HippoVoicePipeline construction -- ignored if `pipeline_factory` is
    given, since baselines don't accept them.

    `pipeline_factory`, if given, is a callable `(llm_client) -> pipeline`
    used instead of the default HippoVoicePipeline construction -- e.g.
    `lambda llm: Mem0Baseline(llm_client=llm)` to run the *real* LoCoMo QA
    benchmark (not just the synthetic signal/noise one) against a baseline,
    under the identical LLM, data, and scoring as HippoVoice. This is the
    only way to compare against Mem0/A-MEM/NaiveRAG in a way that isolates
    the memory architecture's contribution rather than conflating it with
    whatever LLM their own published numbers used. `system_name` is
    required alongside a custom `pipeline_factory` (used in the checkpoint
    fingerprint, so a HippoVoice run and a Mem0-style run against the same
    `checkpoint_path` can never be silently confused for each other).

    `max_qa_per_conversation` caps QA pairs per conversation -- useful for a
    cheap smoke test (a full run is ~150-250 QA pairs x 10 conversations,
    each needing an LLM call, which is slow on a single T4).

    `max_turns_per_conversation` caps how many flattened turns get ingested
    (truncates from the start of the conversation, not sampled) -- added
    after a real run confirmed an API-backed llm_client can hit the free
    tier's daily/per-minute quota partway through a single 369-689 turn
    conversation's ingestion, since each turn is one extraction call with
    no way to combine multiple turns into one request (unlike a local
    model, where ingest_text_turns_batch's "batching" is a single GPU
    forward pass over many turns; an API call has no equivalent -- each
    turn still costs one real request against the provider's quota
    regardless of how the calls are grouped in Python). None (default)
    ingests the whole conversation, unchanged from before this parameter
    existed.

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

    Scoring uses LoCoMo's actual published methodology (stemmed token F1,
    category-branched -- see score_answer), not a simple boolean match.

    Each detail also records the exact "context" string handed to the LLM
    and "retrieved_ids" (which memory ids were surfaced) for that question
    -- the memory store itself isn't persisted anywhere past this call, so
    this is the only way to later trace "was the right fact ever in
    context?" (see print_qa_trace) without a fresh run.

    Returns {"avg_f1": float, "total": int, "total_f1": float,
             "bins": {"near_zero": int, "partial": int, "high": int},
             "details": [...]}
    """
    # Deliberately NOT a module-level or unconditional import here: pipeline.py
    # pulls in memory/store.py, which imports chromadb/networkx/sentence-
    # transformers at module level. A custom pipeline_factory (WeightEditBaseline,
    # Mem0Baseline, ...) never touches HippoVoicePipeline or HippoMemory at all,
    # so forcing that import unconditionally forces those dependencies onto
    # every baseline-comparison run too. Confirmed as a real crash: a Kaggle
    # notebook for the weight-editing benchmark deliberately didn't install
    # chromadb (genuinely unneeded for that baseline) and hit
    # `ModuleNotFoundError: No module named 'chromadb'` from this import line,
    # before ever reaching any code that actually needed it.
    if pipeline_factory is None:
        from pipeline import HippoVoicePipeline

        if system_name is not None:
            raise ValueError("system_name is only meaningful alongside a custom pipeline_factory")
        system_name = "HippoVoice"

        def pipeline_factory(llm):
            return HippoVoicePipeline(
                llm_client=llm, text_only=True,
                decay_lambda=decay_lambda, relevance_weight=relevance_weight,
            )
    elif system_name is None:
        raise ValueError("system_name is required alongside a custom pipeline_factory")

    if pipeline is None:
        if llm_client is not None:
            # `pipeline` past this point is only ever read via `.llm` (see
            # the comment above the pipeline_factory branch for why a full
            # HippoVoicePipeline isn't needed just for that).
            class _LLMOnly:
                pass
            pipeline = _LLMOnly()
            pipeline.llm = llm_client
        else:
            from pipeline import HippoVoicePipeline
            pipeline = HippoVoicePipeline(llm_client=llm_client, text_only=True)

    conversations = load_locomo(data_path)[:num_conversations]

    fingerprint = {
        "system_name": system_name,
        "model_name": getattr(pipeline.llm, "model_name", None),
        "backend": getattr(pipeline.llm, "_backend", None),
        "num_conversations": num_conversations,
        "max_qa_per_conversation": max_qa_per_conversation,
        "max_turns_per_conversation": max_turns_per_conversation,
        "include_adversarial": include_adversarial,
        "decay_lambda": decay_lambda,
        "relevance_weight": relevance_weight,
        "top_k": top_k,
        "commit": _current_commit_hash(),
    }

    total_f1 = 0.0
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
            total_f1 = state["total_f1"]
            total = state["total"]
            details = state["details"]
            start_index = state["next_conversation_index"]
            if verbose:
                avg_so_far = total_f1 / total if total else 0.0
                print(
                    f"Resuming from checkpoint: {start_index}/{len(conversations)} "
                    f"conversations already done (avg F1 {avg_so_far:.3f} over {total})"
                )

    for i, conv in enumerate(conversations):
        if i < start_index:
            continue

        # Fresh pipeline per conversation -- memory shouldn't leak across conversations
        conv_pipeline = pipeline_factory(pipeline.llm)

        turns = _flatten_conversation(conv["conversation"])
        if max_turns_per_conversation:
            turns = turns[:max_turns_per_conversation]
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
            category = qa.get("category")
            if not question or not gold_answer:
                continue

            if hasattr(conv_pipeline, "retrieve"):
                retrieved = conv_pipeline.retrieve(question, top_k=top_k)
                context = build_qa_context(retrieved)
            else:
                # Confirmed as a real crash on a live run: WeightEditBaseline
                # has no .retrieve() by design -- the edited model's own
                # weights ARE the memory, so there's no separate retrieval
                # step feeding context into generation (see its module
                # docstring). This QA loop assumed every system implements
                # .retrieve() until a real run actually reached this line
                # for the first time (every earlier attempt failed during
                # ingestion before ever getting here). No context to inject;
                # the whole point of that baseline is testing whether the
                # edited model answers correctly from zero retrieved context.
                retrieved = []
                context = ""

            predicted = conv_pipeline.llm.generate(
                system=(
                    "Answer the question using only the provided context. "
                    "Give a specific, direct answer (a name, label, date, or "
                    "short phrase) rather than a general description. "
                    "Be concise — one sentence or less."
                ),
                messages=[
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                max_tokens=80,
            ).lower().strip()

            f1 = score_answer(predicted, gold_answer, category)
            total_f1 += f1
            total += 1
            details.append({
                "question": question,
                "gold": gold_answer,
                "predicted": predicted,
                "category": category,
                "f1": round(f1, 4),
                "correct": f1 >= 0.7,
                # Exact context handed to the LLM for this question -- lets
                # a later trace answer "was the right fact ever in context?"
                # without needing another real run, since the memory store
                # itself isn't persisted anywhere past this call.
                "context": context,
                "retrieved_ids": [r.get("id") for r in retrieved],
            })

        if verbose:
            avg_so_far = total_f1 / total if total else 0.0
            print(f"  conversation {i + 1} done -- avg F1 {avg_so_far:.3f} over {total}")

        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "fingerprint": fingerprint,
                    "total_f1": total_f1,
                    "total": total,
                    "details": details,
                    "next_conversation_index": i + 1,
                }, f)

    avg_f1 = total_f1 / total if total > 0 else 0.0
    return {
        "avg_f1": round(avg_f1, 4),
        "total": total,
        "total_f1": round(total_f1, 4),
        "bins": bin_f1_scores(details),
        "details": details,
    }


def build_qa_context(retrieved_memories: list[dict]) -> str:
    return "\n".join(f"- {m.get('content', '')}" for m in retrieved_memories)


def normalize_answer(s: str) -> str:
    """
    Exact reproduction of LoCoMo's official normalize_answer() --
    github.com/snap-research/locomo/blob/main/task_eval/evaluation.py.
    Strip commas, remove articles (a/an/the/and), strip punctuation,
    lowercase, collapse whitespace. Used by f1_score before stemming/
    comparing tokens.
    """
    s = s.replace(",", "")

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Exact reproduction of LoCoMo's official f1_score() -- stemmed,
    normalized token-level F1 (SQuAD-style). Used directly for every
    category except multi-hop (1), where multi_hop_f1 applies instead.
    """
    prediction_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """
    Exact reproduction of LoCoMo's official multi-hop f1() -- splits both
    prediction and ground truth on commas; for each ground-truth sub-answer,
    takes the max f1_score() across all predicted fragments, then averages
    across ground-truth sub-answers. For a non-comma ground truth this
    reduces exactly to f1_score() (both become single-item lists).

    Used only for category 1 (multi-hop) questions -- comma-splitting a
    date like "19 January, 2023" for any other category would fabricate two
    fake sub-answers out of what's just date punctuation.
    """
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return sum(
        max(f1_score(p, gt) for p in predictions) for gt in ground_truths
    ) / len(ground_truths)


def score_answer(predicted: str, gold: str, category: int | None) -> float:
    """
    Score one predicted answer against gold using LoCoMo's actual published
    methodology, branching on category exactly as the official evaluator
    does -- not one uniform formula for every question type.
    """
    if category == ADVERSARIAL_CATEGORY:
        pred_lower = predicted.lower()
        return 1.0 if ("no information available" in pred_lower or "not mentioned" in pred_lower) else 0.0
    if category == MULTI_HOP_CATEGORY:
        return multi_hop_f1(predicted, gold)
    return f1_score(predicted, gold)


def bin_f1_scores(details: list[dict]) -> dict:
    """
    Bucket per-question F1 scores into near_zero (<0.2), partial
    (0.2-0.7), and high (>=0.7). The average F1 alone can't distinguish
    "everything is mediocre" from "half genuinely correct, half genuinely
    wrong" -- both can produce the same mean -- so the distribution shape
    is the actually diagnostic signal.
    """
    bins = {"near_zero": 0, "partial": 0, "high": 0}
    for d in details:
        f1 = d.get("f1", 0.0)
        if f1 < 0.2:
            bins["near_zero"] += 1
        elif f1 < 0.7:
            bins["partial"] += 1
        else:
            bins["high"] += 1
    return bins


def _lookup_categories(details: list[dict], data_path: str | None = None) -> dict[str, int | None]:
    """Map each question string to its LoCoMo category by matching against
    the full dataset -- needed to correctly rescore older checkpoints
    written before category was recorded per-detail."""
    conversations = load_locomo(data_path)
    category_by_question = {}
    for conv in conversations:
        for qa in conv.get("qa", []):
            category_by_question[qa.get("question", "")] = qa.get("category")
    return category_by_question


def rescore_details(details: list[dict], data_path: str | None = None) -> dict:
    """
    Recompute F1 scores from an already-run details list (e.g. loaded from
    a checkpoint file written by a previous run_locomo() call) using the
    current scoring logic, without re-running any ingestion or LLM calls.
    Useful after a scoring-only change (e.g. this file's matcher) when the
    predictions themselves haven't changed and re-running the full (slow,
    GPU-hungry) benchmark would be wasteful.

    Older checkpoints written before category was recorded per-detail get
    their category backfilled by matching question text against the full
    LoCoMo dataset (falls back to plain f1_score, i.e. non-multi-hop
    scoring, for any question that can't be matched).
    """
    needs_category_lookup = any("category" not in d for d in details)
    category_by_question = _lookup_categories(details, data_path) if needs_category_lookup else None

    rescored = []
    total_f1 = 0.0
    for d in details:
        category = d.get("category")
        if category is None and category_by_question is not None:
            category = category_by_question.get(d["question"])
        f1 = score_answer(d["predicted"], d["gold"], category)
        rescored.append({**d, "category": category, "f1": round(f1, 4), "correct": f1 >= 0.7})
        total_f1 += f1

    total = len(rescored)
    return {
        "avg_f1": round(total_f1 / total, 4) if total else 0.0,
        "total": total,
        "total_f1": round(total_f1, 4),
        "bins": bin_f1_scores(rescored),
        "details": rescored,
    }


def print_qa_trace(details: list[dict], question_substring: str) -> None:
    """
    Print the full trace for questions matching a substring: the exact
    context handed to the model, the raw prediction, gold, F1, and category.

    This is the tool for answering "was the right fact ever in context, or
    did retrieval never surface it?" without needing another real run --
    run_locomo() now logs "context" per question in details, so this works
    directly off an existing checkpoint. If "context" is missing (an older
    checkpoint written before this was added), says so explicitly rather
    than silently printing nothing useful.
    """
    matches = [d for d in details if question_substring.lower() in d["question"].lower()]
    if not matches:
        print(f"No question matching {question_substring!r} found.")
        return
    for d in matches:
        print("=" * 70)
        print(f"Q: {d['question']}")
        print(f"gold:      {d['gold']!r}")
        print(f"predicted: {d['predicted']!r}")
        print(f"f1={d.get('f1')}  category={d.get('category')}")
        print("-" * 70)
        if "context" in d:
            print("Context handed to the model:")
            print(d["context"] or "(empty -- nothing retrieved)")
        else:
            print("(context not logged -- this checkpoint predates context logging; rerun to capture it)")
        print()
