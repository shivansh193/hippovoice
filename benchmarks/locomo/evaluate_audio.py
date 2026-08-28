"""
run_locomo_audio -- the real LoCoMo benchmark for HippoAudioPipeline
(Track 2: audio-to-audio + memory), at the same rigor as run_locomo's
text-pipeline runs: same dataset (load_locomo), same conversation
flattening (_flatten_conversation), same F1 scoring (score_answer,
bin_f1_scores) -- all imported from evaluate.py, not reimplemented, so
this can never silently diverge from the methodology every other system's
LoCoMo number in this project already uses. Without this, Track 2's only
evidence was a hand-run 3-turn demo ("what's my dog's name" -> "Max") --
real and genuinely confirmed memory conditioning, but not a number
comparable to HippoVoice's 24.1% avg F1 or any baseline's.

Structurally different from run_locomo in exactly one place, forced by the
audio-to-audio interface itself: a benchmark question has to actually be
spoken to the model, not just handed to conv_pipeline.llm.generate() as a
text string. So the QA step here synthesizes each question to speech
(fresh pyttsx3 engine per call -- see pipeline_audio2audio.py's own
comment on why engine reuse deadlocks pyttsx3's SAPI5 loop on Windows,
confirmed for real during this project's live multi-turn demo) and calls
conv_pipeline.answer_question(), the read-only audio-QA path added to
HippoAudioPipeline specifically for this benchmark (see its docstring).

Ingestion itself stays pure text (conv_pipeline.ingest_text_turn per
turn, same method name/contract as every other baseline) rather than
synthesizing and "speaking" all 369-689 conversation turns: a live
deployment's user really did speak their turns, but a benchmark replaying
a transcript for extraction has no such requirement, and a full audio
round-trip per turn would be a 369-689x needless cost in TTS time and
real-time API calls for content that produces no QA signal on its own.
Ingestion cost here is therefore identical to any text baseline's (one
extraction LLM call per turn) -- only the QA step pays the audio-specific
cost.

Not run at LoCoMo's full 10-conversation scale -- unlike ingestion, each
QA question here is a real Gemini Live API round trip plus a TTS
synthesis, categorically slower than a single conv_pipeline.llm.generate()
call. num_conversations/max_qa_per_conversation default small
deliberately, meant to be widened only after a first real wall-clock
reading, matching this project's "validate cheap before expensive"
discipline (see BUGS.md).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from benchmarks.locomo.evaluate import (
    ADVERSARIAL_CATEGORY,
    _current_commit_hash,
    _flatten_conversation,
    bin_f1_scores,
    load_locomo,
    score_answer,
)


def _synthesize_question_audio(text: str) -> str:
    """
    Fresh pyttsx3 engine per call, matching pipeline_audio2audio.py's own
    fix for the confirmed SAPI5 COM reuse deadlock -- this function gets
    called once per QA pair, i.e. repeatedly within one process, exactly
    the failure condition that bug needs.
    """
    from tts.model import load_tts
    from tts.synthesize import synthesize

    path = tempfile.mktemp(suffix=".wav")
    engine = load_tts()
    synthesize(engine, text, path)
    return path


def run_locomo_audio(
    llm_client,
    audio_model,
    num_conversations: int = 1,
    max_qa_per_conversation: int | None = 10,
    include_adversarial: bool = False,
    data_path: str | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = True,
    decay_lambda: float = 0.001,
    relevance_weight: float = 0.85,
    stm_window: int = 5,
    system_name: str = "HippoAudio",
) -> dict:
    """
    Run the LoCoMo QA benchmark against HippoAudioPipeline.

    llm_client: used for extraction only (memory/extractor.py's
    extract_memories, called by ingest_text_turn), same role llm_client
    plays for every other system's ingestion step.
    audio_model: a loaded AudioToAudioModel (e.g. GeminiLiveAudioModel) --
    shared across all conversations, since it holds no conversation state
    of its own (every respond() call is an independent round trip); only
    each conversation's HippoAudioPipeline (and therefore its memory
    store) is fresh.

    decay_lambda/relevance_weight default to the same LoCoMo-scale values
    run_locomo uses (see that function's docstring for why the pipeline's
    own short-conversation defaults would forget almost everything before
    a 369-663 turn conversation ends).

    Returns the same shape as run_locomo: {"avg_f1", "total", "total_f1",
    "bins", "details"} -- a drop-in match for comparing against any other
    system's result dict from this project.
    """
    conversations = load_locomo(data_path)[:num_conversations]

    fingerprint = {
        "system_name": system_name,
        "extraction_model_name": getattr(llm_client, "model_name", None),
        "audio_model": getattr(audio_model, "model", type(audio_model).__name__),
        "num_conversations": num_conversations,
        "max_qa_per_conversation": max_qa_per_conversation,
        "include_adversarial": include_adversarial,
        "decay_lambda": decay_lambda,
        "relevance_weight": relevance_weight,
        "stm_window": stm_window,
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

    from pipeline_audio2audio import HippoAudioPipeline

    for i, conv in enumerate(conversations):
        if i < start_index:
            continue

        # Fresh pipeline (fresh memory) per conversation -- audio_model/
        # llm_client are shared, matching run_locomo's own "fresh pipeline,
        # shared LLM" pattern.
        conv_pipeline = HippoAudioPipeline(
            audio_model=audio_model,
            llm_client=llm_client,
            decay_lambda=decay_lambda,
            relevance_weight=relevance_weight,
            stm_window=stm_window,
        )

        turns = _flatten_conversation(conv["conversation"])
        if verbose:
            print(f"Conversation {i + 1}/{len(conversations)}: ingesting {len(turns)} turns (text-only)...")
        ingest_start = time.perf_counter()
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

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            gold_raw = qa.get("answer", qa.get("adversarial_answer", ""))
            gold_answer = str(gold_raw).lower().strip()
            category = qa.get("category")
            if not question or not gold_answer:
                continue

            # Retrieved separately (cheap, local -- no API cost) purely for
            # the trace fields below; answer_question does its own
            # retrieve() internally to actually condition generation. Two
            # calls, not a shared one, because answer_question's contract
            # returns (audio_path, transcript) only -- see its docstring.
            retrieved = conv_pipeline.retrieve(question, top_k=5)

            question_audio_path = _synthesize_question_audio(question)
            try:
                _, transcript = conv_pipeline.answer_question(question, question_audio_path)
            finally:
                if os.path.exists(question_audio_path):
                    os.remove(question_audio_path)

            predicted = (transcript or "").lower().strip()
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
                "retrieved_ids": [r.get("id") for r in retrieved],
            })

            if verbose:
                print(f"  QA {qi + 1}/{len(qa_pairs)}  f1={f1:.2f}  "
                      f"gold={gold_answer!r}  predicted={predicted!r}")

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
