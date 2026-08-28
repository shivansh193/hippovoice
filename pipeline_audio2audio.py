"""
HippoAudioPipeline -- memory-integrated pipeline for audio-to-audio models.

Track 2 of the project (see BUGS.md / colab_track2.ipynb). colab_track2.ipynb's
Mini-Omni pass proved memory *capture* works off an audio-to-audio model's
output -- extraction, storage, and retrieval all work fine on a transcript the
model itself produces. What it explicitly did not do is memory *conditioning*:
feeding retrieved context back into that model's own generation, because
Mini-Omni's public inference API (OmniInference/A1_A2) has no obvious
system-prompt or text-context hook the way a text LLM's
generate(system=..., messages=...) does.

This module is that next step, built model-agnostic on purpose: the
AudioToAudioModel interface below makes zero assumptions about any specific
model's internals (no assumption it accepts a text prefix in its input_ids,
no assumption it exposes a system-prompt parameter), so whichever real model
gets chosen after this dry-run validates the plumbing (Gemini Live API,
Moshi, Qwen2.5-Omni, ...) plugs in as one small adapter class without
touching this pipeline's logic at all.

Context gets injected the one way that works regardless of a model's input
API: synthesizing a short spoken summary and prepending it to the audio the
model actually hears, so the model receives it as part of the same
utterance instead of needing a text-injection hook into its own
architecture. This was flagged as the "probably try this first" option in
the Track 2 doc precisely because it needs zero changes to whatever model
ends up plugged in.

Two distinct memory tiers feed that summary, mirroring the actual
architectural distinction in a normal LLM chat session:
  - STM (short-term): the last `stm_window` raw turns, kept verbatim, the
    same role a text LLM's message history / context window plays. This
    didn't exist before -- extraction ran on every turn and the raw text
    was discarded immediately after, so even the exact wording of what was
    just said was gone a moment later. Real, but session-scoped and
    unweighted by salience -- it ages out by being pushed off the end of a
    fixed-size window, not by decay.
  - LTM (long-term): the existing HippoRAG retrieval + Ebbinghaus decay
    system, completely unchanged. Salience-weighted, persistent across
    however many turns, survives across sessions via save()/load().
Both get synthesized into the same spoken context clip; STM covers "what
was just said," LTM covers "what's been retained from further back."

MockAudioToAudioModel exists so this whole pipeline -- extraction, storage,
retrieval, context-audio construction -- can be validated end-to-end on a
local CPU machine with zero model downloads and zero GPU, before spending
any AWS instance time on which real model to commit to. Established
discipline this whole project: validate cheap before validate expensive.
"""

from memory.store import HippoMemory
from memory.extractor import extract_memories
from memory.retriever import hippo_retrieve, DEFAULT_RELEVANCE_WEIGHT, NAME_MATCH_BONUS, \
    _extract_proper_nouns, _content_matches_names, _name_match_ids
from memory.store import _cosine_similarity
from memory.decay import apply_forgetting_cycle
from memory.scorer import DEFAULT_DECAY_LAMBDA

DECAY_EVERY = 10

# Mirrors pipeline.py's SEMANTIC_TYPES/EPISODIC_TYPES exactly -- extraction
# always tags a fragment as one of these four regardless of which pipeline
# calls it, so the routing rule can't diverge between the two pipelines.
SEMANTIC_TYPES = {"fact", "preference", "person"}
EPISODIC_TYPES = {"event"}


class AudioToAudioModel:
    """
    Interface any audio-to-audio backend must implement to plug into
    HippoAudioPipeline. One adapter class per real model -- see
    colab_track2.ipynb for how the Mini-Omni case would map onto this
    (OmniInference.load() -> load(); load_audio + get_input_ids_whisper +
    A1_A2 -> respond()).
    """

    def load(self):
        raise NotImplementedError

    def respond(self, audio_path: str) -> tuple[str, str]:
        """
        audio_path is the (possibly context-prefixed) input audio the model
        should actually hear. Returns (response_audio_path, transcript) --
        transcript is whatever text representation the model exposes for
        the turn (e.g. Mini-Omni's parallel "thinking" text-token stream).
        Callers that can't get a transcript this way should pass user text
        into process_turn() explicitly rather than relying on this return
        value for extraction.
        """
        raise NotImplementedError


class MockAudioToAudioModel(AudioToAudioModel):
    """
    Dry-run stand-in -- no model weights, no download, no GPU. Deterministic
    so tests can assert on exact behavior, and records every audio path it
    was called with so tests can inspect whether context got prepended.
    """

    def __init__(self):
        self.loaded = False
        self.calls: list[str] = []

    def load(self):
        self.loaded = True

    def respond(self, audio_path: str) -> tuple[str, str]:
        self.calls.append(audio_path)
        return (f"{audio_path}.mock_response.wav", f"[mock reply to {audio_path}]")


def _concatenate_audio(first_path: str, second_path: str, output_path: str) -> None:
    """
    Write first_path immediately followed by second_path to output_path, at
    second_path's sample rate -- second_path is always the real audio going
    to whatever downstream model is in use (see _build_context_audio), so
    its rate is the one that actually matters; first_path (the synthesized
    context clip) gets resampled to match if needed.

    Confirmed this mismatch is real, not theoretical, on a local dry run:
    pyttsx3's SAPI5 voice on this machine outputs 22050Hz while the
    synthetic 16000Hz test fixture stood in for a model's expected input
    rate -- concatenating those without resampling would have silently
    produced distorted, wrong-speed audio. Caught here for free instead of
    discovering it on a paid AWS run.
    """
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample

    data1, sr1 = sf.read(first_path)
    data2, sr2 = sf.read(second_path)
    if sr1 != sr2:
        target_len = int(len(data1) * sr2 / sr1)
        data1 = resample(data1, target_len)
    combined = np.concatenate([data1, data2])
    sf.write(output_path, combined, sr2)


class HippoAudioPipeline:
    """
    Memory-integrated audio-to-audio pipeline. Same dual-store architecture
    as HippoVoicePipeline (memory/store.py, memory/scorer.py, memory/decay.py,
    memory/retriever.py all completely unchanged -- see pipeline.py's
    docstring for why semantic/episodic are split and scored differently).
    This class differs only in the generation step: an AudioToAudioModel
    instead of STT -> LLM.generate() -> TTS, plus the context-audio
    construction step neither pipeline needed before now.
    """

    def __init__(
        self,
        audio_model: AudioToAudioModel,
        llm_client=None,
        memory_path: str | None = None,
        decay_lambda: float | None = None,
        relevance_weight: float | None = None,
        tts_engine=None,
        stm_window: int = 5,
    ):
        self.audio_model = audio_model
        self.decay_lambda = decay_lambda if decay_lambda is not None else DEFAULT_DECAY_LAMBDA
        self.relevance_weight = relevance_weight if relevance_weight is not None else DEFAULT_RELEVANCE_WEIGHT

        self.current_turn = 0
        self.semantic_memory = HippoMemory(collection_name="hippoaudio_semantic", persist_path=memory_path)
        self.episodic_memory = HippoMemory(collection_name="hippoaudio_episodic", persist_path=memory_path)

        # STM: last stm_window raw turns, verbatim, unweighted -- see module
        # docstring for why this is architecturally distinct from LTM rather
        # than just "more retrieval." A deque(maxlen=...) ages out the
        # oldest turn automatically once full; that's the whole mechanism,
        # no salience/decay math involved, same as a text LLM's message
        # history sliding out of a fixed context window.
        from collections import deque
        self.stm_window = stm_window
        self.recent_turns = deque(maxlen=stm_window)

        self._llm = llm_client
        self._tts = tts_engine  # lazy-loaded on first real use; injectable for tests

    # ── public API ────────────────────────────────────────────────────────

    def process_turn(self, user_audio_path: str, user_text: str) -> tuple[str, str]:
        """
        One full turn. user_text is a required, explicit param rather than
        derived from user_audio_path here -- real deployments get it from
        whichever transcript channel the chosen model exposes (or a parallel
        lightweight STT call); keeping it explicit decouples this pipeline
        from any one model's specific transcript mechanism, and keeps
        dry-run tests trivial (no audio-to-text step needed to validate
        memory logic with a mock model).
        """
        # 1. Retrieve BEFORE generating -- this is the actual conditioning
        #    step colab_track2.ipynb's Mini-Omni pass never reached.
        #    STM (recent_turns) reflects turns *before* this one -- the
        #    current turn's own audio is the actual input, it doesn't also
        #    need to be echoed back to the model as context about itself.
        retrieved = self.retrieve(user_text, top_k=5)

        # 2. Build the context-injected audio the model will actually hear,
        #    combining STM (recent verbatim turns) and LTM (retrieved).
        input_audio = self._build_context_audio(retrieved, list(self.recent_turns), user_audio_path)

        # 3. Real audio-to-audio turn.
        response_audio, transcript = self.audio_model.respond(input_audio)

        # 4. Extract + store from what the user said -- same as
        #    HippoVoicePipeline, extraction always runs on the user's turn,
        #    never the model's own reply. STM updates after generation for
        #    the same reason retrieval happens before it: this turn becomes
        #    "recent history" for the *next* turn, not for itself.
        self._store_memories(user_text)
        self.recent_turns.append(user_text)
        self._maybe_decay()
        self.current_turn += 1
        return response_audio, transcript

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Deliberately mirrors HippoVoicePipeline.retrieve() in pipeline.py --
        same semantic/episodic merge, same name-match bonus, same reasoning
        for why semantic candidates are scored by relevance alone (see that
        method's docstring for the full explanation, including the ~90%
        noise-rate regression a naive blended score reproduced). Kept as a
        separate copy rather than imported cross-module so this pipeline
        stays independently testable without pulling in pipeline.py's STT/
        TTS-specific code paths; if this drifts from pipeline.py's version,
        that's a bug to fix in both places, not a sanctioned divergence.
        """
        query_names = _extract_proper_nouns(query)

        semantic_pool = self.semantic_memory.search(query, top_k=top_k * 2)
        found_ids = {r["id"] for r in semantic_pool if "id" in r}
        extra_ids = [mid for mid in _name_match_ids(query_names, self.semantic_memory) if mid not in found_ids]
        if extra_ids:
            extra = [(mid, self.semantic_memory.get_by_id(mid)) for mid in extra_ids]
            extra = [(mid, m) for mid, m in extra if m is not None]
            if extra:
                texts = [query] + [m.get("content", "") for _, m in extra]
                embeddings = self.semantic_memory.embedder.encode(texts)
                query_emb, mem_embs = embeddings[0], embeddings[1:]
                for (mid, m), mem_emb in zip(extra, mem_embs):
                    distance = 1.0 - _cosine_similarity(query_emb, mem_emb)
                    semantic_pool.append({**m, "id": mid, "_distance": distance})

        for r in semantic_pool:
            relevance = 1.0 - r.get("_distance", 0.0)
            name_bonus = NAME_MATCH_BONUS if _content_matches_names(r.get("content", ""), query_names) else 0.0
            r["_relevance"] = round(relevance, 6)
            r["current_salience"] = 1.0
            r["_score"] = round(relevance + name_bonus, 6)

        episodic_pool = hippo_retrieve(
            query, self.episodic_memory, self.episodic_memory.graph, self.current_turn, top_k * 2,
            relevance_weight=self.relevance_weight, decay_lambda=self.decay_lambda,
        )

        combined = sorted(semantic_pool + episodic_pool, key=lambda r: r.get("_score", 0.0), reverse=True)
        return combined[:top_k]

    def save(self, path: str):
        from pathlib import Path
        import json
        self.semantic_memory.save(str(Path(path, "semantic")))
        self.episodic_memory.save(str(Path(path, "episodic")))
        Path(path, "state.json").write_text(json.dumps({
            "current_turn": self.current_turn,
            "recent_turns": list(self.recent_turns),
        }))

    def load(self, path: str):
        from pathlib import Path
        import json
        self.semantic_memory.load(str(Path(path, "semantic")))
        self.episodic_memory.load(str(Path(path, "episodic")))
        state_file = Path(path) / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            self.current_turn = state["current_turn"]
            # older saves won't have this key -- treat as empty STM rather
            # than KeyError, since a save from before STM existed is still
            # a valid save, just without recent-turn history to restore.
            self.recent_turns.extend(state.get("recent_turns", []))

    # ── internal ──────────────────────────────────────────────────────────

    def _build_context_audio(self, retrieved: list[dict], recent_turns: list[str], user_audio_path: str) -> str:
        """
        Model-agnostic context injection: synthesize a short spoken summary
        of STM (recent_turns, verbatim) and LTM (retrieved, salience-ranked)
        and prepend it to the user's actual input audio. Returns
        user_audio_path unchanged when both are empty (turn 1 specifically
        -- no prior turns for STM, nothing stored yet for LTM) -- no point
        synthesizing and concatenating silence-equivalent content.
        """
        if not retrieved and not recent_turns:
            return user_audio_path

        parts = []
        if recent_turns:
            parts.append("earlier in this conversation: " + "; ".join(recent_turns))
        if retrieved:
            parts.append("some other things to remember: " + "; ".join(
                r.get("content", "") for r in retrieved if r.get("content")
            ))
        summary = ". ".join(parts)

        # Confirmed for real on a live multi-turn run (not just suspected):
        # reusing one pyttsx3 engine instance across repeated synthesize()
        # calls reliably deadlocks its SAPI5 COM loop on Windows. Turn 1
        # never needs this method (nothing to inject yet), so a cached
        # engine's *first* real use always looks fine in a quick smoke
        # test -- the deadlock only shows up on the second synthesis call
        # in the same process, i.e. turn 3+ of a real conversation, which
        # is exactly the multi-turn case this whole method exists for. Real
        # usage therefore builds a fresh engine every call; self._tts stays
        # purely a test-injection seam (existing tests mock this out
        # entirely, so they never exercise the reuse path this comment
        # warns about).
        if self._tts is not None:
            engine = self._tts
        else:
            from tts.model import load_tts
            engine = load_tts()

        import tempfile
        import os
        from tts.synthesize import synthesize

        context_audio_path = tempfile.mktemp(suffix=".wav")
        synthesize(engine, summary, context_audio_path)

        combined_path = tempfile.mktemp(suffix=".wav")
        _concatenate_audio(context_audio_path, user_audio_path, combined_path)
        os.remove(context_audio_path)
        return combined_path

    def _store_memories(self, text: str):
        new_memories = extract_memories(text, self.llm)
        for m in new_memories:
            self._add_memory(m)

    def _add_memory(self, m: dict):
        m.setdefault("base_weight", 1.0)
        m.setdefault("recall_count", 0)
        m.setdefault("emotion", {"label": "neutral", "intensity": 0.3})
        m["turn_created"] = self.current_turn
        target = self.semantic_memory if m.get("type") in SEMANTIC_TYPES else self.episodic_memory
        target.add(m)

    def _maybe_decay(self):
        if self.current_turn > 0 and self.current_turn % DECAY_EVERY == 0:
            all_memories = self.episodic_memory.get_all()
            active, forgotten = apply_forgetting_cycle(
                all_memories, self.current_turn, self.llm, decay_lambda=self.decay_lambda
            )
            all_ids = {m["id"] for m in all_memories}
            active_ids = {m["id"] for m in active if "id" in m}
            forgotten_ids = {m["id"] for m in forgotten if "id" in m}
            compressed_away_ids = all_ids - active_ids - forgotten_ids
            for mid in forgotten_ids | compressed_away_ids:
                self.episodic_memory.delete(mid)
            for m in active:
                if "id" not in m:
                    self.episodic_memory.add(m)

    @property
    def llm(self):
        if self._llm is None:
            from llm.client import LLMClient
            self._llm = LLMClient()
        return self._llm
