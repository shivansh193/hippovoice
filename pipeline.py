"""
HippoVoicePipeline — end-to-end voice companion with hippocampal memory.

Audio in → STT → memory extract → store → retrieve → LLM → TTS → audio out.
Forgetting cycle runs every DECAY_EVERY turns.

Memory is split by type into two stores with different temporal behavior,
mirroring the consolidation/retrieval distinction in memory research:
  - semantic_memory (type in fact/preference/person): durable facts about
    the person -- who they are, what they prefer, who's in their life. These
    don't decay; they're the companion's persistent model of who it's
    talking to, corrected by newer information rather than aged out.
  - episodic_memory (type == event): specific moments/events. Ebbinghaus
    decay + emotional consolidation apply here -- forgetting most of an
    average day while keeping emotionally significant events is the
    intended behavior for this store specifically.

A single decaying-trace store trying to serve both jobs means durable facts
either get deleted once age crosses FORGET_THRESHOLD, or the decay constant
gets detuned to protect them at the cost of not forgetting noise properly.
Splitting by type removes that tension.
"""

from pathlib import Path

from memory.extractor import extract_turn, extract_memories, tag_emotion_text
from memory.store import HippoMemory
from memory.retriever import hippo_retrieve
from memory.decay import apply_forgetting_cycle
from llm.context import build_system_prompt, BASE_COMPANION_PROMPT

DECAY_EVERY = 10  # apply forgetting cycle every N turns

# memory/extractor.py's EXTRACTION_PROMPT always tags each fragment's type as
# one of these four; SEMANTIC_TYPES + EPISODIC_TYPES fully partition that set.
SEMANTIC_TYPES = {"fact", "preference", "person"}
EPISODIC_TYPES = {"event"}


class HippoVoicePipeline:
    """
    Full pipeline: audio in → audio out, with persistent hippocampal memory.

    text_only=True skips STT/TTS — useful for benchmarking with text input.
    """

    def __init__(
        self,
        llm_client=None,
        text_only: bool = False,
        memory_path: str | None = None,
    ):
        self.text_only = text_only
        self.current_turn = 0
        self.semantic_memory = HippoMemory(
            collection_name="hippovoice_semantic", persist_path=memory_path
        )
        self.episodic_memory = HippoMemory(
            collection_name="hippovoice_episodic", persist_path=memory_path
        )

        # Lazy-load heavy models so unit tests that mock them don't pay load cost
        self._llm = llm_client
        self._stt = None
        self._tts = None

    # ── public API ────────────────────────────────────────────────────────────

    def process_turn(self, audio_path: str, output_path: str) -> str:
        """Full voice turn: audio_path in, WAV response written to output_path."""
        assert not self.text_only, "Call process_text_turn() in text_only mode."

        from stt.transcribe import transcribe_with_embedding
        from tts.synthesize import synthesize

        if self._stt is None:
            from stt.model import load_canary
            self._stt = load_canary()
        if self._tts is None:
            from tts.model import load_fish_tts
            self._tts = load_fish_tts()

        transcript, audio_emb = transcribe_with_embedding(self._stt, audio_path)
        response_text = self._run_memory_and_generate(transcript, audio_emb)
        synthesize(self._tts, response_text, output_path)
        return output_path

    def process_text_turn(self, text: str) -> str:
        """Text-only turn (for benchmarking). Returns response text."""
        import numpy as np
        # Use a zero embedding — prosody boost won't fire, which is correct for text-only
        dummy_emb = np.zeros(1280, dtype=np.float32)
        return self._run_memory_and_generate(text, dummy_emb)

    def ingest_text_turn(self, text: str):
        """Ingest a turn into memory without generating a response (for benchmark setup)."""
        import numpy as np
        dummy_emb = np.zeros(1280, dtype=np.float32)
        self._store_memories(text, dummy_emb)
        self._maybe_decay()
        self.current_turn += 1

    def ingest_text_turns_batch(self, texts: list[str]):
        """
        Ingest many turns via one batched extraction call instead of one
        generate() call per turn -- storage, decay, and turn ordering stay
        fully sequential (identical semantics to calling ingest_text_turn()
        once per text). Each turn's extraction is independent of the
        others, so batching only changes wall-clock cost on the LLM side,
        not the result.
        """
        import numpy as np
        from memory.extractor import extract_memories_batch, tag_emotion_audio

        dummy_emb = np.zeros(1280, dtype=np.float32)
        batch_memories = extract_memories_batch(texts, self.llm)
        for text, new_memories in zip(texts, batch_memories):
            emotion = tag_emotion_audio(text, dummy_emb)
            for m in new_memories:
                m["emotion"] = emotion
                self._add_memory(m)
            self._maybe_decay()
            self.current_turn += 1

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Direct retrieval for benchmark evaluation.

        Queries both stores and combines results -- each contributes up to
        top_k, since they answer different kinds of questions ("what do you
        know about me" vs. "what happened when") rather than competing for
        the same slots.
        """
        semantic_results = self.semantic_memory.search(query, top_k=top_k)
        episodic_results = hippo_retrieve(
            query, self.episodic_memory, self.episodic_memory.graph, self.current_turn, top_k
        )
        return semantic_results + episodic_results

    def save(self, path: str):
        """Persist memory state across sessions."""
        self.semantic_memory.save(str(Path(path, "semantic")))
        self.episodic_memory.save(str(Path(path, "episodic")))
        # Also save turn counter
        import json
        Path(path, "state.json").write_text(
            json.dumps({"current_turn": self.current_turn})
        )

    def load(self, path: str):
        """Restore memory state from a previous session."""
        import json
        self.semantic_memory.load(str(Path(path, "semantic")))
        self.episodic_memory.load(str(Path(path, "episodic")))
        state_file = Path(path) / "state.json"
        if state_file.exists():
            self.current_turn = json.loads(state_file.read_text())["current_turn"]

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_memory_and_generate(self, transcript: str, audio_emb) -> str:
        self._store_memories(transcript, audio_emb)
        retrieved = self.retrieve(transcript, top_k=5)
        system = build_system_prompt(retrieved, BASE_COMPANION_PROMPT)
        response = self.llm.generate(
            system=system,
            messages=[{"role": "user", "content": transcript}],
            max_tokens=256,
        )
        self._maybe_decay()
        self.current_turn += 1
        return response

    def _store_memories(self, text: str, audio_emb):
        new_memories = extract_turn(text, audio_emb, self.llm)
        for m in new_memories:
            self._add_memory(m)

    def _add_memory(self, m: dict):
        m.setdefault("base_weight", 1.0)
        m.setdefault("recall_count", 0)
        m["turn_created"] = self.current_turn
        target = self.semantic_memory if m.get("type") in SEMANTIC_TYPES else self.episodic_memory
        target.add(m)

    def _maybe_decay(self):
        """
        Forgetting cycle -- episodic memory only. Semantic memory (durable
        facts/preferences/people) never decays, so it has nothing to do here.
        """
        if self.current_turn > 0 and self.current_turn % DECAY_EVERY == 0:
            all_memories = self.episodic_memory.get_all()
            active, forgotten = apply_forgetting_cycle(all_memories, self.current_turn, self.llm)

            all_ids = {m["id"] for m in all_memories}
            active_ids = {m["id"] for m in active if "id" in m}
            forgotten_ids = {m["id"] for m in forgotten if "id" in m}
            # Memories that ended up in neither list were merged into a new
            # compressed entry (apply_forgetting_cycle silently drops them
            # from its return value) -- their content now lives on in that
            # entry, so the originals should go too, same as anything
            # explicitly forgotten.
            compressed_away_ids = all_ids - active_ids - forgotten_ids

            for mid in forgotten_ids | compressed_away_ids:
                self.episodic_memory.delete(mid)

            for m in active:
                if "id" not in m:
                    # Newly-created synthetic compressed entry -- persist it.
                    # (Existing active entries are already correctly in the
                    # store and are left untouched.)
                    self.episodic_memory.add(m)

    @property
    def llm(self):
        if self._llm is None:
            from llm.client import LLMClient
            self._llm = LLMClient()
        return self._llm
