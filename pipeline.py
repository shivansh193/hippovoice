"""
HippoVoicePipeline — end-to-end voice companion with hippocampal memory.

Audio in → STT → memory extract → store → retrieve → LLM → TTS → audio out.
Forgetting cycle runs every DECAY_EVERY turns.
"""

import uuid
from pathlib import Path

from memory.extractor import extract_turn, extract_memories, tag_emotion_text
from memory.store import HippoMemory
from memory.retriever import hippo_retrieve
from memory.decay import apply_forgetting_cycle
from llm.context import build_system_prompt, BASE_COMPANION_PROMPT

DECAY_EVERY = 10  # apply forgetting cycle every N turns


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
        self.memory = HippoMemory(persist_path=memory_path)

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

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Direct retrieval for benchmark evaluation."""
        return hippo_retrieve(query, self.memory, self.memory.graph, self.current_turn, top_k)

    def save(self, path: str):
        """Persist memory state across sessions."""
        self.memory.save(path)
        # Also save turn counter
        import json
        Path(path, "state.json").write_text(
            json.dumps({"current_turn": self.current_turn})
        )

    def load(self, path: str):
        """Restore memory state from a previous session."""
        import json
        self.memory.load(path)
        state_file = Path(path) / "state.json"
        if state_file.exists():
            self.current_turn = json.loads(state_file.read_text())["current_turn"]

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_memory_and_generate(self, transcript: str, audio_emb) -> str:
        self._store_memories(transcript, audio_emb)
        retrieved = hippo_retrieve(
            transcript, self.memory, self.memory.graph, self.current_turn, top_k=5
        )
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
            m.setdefault("base_weight", 1.0)
            m.setdefault("recall_count", 0)
            m["turn_created"] = self.current_turn
            self.memory.add(m)

    def _maybe_decay(self):
        if self.current_turn > 0 and self.current_turn % DECAY_EVERY == 0:
            all_memories = self.memory.get_all()
            active, forgotten = apply_forgetting_cycle(all_memories, self.current_turn, self.llm)
            for m in forgotten:
                mid = m.get("id")
                if mid:
                    self.memory.delete(mid)

    @property
    def llm(self):
        if self._llm is None:
            from llm.client import LLMClient
            self._llm = LLMClient()
        return self._llm
