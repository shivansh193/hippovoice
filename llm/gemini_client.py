"""
GeminiTextLLM -- API-backed alternative to LLMClient (llm/client.py), same
generate(system, messages, max_tokens) -> str interface, so it drops into
anything expecting an LLMClient (extraction, memory/extractor.py's
EXTRACTION_PROMPT calls, etc.) without changes elsewhere.

Exists specifically so running HippoAudioPipeline end-to-end doesn't
require loading a real local model -- Qwen3-4B on this machine's CPU-only
setup takes minutes just to load and is far too slow per-call for a live
multi-turn demo (established this session: ~300s load, ~10s+ per
generation on CPU). Since Gemini's Live API is already the audio-to-audio
backend, using a lightweight Gemini text model for extraction too keeps
the whole pipeline API-based end-to-end with nothing heavy running
locally, consistent with this project's standing rule to never run
real/heavy models on local CPU-only hardware.

Retries on ServerError (5xx) with exponential backoff -- confirmed as a
real, not hypothetical, cost on a Kaggle run: a transient
`503 UNAVAILABLE ... currently experiencing high demand` mid-ingestion
killed an entire conversation's worth of already-completed extraction
calls (over 100 turns in), since run_locomo has no per-turn retry of its
own and the whole run_locomo() call was wrapped in one failure-tracking
step(). Google's own error message says to retry later; this makes that
automatic instead of losing the whole run to one blip.
"""

import time


class GeminiTextLLM:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite",
                 max_retries: int = 4, retry_base_delay: float = 2.0):
        self._api_key = api_key
        self.model_name = model
        self._backend = "gemini-api"
        self._client = None
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()

    def generate(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        self._ensure_client()
        from google.genai import types
        from google.genai.errors import ServerError

        user_content = messages[-1]["content"] if messages else ""
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=config,
                )
                return response.text or ""
            except ServerError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
        raise last_error
