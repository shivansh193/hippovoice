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
"""


class GeminiTextLLM:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite"):
        self._api_key = api_key
        self.model_name = model
        self._backend = "gemini-api"
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()

    def generate(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        self._ensure_client()
        from google.genai import types

        user_content = messages[-1]["content"] if messages else ""
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""
