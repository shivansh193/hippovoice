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

Three real, confirmed failure modes drive everything below -- none of
this is speculative hardening:

1. Retries on ServerError (5xx) with exponential backoff: a transient
   `503 UNAVAILABLE ... currently experiencing high demand` mid-ingestion
   killed an entire conversation's worth of already-completed extraction
   calls, since run_locomo has no per-turn retry of its own.

2. Proactive throttling + retry on 429 RESOURCE_EXHAUSTED: the free
   tier's actual quota, confirmed directly from a real 429 response
   (`quotaValue: '15'`, `generate_content_free_tier_requests`, model
   `gemini-3.5-flash-lite`) is 15 requests/minute. A 419-turn LoCoMo
   conversation makes one extraction call per turn with no pacing, so it
   blows through that quota every run, not occasionally -- confirmed
   twice on real runs (WeightEdit's ingestion and HippoAudio's, both hit
   429 partway through). Reactive retry alone would just hit the same
   wall again a few requests later; min_interval_seconds paces calls to
   stay under the quota in the first place. Public docs beyond RPM are
   gated behind a per-account AI Studio dashboard this project has no
   access to, so this only encodes the one number confirmed directly by
   the API itself -- see BUGS.md for that trail.

3. Fallback model on exhausted retries: quota is tracked per model name
   (the 429's own `quotaDimensions.model` field confirms this), so
   switching to a genuinely different model resets the budget instead of
   retrying into the same wall. gemini-2.5-flash and
   gemini-flash-lite-latest were both confirmed live and working as of
   this writing; gemini-2.5-flash was chosen over the "latest" alias
   specifically because both of gemini-3.5-flash-lite's own 404 messages
   for older models point at "use gemini-3.5-flash-lite" -- suggesting
   flash-lite aliases may resolve to (and share quota with) the exact
   model already being avoided, which a differently-named model in a
   different tier (flash, not flash-lite) can't. The switch is sticky
   for the rest of this instance's life, not per-call -- retrying the
   primary again next call would rediscover the same exhausted quota.

Also sets an explicit per-request timeout (HttpOptions.timeout, in
milliseconds) so a stalled request can't block an entire ingestion loop
indefinitely -- extraction calls are short (a few hundred output tokens
at most), so 30s is generous, not tight.
"""

import re
import time


def _extract_retry_delay_seconds(details) -> float | None:
    """Search an APIError's .details (the raw parsed error response) for a
    RetryInfo.retryDelay hint (e.g. "13s") without assuming exactly where
    in the nested structure it lives -- google.genai wraps the response
    differently across error shapes, and guessing one fixed path risks
    silently never finding it on a shape that shifts slightly."""
    if isinstance(details, dict):
        if "retryDelay" in details:
            match = re.match(r"([\d.]+)", str(details["retryDelay"]))
            if match:
                return float(match.group(1))
        for value in details.values():
            found = _extract_retry_delay_seconds(value)
            if found is not None:
                return found
    elif isinstance(details, list):
        for item in details:
            found = _extract_retry_delay_seconds(item)
            if found is not None:
                return found
    return None


class GeminiTextLLM:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite",
                 max_retries: int = 4, retry_base_delay: float = 2.0,
                 min_interval_seconds: float = 4.5, timeout_ms: int = 30_000,
                 fallback_model: str | None = "gemini-2.5-flash"):
        self._api_key = api_key
        self.model_name = model
        self._backend = "gemini-api"
        self._client = None
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        # 60s / 15 requests-per-minute (the free tier's own stated limit,
        # confirmed from a real 429 response) + a small buffer.
        self.min_interval_seconds = min_interval_seconds
        self.timeout_ms = timeout_ms
        self.fallback_model = fallback_model
        self._last_call_time = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            http_options = types.HttpOptions(timeout=self.timeout_ms)
            self._client = (
                genai.Client(api_key=self._api_key, http_options=http_options)
                if self._api_key else genai.Client(http_options=http_options)
            )

    def _throttle(self):
        if self._last_call_time is not None:
            elapsed = time.time() - self._last_call_time
            wait = self.min_interval_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_call_time = time.time()

    def generate(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        self._ensure_client()
        from google.genai import types
        from google.genai.errors import ClientError, ServerError

        user_content = messages[-1]["content"] if messages else ""
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
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
                    time.sleep(self.retry_base_delay * (2 ** attempt))
            except ClientError as e:
                if e.code != 429:
                    raise  # a genuinely bad request (400/401/...) retrying won't fix
                last_error = e
                if attempt < self.max_retries:
                    delay = _extract_retry_delay_seconds(e.details)
                    if delay is None:
                        delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay + 1.0)  # +1s buffer past the API's own hint

        # Retries on the current model exhausted. If a different, real
        # fallback model is configured and we haven't already switched to
        # it, switch permanently (not just for this call) -- quota is
        # tracked per model name, so a fresh model means a fresh budget,
        # and retrying the exhausted one again next call would just
        # rediscover the same wall.
        if self.fallback_model and self.model_name != self.fallback_model:
            self.model_name = self.fallback_model
            return self.generate(system, messages, max_tokens)

        raise last_error
