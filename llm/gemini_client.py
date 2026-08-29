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

Real, confirmed failure modes drive everything below -- none of this is
speculative hardening:

1. Retries on ServerError (5xx) with exponential backoff: a transient
   `503 UNAVAILABLE ... currently experiencing high demand` mid-ingestion
   killed an entire conversation's worth of already-completed extraction
   calls, since run_locomo has no per-turn retry of its own.

2. Proactive throttling + retry on 429 RESOURCE_EXHAUSTED: the free
   tier's actual per-minute quota, confirmed directly from a real 429
   response (`quotaValue: '15'`, `generate_content_free_tier_requests`,
   model `gemini-3.5-flash-lite`) is 15 requests/minute. A 419-turn
   LoCoMo conversation makes one extraction call per turn with no pacing,
   so it blows through that every run, not occasionally -- confirmed
   twice. min_interval_seconds paces calls to stay under the quota rather
   than just reacting after every violation.

3. Retries do NOT help a *daily* quota, and this project found that out
   for real rather than assuming a fallback model would help. The first
   fallback tried, gemini-2.5-flash, seemed reasonable (confirmed live
   and working, genuinely different model/tier from the exhausted one)
   -- but its own 429 turned out to be
   `GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: '20'`:
   a **20-requests-per-day** cap, not per-minute. That's unusable for a
   419-turn conversation regardless of how carefully retries are paced --
   no amount of waiting seconds or minutes fixes a daily cap. This isn't
   published anywhere public (Google gates per-model quota numbers behind
   a personal AI Studio dashboard this project has no access to), so it
   could only be discovered by actually hitting it. Fixed two ways:
   `_is_daily_quota` detects a RESOURCE_EXHAUSTED whose quotaId contains
   "PerDay" and skips retrying it with a short delay (pointless -- it
   will fail identically every time within the same day), moving straight
   to the fallback-or-raise decision instead of wasting the retry budget.
   `fallback_model` now defaults to None rather than a specific model
   name: this project doesn't have a verified-good high-volume fallback
   candidate, and defaulting to one that's actively worse than the
   primary (as gemini-2.5-flash turned out to be) would make things worse
   silently. Pass a fallback explicitly once one is actually confirmed
   suitable for the call volume in question.

Also sets an explicit per-request timeout (HttpOptions.timeout, in
milliseconds) so a stalled request can't block an entire ingestion loop
indefinitely -- extraction calls are short (a few hundred output tokens
at most), so 30s is generous, not tight.

See also: benchmarks/locomo/evaluate.py's and evaluate_audio.py's
max_turns_per_conversation -- the actual structural fix for the call-
volume problem this project ran into (there's no way to combine multiple
turns into one API request the way local-model batching combines them
into one GPU forward pass; each turn is a real, separate request against
whatever quota exists, so the only way to use meaningfully fewer requests
is to ingest fewer turns).
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


def _is_daily_quota(details) -> bool:
    """True if a RESOURCE_EXHAUSTED error's .details names a per-day quota
    (quotaId containing "PerDay") rather than a per-minute one. Confirmed
    for real: gemini-2.5-flash's free-tier quotaId is
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier" -- retrying that
    with any delay under 24 hours fails identically every time, so this
    is checked before spending the retry budget pointlessly."""
    if isinstance(details, dict):
        if "PerDay" in str(details.get("quotaId", "")):
            return True
        return any(_is_daily_quota(v) for v in details.values())
    elif isinstance(details, list):
        return any(_is_daily_quota(item) for item in details)
    return False


class GeminiTextLLM:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite",
                 max_retries: int = 4, retry_base_delay: float = 2.0,
                 min_interval_seconds: float = 4.5, timeout_ms: int = 30_000,
                 fallback_model: str | None = None):
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
        # No default fallback -- see module docstring for why guessing one
        # backfired for real (gemini-2.5-flash's 20-requests-PER-DAY cap).
        # Pass one explicitly only once it's actually confirmed to handle
        # the call volume in question.
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
                if _is_daily_quota(e.details):
                    # No amount of waiting seconds/minutes fixes a daily cap
                    # -- stop retrying immediately and go straight to the
                    # fallback-or-raise decision below instead of burning
                    # the retry budget on guaranteed-identical failures.
                    break
                if attempt < self.max_retries:
                    delay = _extract_retry_delay_seconds(e.details)
                    if delay is None:
                        delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay + 1.0)  # +1s buffer past the API's own hint

        # Retries on the current model exhausted (or skipped, for a daily
        # cap). If a different, real fallback model is configured and we
        # haven't already switched to it, switch permanently (not just for
        # this call) -- quota is tracked per model name, so a fresh model
        # means a fresh budget, and retrying the exhausted one again next
        # call would just rediscover the same wall.
        if self.fallback_model and self.model_name != self.fallback_model:
            self.model_name = self.fallback_model
            return self.generate(system, messages, max_tokens)

        raise last_error
