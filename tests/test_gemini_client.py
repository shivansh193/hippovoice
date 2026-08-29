"""
Tests for llm/gemini_client.py's retry/throttle/fallback logic -- the
actual API call is never exercised here (no network, no key needed), but
this behavior is pure logic and fully mockable. Added after real,
confirmed Kaggle-run failures: a transient 503 mid-ingestion (server
error), and hitting the free tier's confirmed 15-requests/minute quota
partway through a 419-turn conversation's extraction calls (429 rate
limit) -- both killed an otherwise-successful run. See the module
docstring for the full trail.
"""

from unittest.mock import MagicMock, patch

import pytest

from llm.gemini_client import GeminiTextLLM, _extract_retry_delay_seconds


def _server_error():
    from google.genai.errors import ServerError
    return ServerError(503, {"error": {"message": "high demand"}})


def _rate_limit_error(retry_delay="13s"):
    from google.genai.errors import ClientError
    return ClientError(429, {
        "error": {
            "code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}],
        }
    })


def _bad_request_error():
    from google.genai.errors import ClientError
    return ClientError(400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}})


def _ok_response(text="ok"):
    r = MagicMock()
    r.text = text
    return r


# ── ServerError (5xx) retry ──────────────────────────────────────────────────

def test_generate_retries_on_server_error_then_succeeds():
    llm = GeminiTextLLM(max_retries=3, retry_base_delay=0.0, fallback_model=None)
    llm._client = MagicMock()
    llm._client.models.generate_content.side_effect = [_server_error(), _server_error(), _ok_response()]

    with patch("time.sleep"):
        result = llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert llm._client.models.generate_content.call_count == 3


def test_generate_does_not_retry_on_success():
    llm = GeminiTextLLM(max_retries=3, retry_base_delay=0.0, fallback_model=None)
    llm._client = MagicMock()
    llm._client.models.generate_content.return_value = _ok_response("first try")

    result = llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert result == "first try"
    assert llm._client.models.generate_content.call_count == 1


# ── 429 rate limit: retry + fallback ─────────────────────────────────────────

def test_generate_retries_on_429_then_succeeds():
    llm = GeminiTextLLM(max_retries=3, retry_base_delay=0.0, fallback_model=None)
    llm._client = MagicMock()
    llm._client.models.generate_content.side_effect = [_rate_limit_error(), _ok_response()]

    with patch("time.sleep"):
        result = llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert llm._client.models.generate_content.call_count == 2


def test_generate_does_not_retry_non_429_client_error():
    """A 400 (bad request) won't fix itself on retry -- confirms it raises
    immediately instead of burning through the retry budget pointlessly."""
    from google.genai.errors import ClientError
    llm = GeminiTextLLM(max_retries=3, retry_base_delay=0.0, fallback_model=None)
    llm._client = MagicMock()
    llm._client.models.generate_content.side_effect = _bad_request_error()

    with patch("time.sleep"):
        with pytest.raises(ClientError):
            llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert llm._client.models.generate_content.call_count == 1


def test_generate_switches_to_fallback_model_after_exhausting_429_retries():
    """Confirmed as a real cost: the free tier's 15 req/min quota is
    tracked per model name, so a genuinely different fallback model gets
    its own fresh budget instead of retrying into the same wall."""
    llm = GeminiTextLLM(max_retries=1, retry_base_delay=0.0, fallback_model="gemini-2.5-flash")
    llm._client = MagicMock()
    # 2 failures against the primary (max_retries=1 -> 2 attempts), then
    # success on the first attempt against the fallback.
    llm._client.models.generate_content.side_effect = [_rate_limit_error(), _rate_limit_error(), _ok_response("from fallback")]

    with patch("time.sleep"):
        result = llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert result == "from fallback"
    assert llm.model_name == "gemini-2.5-flash"  # switch is sticky
    assert llm._client.models.generate_content.call_count == 3


def test_generate_stays_on_fallback_for_subsequent_calls():
    llm = GeminiTextLLM(max_retries=0, retry_base_delay=0.0, fallback_model="gemini-2.5-flash")
    llm._client = MagicMock()
    llm._client.models.generate_content.side_effect = [_rate_limit_error(), _ok_response("fallback-1")]

    with patch("time.sleep"):
        llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    # Second call: model_name is already the fallback, so no primary attempt at all.
    llm._client.models.generate_content.side_effect = [_ok_response("fallback-2")]
    result = llm.generate(system="sys", messages=[{"role": "user", "content": "hi again"}])

    assert result == "fallback-2"
    assert llm._client.models.generate_content.call_args.kwargs["model"] == "gemini-2.5-flash"


def test_generate_raises_after_exhausting_fallback_too():
    llm = GeminiTextLLM(max_retries=0, retry_base_delay=0.0, fallback_model="gemini-2.5-flash")
    llm._client = MagicMock()
    llm._client.models.generate_content.side_effect = _rate_limit_error()

    from google.genai.errors import ClientError
    with patch("time.sleep"):
        with pytest.raises(ClientError):
            llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    # 1 attempt on primary + 1 attempt on fallback (max_retries=0 each) = 2 calls.
    assert llm._client.models.generate_content.call_count == 2


def test_no_fallback_configured_raises_after_retries():
    llm = GeminiTextLLM(max_retries=1, retry_base_delay=0.0, fallback_model=None)
    llm._client = MagicMock()
    llm._client.models.generate_content.side_effect = _rate_limit_error()

    from google.genai.errors import ClientError
    with patch("time.sleep"):
        with pytest.raises(ClientError):
            llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert llm._client.models.generate_content.call_count == 2  # initial + 1 retry, no fallback


# ── retryDelay extraction ─────────────────────────────────────────────────────

def test_extract_retry_delay_seconds_finds_nested_value():
    details = {
        "error": {
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help"},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13.5s"},
            ]
        }
    }
    assert _extract_retry_delay_seconds(details) == 13.5


def test_extract_retry_delay_seconds_returns_none_when_absent():
    assert _extract_retry_delay_seconds({"error": {"message": "no retry info here"}}) is None


# ── throttle ──────────────────────────────────────────────────────────────────

def test_throttle_sleeps_to_maintain_minimum_interval():
    llm = GeminiTextLLM(min_interval_seconds=4.5)
    # _throttle() calls time.time() twice per invocation (once to compute
    # elapsed, once to record the new _last_call_time) -- a fixed return
    # value keeps "now" consistent across both without depending on call order.
    with patch("time.time", return_value=101.0), patch("time.sleep") as mock_sleep:
        llm._last_call_time = 100.0
        llm._throttle()
    mock_sleep.assert_called_once()
    assert mock_sleep.call_args[0][0] == pytest.approx(3.5, abs=0.01)


def test_throttle_does_not_sleep_on_first_call():
    llm = GeminiTextLLM(min_interval_seconds=4.5)
    with patch("time.sleep") as mock_sleep:
        llm._throttle()
    mock_sleep.assert_not_called()
