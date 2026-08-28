"""
Tests for llm/gemini_client.py's retry logic -- the actual API call is
never exercised here (no network, no key needed), but the retry/backoff
behavior itself is pure logic and fully mockable. Added after a real
Kaggle run lost an entire 419-turn conversation's extraction progress to
one transient 503 mid-ingestion -- confirms the fix actually retries
instead of propagating the first failure.
"""

from unittest.mock import MagicMock, patch

import pytest

from llm.gemini_client import GeminiTextLLM


def _server_error():
    from google.genai.errors import ServerError
    return ServerError(503, {"error": {"message": "high demand"}})


def test_generate_retries_on_server_error_then_succeeds():
    llm = GeminiTextLLM(max_retries=3, retry_base_delay=0.0)
    llm._client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "ok"
    llm._client.models.generate_content.side_effect = [_server_error(), _server_error(), mock_response]

    with patch("time.sleep"):
        result = llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert llm._client.models.generate_content.call_count == 3


def test_generate_raises_after_exhausting_retries():
    llm = GeminiTextLLM(max_retries=2, retry_base_delay=0.0)
    llm._client = MagicMock()
    llm._client.models.generate_content.side_effect = _server_error()

    from google.genai.errors import ServerError
    with patch("time.sleep"):
        with pytest.raises(ServerError):
            llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    # max_retries=2 -- initial attempt + 2 retries = 3 calls total
    assert llm._client.models.generate_content.call_count == 3


def test_generate_does_not_retry_on_success():
    llm = GeminiTextLLM(max_retries=3, retry_base_delay=0.0)
    llm._client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "first try"
    llm._client.models.generate_content.return_value = mock_response

    result = llm.generate(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert result == "first try"
    assert llm._client.models.generate_content.call_count == 1
