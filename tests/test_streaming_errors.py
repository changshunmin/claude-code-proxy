"""Streaming failures must end as Anthropic SSE error events.

Once ``StreamingResponse`` has started, raising ``HTTPException`` cannot be
turned into an HTTP error response: Starlette raises
``RuntimeError: Caught handled exception, but response already started.``
and uvicorn logs a noisy ASGI traceback. Upstream failures must instead be
reported inside the stream, the way the official Anthropic API does.
"""

import json
import logging
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import httpx
import pytest
from fastapi import HTTPException
from openai import APIConnectionError

from src.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest

logger = logging.getLogger("test.streaming_errors")


def make_request():
    return ClaudeMessagesRequest(
        model="deepseek-flash",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )


async def failing_stream(exc):
    raise exc
    yield  # pragma: no cover - makes this an async generator


def parse_sse(chunk):
    lines = chunk.strip().split("\n")
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    return lines[0][len("event: "):], json.loads(lines[1][len("data: "):])


async def collect_exchange(exc):
    stream = convert_openai_streaming_to_claude_with_cancellation(
        failing_stream(exc),
        make_request(),
        logger,
        None,
        None,
        "test-request-id",
    )
    chunks = [chunk async for chunk in stream]
    return [parse_sse(chunk) for chunk in chunks]


async def test_connection_error_mid_stream_becomes_sse_error_event():
    events = await collect_exchange(
        HTTPException(status_code=500, detail="Connection error.")
    )

    assert events[0][0] == "message_start"
    event, data = events[-1]
    assert event == "error"
    assert data["type"] == "error"
    assert data["error"]["type"] == "api_error"
    assert data["error"]["message"] == "Connection error."


@pytest.mark.parametrize(
    "status_code,error_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (599, "api_error"),
    ],
)
async def test_http_status_maps_to_anthropic_error_type(status_code, error_type):
    events = await collect_exchange(
        HTTPException(status_code=status_code, detail=f"upstream {status_code}")
    )

    event, data = events[-1]
    assert event == "error"
    assert data["error"]["type"] == error_type


async def test_context_overflow_keeps_prompt_too_long_message():
    events = await collect_exchange(
        HTTPException(
            status_code=400,
            detail="prompt is too long: 262145 tokens > 262144 maximum",
        )
    )

    event, data = events[-1]
    assert event == "error"
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["message"] == "prompt is too long: 262145 tokens > 262144 maximum"


class _FakeCompletions:
    async def create(self, **kwargs):
        request = httpx.Request("POST", "http://upstream.example/v1/chat/completions")
        try:
            raise httpx.ConnectError("Connection refused")
        except httpx.ConnectError as cause:
            raise APIConnectionError(request=request) from cause


class _FakeChat:
    completions = _FakeCompletions()


class _FakeOpenAIClient:
    chat = _FakeChat()


async def test_streaming_client_logs_underlying_exception_cause(caplog):
    client = OpenAIClient(api_key="sk-test", base_url="http://upstream.example/v1")
    client.client = _FakeOpenAIClient()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as exc_info:
            await client.create_chat_completion_stream(
                {"model": "deepseek-flash", "messages": []}, "test-request-id"
            ).__anext__()

    assert exc_info.value.status_code == 500
    assert "APIConnectionError" in caplog.text
    assert "ConnectError" in caplog.text
    assert "Connection refused" in caplog.text
