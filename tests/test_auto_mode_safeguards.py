"""Auto mode server-side classifier capability advertisement.

Claude Code's auto mode pairs the ``anthropic-beta: dangerous-tool-use-*``
header with a ``safeguards`` request field and expects a ``safeguard_results``
field in the response (top-level for non-streaming, ``message_delta.delta``
for streaming).  This relay translates to an OpenAI-protocol upstream that
cannot run Anthropic's server-side check, so the honest answer is
``status.type == "unsupported"``: Claude Code then falls back to its own
classifier instead of assuming something dropped the result.
"""

import json
import logging
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import pytest

from src.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.models.claude import ClaudeMessagesRequest

logger = logging.getLogger("test.auto_mode_safeguards")

UNSUPPORTED = [{"type": "dangerous_tool_use", "status": {"type": "unsupported"}}]
SAFEGUARDS = [{"type": "dangerous_tool_use", "classifier_context": {"tool_uses": []}}]


def make_request(**overrides):
    payload = {
        "model": "deepseek-flash",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    payload.update(overrides)
    return ClaudeMessagesRequest(**payload)


def make_openai_response():
    return {
        "id": "msg_1",
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {},
    }


def test_request_model_keeps_safeguards_field():
    request = make_request(safeguards=SAFEGUARDS)
    assert request.safeguards == SAFEGUARDS


def test_non_streaming_declares_server_check_unsupported():
    response = convert_openai_to_claude_response(
        make_openai_response(), make_request(stream=False, safeguards=SAFEGUARDS)
    )
    assert response["safeguard_results"] == UNSUPPORTED


def test_non_streaming_omits_results_when_not_requested():
    response = convert_openai_to_claude_response(
        make_openai_response(), make_request(stream=False)
    )
    assert "safeguard_results" not in response


def parse_sse(chunk):
    lines = chunk.strip().split("\n")
    return lines[0][len("event: "):], json.loads(lines[1][len("data: "):])


async def stream_lines(*lines):
    for line in lines:
        yield line


async def collect_stream(request):
    lines = [
        'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    ]
    stream = convert_openai_streaming_to_claude_with_cancellation(
        stream_lines(*lines), request, logger, None, None, "test-request-id"
    )
    return [parse_sse(chunk) async for chunk in stream]


async def test_streaming_declares_server_check_unsupported():
    events = await collect_stream(make_request(safeguards=SAFEGUARDS))
    deltas = [data for event, data in events if event == "message_delta"]
    assert deltas[-1]["delta"]["safeguard_results"] == UNSUPPORTED


async def test_streaming_omits_results_when_not_requested():
    events = await collect_stream(make_request())
    deltas = [data for event, data in events if event == "message_delta"]
    assert "safeguard_results" not in deltas[-1]["delta"]


def test_duplicate_safeguard_types_collapse_to_one_result():
    request = make_request(
        safeguards=[
            {"type": "dangerous_tool_use", "classifier_context": {"n": 1}},
            {"type": "dangerous_tool_use", "classifier_context": {"n": 2}},
        ]
    )
    response = convert_openai_to_claude_response(make_openai_response(), request)
    # Claude Code treats more than one dangerous_tool_use result as
    # unrecognized, so the relay must answer with a single verdict per type.
    assert response["safeguard_results"] == UNSUPPORTED
