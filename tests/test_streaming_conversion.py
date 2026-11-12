"""Unit tests for the OpenAI -> Anthropic streaming conversion."""

import json
import logging

from src.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.models.claude import ClaudeMessagesRequest

logger = logging.getLogger("test.streaming_conversion")


def make_request():
    return ClaudeMessagesRequest(
        model="deepseek-flash",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )


def chunk(delta, finish_reason=None, usage=None):
    payload = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}"


async def stream_of(chunks):
    for item in chunks:
        yield item


async def collect(chunks):
    stream = convert_openai_streaming_to_claude_with_cancellation(
        stream_of(chunks), make_request(), logger, None, None, "test-rid"
    )
    events = []
    async for raw in stream:
        lines = raw.strip().split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        events.append(
            (lines[0][len("event: "):], json.loads(lines[1][len("data: "):]))
        )
    return events


async def test_text_after_tool_call_opens_a_new_block():
    """Each SSE content block index must be started once and stopped once;
    reusing a closed index makes Claude Code's parser abort the turn."""
    events = await collect(
        [
            chunk({"content": "hello"}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "Bash"},
                        }
                    ]
                }
            ),
            chunk({"content": "world"}),
            chunk({}, finish_reason="stop"),
        ]
    )

    starts = [
        (data["index"], data["content_block"]["type"])
        for event, data in events
        if event == "content_block_start"
    ]
    assert starts == [(0, "text"), (1, "tool_use"), (2, "text")]

    for index in (0, 1, 2):
        assert (
            sum(
                1
                for event, data in events
                if event == "content_block_start" and data["index"] == index
            )
            == 1
        )
        assert (
            sum(
                1
                for event, data in events
                if event == "content_block_stop" and data["index"] == index
            )
            == 1
        )


async def test_reasoning_after_text_closes_text_before_new_block():
    events = await collect(
        [
            chunk({"content": "answer"}),
            chunk({"reasoning": "hm"}),
            chunk({}, finish_reason="stop"),
        ]
    )

    block_events = [
        (event, data.get("index"))
        for event, data in events
        if event in ("content_block_start", "content_block_stop")
    ]
    assert block_events == [
        ("content_block_start", 0),
        ("content_block_stop", 0),
        ("content_block_start", 1),
        ("content_block_stop", 1),
    ]


async def test_null_usage_counters_do_not_abort_stream():
    events = await collect(
        [
            chunk({"content": "hi"}),
            chunk(
                {},
                finish_reason="stop",
                usage={
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "prompt_tokens_details": {"cached_tokens": None},
                },
            ),
        ]
    )

    assert events[-1][0] == "message_stop"
    message_delta = [data for event, data in events if event == "message_delta"][0]
    assert message_delta["usage"]["input_tokens"] == 0
    assert message_delta["usage"]["output_tokens"] == 0
    assert message_delta["usage"]["cache_read_input_tokens"] == 0
