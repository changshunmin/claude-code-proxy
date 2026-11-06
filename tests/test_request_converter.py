"""Unit tests for Claude -> OpenAI request conversion."""

import base64
import logging

from src.core.model_manager import model_manager

from src.conversion.request_converter import (
    convert_claude_to_openai,
    convert_claude_tool_results,
    estimate_claude_token_count,
    estimate_text_tokens,
)
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest

B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64).decode()


def test_tool_result_image_is_sent_as_image_not_text():
    msg = ClaudeMessage(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": B64,
                        },
                    }
                ],
            }
        ],
    )

    messages = convert_claude_tool_results(msg)

    tool_msg = messages[0]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "toolu_1"
    assert B64 not in tool_msg["content"]

    image_msgs = [m for m in messages if m["role"] == "user"]
    assert len(image_msgs) == 1
    image_blocks = [
        b for b in image_msgs[0]["content"] if b.get("type") == "image_url"
    ]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"] == f"data:image/png;base64,{B64}"


def test_tool_result_text_and_image_keeps_text_in_tool_message():
    msg = ClaudeMessage(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": [
                    {"type": "text", "text": "rendered 900x600"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": B64,
                        },
                    },
                ],
            }
        ],
    )

    messages = convert_claude_tool_results(msg)

    assert messages[0]["role"] == "tool"
    assert messages[0]["content"] == "rendered 900x600"
    assert B64 not in str(messages[0])
    user_msg = [m for m in messages if m["role"] == "user"][0]
    assert any(b.get("type") == "image_url" for b in user_msg["content"])


def test_tool_result_plain_text_unchanged():
    msg = ClaudeMessage(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "toolu_3",
                "content": "ok",
            }
        ],
    )

    messages = convert_claude_tool_results(msg)

    assert messages == [{"role": "tool", "tool_call_id": "toolu_3", "content": "ok"}]


# --- tool_choice mapping -----------------------------------------------------

def _request(**overrides):
    data = {
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
    }
    data.update(overrides)
    return ClaudeMessagesRequest(**data)


TOOL = {"name": "Bash", "description": "run a command", "input_schema": {"type": "object"}}


def test_tool_choice_any_maps_to_required():
    out = convert_claude_to_openai(
        _request(tool_choice={"type": "any"}, tools=[TOOL]), model_manager
    )
    assert out["tool_choice"] == "required"


def test_tool_choice_none_maps_to_none():
    out = convert_claude_to_openai(
        _request(tool_choice={"type": "none"}, tools=[TOOL]), model_manager
    )
    assert out["tool_choice"] == "none"


def test_tool_choice_specific_function_is_preserved():
    out = convert_claude_to_openai(
        _request(tool_choice={"type": "tool", "name": "Bash"}, tools=[TOOL]),
        model_manager,
    )
    assert out["tool_choice"] == {
        "type": "function",
        "function": {"name": "Bash"},
    }


# --- debug logging -----------------------------------------------------------

def test_debug_log_includes_tools_and_tool_choice(caplog):
    with caplog.at_level(logging.DEBUG, logger="src.conversion.request_converter"):
        convert_claude_to_openai(
            _request(tool_choice={"type": "auto"}, tools=[TOOL]), model_manager
        )
    assert '"tools"' in caplog.text
    assert '"tool_choice"' in caplog.text


# --- token estimation --------------------------------------------------------

def test_count_tokens_does_not_undercount_chinese():
    # 4 CJK chars must cost at least one token each; punctuation is cheap.
    assert estimate_text_tokens("你好，世界") >= 4


def test_count_tokens_includes_tool_schemas():
    plain = estimate_claude_token_count(
        None, [ClaudeMessage(role="user", content="hi")], None
    )
    with_tools = estimate_claude_token_count(
        None, [ClaudeMessage(role="user", content="hi")], [TOOL]
    )
    assert with_tools > plain


def test_count_tokens_ignores_base64_image_payload():
    msg = ClaudeMessage(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "toolu_4",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": B64},
                    }
                ],
            }
        ],
    )
    assert estimate_claude_token_count(None, [msg], None) < 100
