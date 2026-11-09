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


def test_tool_choice_any_maps_to_auto_like_official_endpoint():
    # Official DeepSeek compatibility: "any" is treated as "auto".
    out = convert_claude_to_openai(
        _request(tool_choice={"type": "any"}, tools=[TOOL]), model_manager
    )
    assert out["tool_choice"] == "auto"


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


def test_max_tokens_optional_like_official_endpoint():
    out = convert_claude_to_openai(
        ClaudeMessagesRequest(
            model="deepseek-flash", messages=[{"role": "user", "content": "hi"}]
        ),
        model_manager,
    )
    assert 0 < out["max_tokens"] <= model_manager.config.model_max_output


def test_server_tools_are_skipped_not_failing():
    out = convert_claude_to_openai(
        _request(
            tools=[
                TOOL,
                {"type": "web_search_20250305", "name": "web_search"},
            ]
        ),
        model_manager,
    )
    assert [t["function"]["name"] for t in out["tools"]] == ["Bash"]


def test_url_image_is_passed_through_as_image_url():
    from src.conversion.request_converter import convert_claude_user_message

    msg = ClaudeMessage(
        role="user",
        content=[
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/a.png"},
            },
            {"type": "text", "text": "what is this"},
        ],
    )
    converted = convert_claude_user_message(msg)
    assert {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
    } in converted["content"]


def test_text_after_tool_result_is_preserved():
    msg = ClaudeMessage(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "toolu_5",
                "content": "ok",
            },
            {"type": "text", "text": "and then?"},
        ],
    )
    messages = convert_claude_tool_results(msg)
    assert messages[0]["role"] == "tool"
    assert messages[1] == {"role": "user", "content": "and then?"}


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


# --- mid-conversation system messages (CC 2.1.x beta) ------------------------

def test_system_role_message_is_accepted_and_folded_into_system_prompt():
    """Claude Code 2.1.x sends SessionStart hook context as a message with
    role="system" when it probes the beta mid-conversation-system shape.  The
    proxy must accept it instead of returning 422 and folding it into the
    leading system prompt keeps the instructions in front of the conversation."""
    req = ClaudeMessagesRequest(
        model="claude-opus-5-5",
        max_tokens=128,
        system="TOP-LEVEL",
        messages=[
            ClaudeMessage(role="user", content="hi"),
            ClaudeMessage(
                role="system",
                content=[{"type": "text", "text": "HOOK CONTEXT"}],
            ),
        ],
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert converted["messages"][0]["role"] == "system"
    assert "TOP-LEVEL" in converted["messages"][0]["content"]
    assert "HOOK CONTEXT" in converted["messages"][0]["content"]
    assert [m["role"] for m in converted["messages"][1:]] == ["user"]
    assert converted["messages"][1]["content"] == "hi"


def test_system_role_message_without_top_level_system_creates_one():
    req = ClaudeMessagesRequest(
        model="claude-opus-5-5",
        max_tokens=128,
        messages=[
            ClaudeMessage(role="system", content="HOOK CONTEXT"),
            ClaudeMessage(role="user", content="hi"),
        ],
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert converted["messages"][0] == {
        "role": "system",
        "content": "HOOK CONTEXT",
    }
    assert converted["messages"][1]["content"] == "hi"


def test_system_role_message_ignores_non_text_blocks():
    req = ClaudeMessagesRequest(
        model="claude-opus-5-5",
        max_tokens=128,
        messages=[
            ClaudeMessage(
                role="system",
                content=[
                    {"type": "text", "text": "KEEP"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": B64},
                    },
                ],
            ),
            ClaudeMessage(role="user", content="hi"),
        ],
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert converted["messages"][0]["content"] == "KEEP"
