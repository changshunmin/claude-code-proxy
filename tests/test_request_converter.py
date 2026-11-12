"""Unit tests for Claude -> OpenAI request conversion."""

import base64
import logging

import pytest
from pydantic import ValidationError

from src.core.model_manager import model_manager

from src.conversion.request_converter import (
    convert_claude_to_openai,
    convert_claude_tool_results,
    estimate_claude_token_count,
    estimate_text_tokens,
    estimate_tokens,
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


def test_tool_choice_without_tools_is_not_forwarded():
    # OpenAI-compatible upstreams reject tool_choice when tools is absent.
    out = convert_claude_to_openai(_request(tool_choice={"type": "auto"}), model_manager)
    assert "tools" not in out
    assert "tool_choice" not in out


def test_tool_choice_with_only_server_tools_is_not_forwarded():
    out = convert_claude_to_openai(
        _request(
            tool_choice={"type": "auto"},
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        ),
        model_manager,
    )
    assert "tools" not in out
    assert "tool_choice" not in out


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


def test_base64_image_without_media_type_defaults_to_png():
    from src.conversion.request_converter import convert_claude_user_message

    msg = ClaudeMessage(
        role="user",
        content=[{"type": "image", "source": {"type": "base64", "data": B64}}],
    )
    converted = convert_claude_user_message(msg)
    assert converted["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{B64}"},
        }
    ]


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


def test_estimate_tokens_ignores_base64_image_payload():
    huge = "A" * 2_000_000
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{huge}"},
                },
            ],
        }
    ]
    assert estimate_tokens(messages) < 1000


def test_huge_base64_image_does_not_collapse_max_tokens():
    huge = "A" * 2_000_000
    req = ClaudeMessagesRequest(
        model="deepseek-flash",
        max_tokens=64000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": huge,
                        },
                    },
                    {"type": "text", "text": "describe this"},
                ],
            }
        ],
    )
    out = convert_claude_to_openai(req, model_manager)
    assert out["max_tokens"] > model_manager.config.min_tokens_limit


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


def test_unknown_content_block_type_is_accepted_and_dropped():
    """Claude Code keeps adding metadata block types (tool_removal, ...).
    The model layer must accept them and the converter must drop them."""
    req = _request(
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "gone"},
                    {
                        "type": "tool_removal",
                        "tool": {"type": "tool_reference", "name": "X"},
                    },
                ],
            },
            {"role": "user", "content": "hi"},
        ]
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert converted["messages"][0]["content"] == "gone"
    assert [m["role"] for m in converted["messages"][1:]] == ["user"]


def test_unknown_block_without_text_is_not_forwarded():
    req = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "tool_removal", "tool": {"type": "tool_reference"}}
                ],
            },
            {"role": "user", "content": "hi"},
        ]
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert [m["content"] for m in converted["messages"]] == ["hi"]


def test_malformed_known_block_still_fails_validation():
    # The catch-all must not mask errors in block types we understand.
    with pytest.raises(ValidationError):
        _request(messages=[{"role": "user", "content": [{"type": "text"}]}])


def test_tool_result_after_interleaved_system_message_is_kept():
    """CC 2.1.x can insert a role="system" metadata message between a
    tool_use turn and its tool_result. The pair must stay adjacent after
    conversion or the upstream drops/rejects the tool output."""
    req = ClaudeMessagesRequest(
        model="deepseek-flash",
        max_tokens=128,
        messages=[
            ClaudeMessage(role="user", content="hi"),
            ClaudeMessage(
                role="assistant",
                content=[
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}
                ],
            ),
            ClaudeMessage(
                role="system",
                content=[
                    {"type": "text", "text": "hook"},
                    {
                        "type": "tool_addition",
                        "tool": {"type": "tool_reference", "name": "X"},
                    },
                ],
            ),
            ClaudeMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }
                ],
            ),
        ],
    )

    converted = convert_claude_to_openai(req, model_manager)
    roles = [m["role"] for m in converted["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert converted["messages"][3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "ok",
    }


def test_tool_addition_only_user_message_is_not_forwarded():
    """A metadata-only block leaves no content; an empty OpenAI message would
    make the upstream reject the whole request."""
    req = ClaudeMessagesRequest(
        model="deepseek-flash",
        max_tokens=128,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    {
                        "type": "tool_addition",
                        "tool": {"type": "tool_reference", "name": "X"},
                    }
                ],
            ),
            ClaudeMessage(role="user", content="hi"),
        ],
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert [m["content"] for m in converted["messages"]] == ["hi"]


def test_assistant_message_with_only_tool_addition_is_not_forwarded():
    req = ClaudeMessagesRequest(
        model="deepseek-flash",
        max_tokens=128,
        messages=[
            ClaudeMessage(role="user", content="hi"),
            ClaudeMessage(
                role="assistant",
                content=[
                    {
                        "type": "tool_addition",
                        "tool": {"type": "tool_reference", "name": "X"},
                    }
                ],
            ),
            ClaudeMessage(role="user", content="again"),
        ],
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert [m["role"] for m in converted["messages"]] == ["user", "user"]


def test_tool_addition_block_is_accepted_and_ignored():
    """Claude Code 2.1.x announces deferred tools with ``tool_addition``
    blocks. They are client-side metadata, so the request must validate and
    conversion keeps the accompanying text while dropping the block."""
    req = ClaudeMessagesRequest(
        model="deepseek-flash",
        max_tokens=128,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "The following tools just became available and are ready to use:\nSubagentHandback",
                    },
                    {
                        "type": "tool_addition",
                        "tool": {
                            "type": "tool_reference",
                            "name": "SubagentHandback",
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            )
        ],
    )

    converted = convert_claude_to_openai(req, model_manager)
    assert converted["messages"] == [
        {
            "role": "user",
            "content": "The following tools just became available and are ready to use:\nSubagentHandback",
        }
    ]
