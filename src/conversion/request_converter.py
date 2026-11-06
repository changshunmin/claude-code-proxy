import json
from typing import Dict, Any, List, Optional
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage
from src.core.config import config
import logging

logger = logging.getLogger(__name__)


def estimate_text_tokens(text: str) -> int:
    """Approximate tokenizer cost: CJK at one token per character (conservative —
    undercounting risks overflowing the upstream window), latin/code at four
    characters per token."""
    cjk = 0
    for ch in text:
        if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯":
            cjk += 1
    return cjk + (len(text) - cjk) // 4


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate used only to keep prompt + output inside the window."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_text_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_text_tokens(json.dumps(block, ensure_ascii=False))
    return total


def _block_field(block, name, default=None):
    """Read a block field from either a pydantic model or a raw dict."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def estimate_claude_token_count(system, messages, tools=None) -> int:
    """CJK-aware estimate for the /v1/messages/count_tokens endpoint.

    The upstream has no tokenizer endpoint, so this stays an estimate — but it
    counts CJK honestly and includes tool schemas, unlike a flat 4-chars/token.
    Images and redacted thinking are skipped: their payloads are base64 and
    would wildly inflate the count.
    """
    total = 0

    def add(text):
        nonlocal total
        if text:
            total += estimate_text_tokens(str(text))

    def add_content(content):
        if content is None:
            return
        if isinstance(content, str):
            add(content)
            return
        if isinstance(content, dict):
            add(json.dumps(content, ensure_ascii=False))
            return
        if not isinstance(content, list):
            return
        for block in content:
            btype = _block_field(block, "type")
            if btype == Constants.CONTENT_TEXT:
                add(_block_field(block, "text"))
            elif btype == Constants.CONTENT_THINKING:
                add(_block_field(block, "thinking"))
            elif btype == Constants.CONTENT_TOOL_USE:
                add(_block_field(block, "name"))
                add(json.dumps(_block_field(block, "input", {}) or {}, ensure_ascii=False))
            elif btype == Constants.CONTENT_TOOL_RESULT:
                inner = _block_field(block, "content")
                if isinstance(inner, list):
                    for item in inner:
                        itype = _block_field(item, "type")
                        if itype == Constants.CONTENT_TEXT:
                            add(_block_field(item, "text"))
                        elif itype in (Constants.CONTENT_IMAGE, "image"):
                            continue
                        elif isinstance(item, dict):
                            add(json.dumps(item, ensure_ascii=False))
                elif isinstance(inner, str):
                    add(inner)
                elif inner is not None:
                    add(json.dumps(inner, ensure_ascii=False))
            elif btype in (Constants.CONTENT_IMAGE, Constants.CONTENT_REDACTED_THINKING):
                continue
            elif isinstance(block, dict):
                add(json.dumps(block, ensure_ascii=False))

    add_content(system)
    for msg in messages:
        add_content(_block_field(msg, "content"))

    for tool in tools or []:
        if isinstance(tool, dict):
            add(json.dumps(tool, ensure_ascii=False))
        else:
            add(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    },
                    ensure_ascii=False,
                )
            )

    return max(1, total)


def resolve_max_tokens(
    claude_request: ClaudeMessagesRequest, openai_messages: List[Dict[str, Any]]
) -> int:
    """Clamp the requested output size to both the output cap and the model window."""
    requested = max(claude_request.max_tokens, config.min_tokens_limit)
    limit = min(requested, config.max_tokens_limit, config.model_max_output)

    remaining = config.model_max_context - estimate_tokens(openai_messages)
    # Leave headroom so the estimate being off does not trip the upstream 400
    safe_remaining = max(config.min_tokens_limit, remaining - 2048)
    return min(limit, safe_remaining)


def apply_thinking_config(
    openai_request: Dict[str, Any], claude_request: ClaudeMessagesRequest
) -> None:
    """Translate Claude's thinking/effort request into the upstream's controls."""
    thinking = claude_request.thinking
    wants_thinking = thinking is not None and (thinking.type or "").lower() not in (
        "disabled",
        "off",
        "none",
    )
    if thinking is not None and thinking.type is None:
        # Legacy shape: {"thinking": {"enabled": true}}
        wants_thinking = True

    # The OpenAI SDK rejects unknown keyword arguments, so vendor-specific
    # fields have to ride in `extra_body` to reach the request JSON.
    extra_body = openai_request.setdefault("extra_body", {})

    if config.thinking_mode == "off" or not wants_thinking:
        extra_body["chat_template_kwargs"] = {"thinking": False}
        return

    effort = config.resolve_effort(
        claude_request.output_config.effort if claude_request.output_config else None
    )
    if effort == "none":
        extra_body["chat_template_kwargs"] = {"thinking": False}
        return
    extra_body["reasoning_effort"] = effort


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages):
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                        if hasattr(block, "type")
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)

        i += 1

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": resolve_max_tokens(claude_request, openai_messages),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }

    apply_thinking_config(openai_request, claude_request)
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            # Claude's "any" requires some tool call; OpenAI spells it "required"
            openai_request["tool_choice"] = "required"
        elif choice_type == "none":
            openai_request["tool_choice"] = "none"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    return openai_request


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})
        elif block.type == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            if (
                isinstance(block.source, dict)
                and block.source.get("type") == "base64"
                and "media_type" in block.source
                and "data" in block.source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []
    reasoning_parts = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block.type == Constants.CONTENT_THINKING:
            # Upstream takes prior reasoning back as a flat `reasoning` field
            if block.thinking:
                reasoning_parts.append(block.thinking)
        elif block.type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    if reasoning_parts:
        openai_message["reasoning"] = "".join(reasoning_parts)
        if openai_message["content"] is None and not tool_calls:
            # An assistant turn that only thought would otherwise look empty
            openai_message["content"] = ""

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format.

    OpenAI tool messages only carry text, so image blocks returned by tools
    (e.g. the Read tool) are re-attached as a follow-up user message with
    image_url blocks. Serializing them into the tool text instead would make
    the upstream tokenize the raw base64 (~1.45 chars/token).
    """
    tool_messages = []
    images = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                text, image_blocks = parse_tool_result_content(block.content)
                if image_blocks and not text:
                    text = "[image attached in the following message]"
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": text,
                    }
                )
                images.extend(image_blocks)

    if images:
        tool_messages.append({"role": Constants.ROLE_USER, "content": images})

    return tool_messages


def parse_tool_result_content(content):
    """Split tool result content into (text, image_url blocks)."""
    if content is None:
        return "No content provided", []

    if isinstance(content, str):
        return content, []

    if isinstance(content, dict):
        content = [content]

    if not isinstance(content, list):
        return str(content), []

    result_parts = []
    images = []
    for item in content:
        if isinstance(item, str):
            result_parts.append(item)
        elif isinstance(item, dict) and item.get("type") == Constants.CONTENT_IMAGE:
            source = item.get("source") or {}
            if source.get("type") == "base64" and source.get("data"):
                media_type = source.get("media_type") or "image/png"
                images.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{source['data']}"
                        },
                    }
                )
        elif isinstance(item, dict):
            if item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif "text" in item:
                result_parts.append(item.get("text", ""))
            else:
                try:
                    result_parts.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    result_parts.append(str(item))
        else:
            result_parts.append(str(item))

    return "\n".join(part for part in result_parts if part).strip(), images
