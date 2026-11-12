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
                    # Image payloads are base64, and the upstream bills them by
                    # pixels rather than characters; counting the data URL as
                    # text would collapse max_tokens to the minimum.
                    if block.get("type") in ("image_url", "image"):
                        continue
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
    requested = claude_request.max_tokens
    if requested is None:
        # The official endpoint makes max_tokens optional and picks a
        # model-dependent default; ours starts at the hard output cap and is
        # then clamped to the remaining window below.
        requested = config.model_max_output
    requested = max(requested, config.min_tokens_limit)
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


def extract_text_content(content) -> str:
    """Flatten a Claude message `content` (string or block list) to plain text,
    skipping non-text blocks such as images."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
            parts.append(block.get("text", ""))
    return "\n\n".join(parts)


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []

    # Collect system prompt text.  Besides the top-level `system` field, Claude
    # Code 2.1.x sends mid-conversation system messages (beta) with
    # role="system" inside `messages`; folding them into the leading system
    # message mirrors the client's own fallback shape and keeps the upstream
    # conversation starting with a single system turn.
    system_parts = []
    if claude_request.system:
        system_parts.append(extract_text_content(claude_request.system))
    for msg in claude_request.messages:
        if msg.role == Constants.ROLE_SYSTEM:
            system_parts.append(extract_text_content(msg.content))

    system_text = "\n\n".join(part.strip() for part in system_parts if part.strip())
    if system_text:
        openai_messages.append(
            {"role": Constants.ROLE_SYSTEM, "content": system_text}
        )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_SYSTEM:
            # Already merged into the leading system message above.
            pass
        elif msg.role == Constants.ROLE_USER:
            # Tool results must follow their assistant tool_use turn upstream.
            # Convert them wherever they appear instead of only when adjacent:
            # CC can insert a role="system" metadata message in between.
            has_tool_result = isinstance(msg.content, list) and any(
                block.type == Constants.CONTENT_TOOL_RESULT for block in msg.content
            )
            if has_tool_result:
                openai_messages.extend(convert_claude_tool_results(msg))
            else:
                openai_message = convert_claude_user_message(msg)
                # Metadata-only blocks (e.g. tool_addition) leave nothing to send;
                # an empty content array is rejected by OpenAI-compatible upstreams.
                if openai_message["content"] != []:
                    openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            # An assistant turn with neither text nor tool calls is invalid
            # upstream; it can appear when every block was metadata-only.
            if openai_message.get("content") is not None or openai_message.get(
                "tool_calls"
            ):
                openai_messages.append(openai_message)

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
            # Server tools (web_search_20250305, ...) are executed by the
            # official DeepSeek API, not by the client, so there is nothing to
            # forward to an OpenAI-compatible upstream. Ignore them instead of
            # failing the whole request.
            if tool.type and tool.type != "custom":
                logger.debug(f"Skipping server-side tool: {tool.name} ({tool.type})")
                continue
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

    # Convert tool choice. Upstreams reject tool_choice unless tools is set,
    # so only forward it when client tools actually made it into the request.
    if claude_request.tool_choice and "tools" in openai_request:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            # The official DeepSeek Anthropic endpoint treats "any" as "auto"
            # (it does not force a tool call); mirror that behaviour.
            openai_request["tool_choice"] = "auto"
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
            image_block = image_block_to_openai(block)
            if image_block:
                openai_content.append(image_block)

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

    The official endpoint allows text blocks in the same user message after
    the tool_result blocks; keep them as a follow-up user message instead of
    dropping them.
    """
    tool_messages = []
    images = []
    trailing_text = []

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
            elif block.type == Constants.CONTENT_TEXT and block.text:
                trailing_text.append(block.text)
            elif block.type == Constants.CONTENT_IMAGE:
                image_blocks = image_block_to_openai(block)
                if image_blocks:
                    images.append(image_blocks)

    if trailing_text or images:
        if trailing_text and not images:
            user_content: Any = "\n".join(trailing_text)
        else:
            user_content = []
            if trailing_text:
                user_content.append({"type": "text", "text": "\n".join(trailing_text)})
            user_content.extend(images)
        tool_messages.append({"role": Constants.ROLE_USER, "content": user_content})

    return tool_messages


def image_block_to_openai(block) -> Optional[Dict[str, Any]]:
    """Convert a Claude image block (base64 or url) to an OpenAI image block."""
    source = getattr(block, "source", None)
    if not isinstance(source, dict):
        return None
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    if source.get("type") == "base64" and source.get("data"):
        media_type = source.get("media_type") or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source['data']}"},
        }
    return None


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
