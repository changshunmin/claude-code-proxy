import json
import uuid
from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.core.errors import anthropic_error_body, anthropic_error_type, prompt_too_long_message
from src.core.logging import claude_request_params
from src.conversion.request_converter import estimate_claude_token_count
from src.models.claude import ClaudeMessagesRequest


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})
    response_id = openai_response.get("id", f"msg_{uuid.uuid4()}")

    # Build Claude content blocks
    content_blocks = []

    # Add reasoning first: Claude expects thinking ahead of text/tool_use.
    # Thinking blocks need a signature; the official DeepSeek endpoint uses
    # the message id, and our converter ignores it when replaying history.
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        content_blocks.append(
            {
                "type": Constants.CONTENT_THINKING,
                "thinking": reasoning,
                "signature": response_id,
            }
        )

    # Add text content
    text_content = message.get("content")
    if text_content is not None and text_content != "":
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Usage follows Anthropic semantics: input_tokens excludes cache reads,
    # cache_read_input_tokens is the cached portion. service_tier mirrors the
    # official endpoint's response.
    usage = openai_response.get("usage", {}) or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens", 0) or 0
    prompt_tokens = usage.get("prompt_tokens", 0) or 0

    # Build Claude response
    claude_response = {
        "id": response_id,
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _claude_usage(
            max(prompt_tokens - cached_tokens, 0),
            cached_tokens,
            usage.get("completion_tokens") or 0,
        ),
    }

    if original_request.safeguards:
        claude_response["safeguard_results"] = _unsupported_safeguard_results(
            original_request
        )

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""
    async for chunk in convert_openai_streaming_to_claude_with_cancellation(
        openai_stream, original_request, logger, None, None, None
    ):
        yield chunk


def _claude_usage(
    input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    """Build the 5-key Anthropic usage object shared by every response path."""
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read_input_tokens,
        "output_tokens": output_tokens,
        "service_tier": "standard",
    }


def _map_stop_reason(finish_reason):
    if finish_reason == "length":
        return Constants.STOP_MAX_TOKENS
    if finish_reason in ("tool_calls", "function_call"):
        return Constants.STOP_TOOL_USE
    return Constants.STOP_END_TURN


def _unsupported_safeguard_results(original_request: ClaudeMessagesRequest):
    """Answer auto mode's server-side classifier request with "unsupported".

    The relay forwards to an OpenAI-protocol upstream, which cannot run
    Anthropic's server-side check. Saying so explicitly makes Claude Code
    fall back to its own classifier for the session instead of assuming a
    proxy dropped the result. One verdict per type: more than one
    ``dangerous_tool_use`` entry is treated as unrecognized.
    """
    types = []
    for item in original_request.safeguards or []:
        result_type = item.get("type") if isinstance(item, dict) else None
        if result_type and result_type not in types:
            types.append(result_type)
    return [
        {"type": result_type, "status": {"type": "unsupported"}}
        for result_type in types
    ]


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support.

    Reasoning arrives as `delta.reasoning` (DeepSeek / vLLM style) and is emitted as a
    Claude thinking block ahead of the text and tool_use blocks.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    # The official endpoint reports the prompt size in message_start; mirror
    # that with our tokenizer estimate so clients see a non-zero context
    # immediately instead of waiting for message_delta.
    estimated_input_tokens = estimate_claude_token_count(
        original_request.system, original_request.messages, original_request.tools
    )

    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield sse(
        Constants.EVENT_MESSAGE_START,
        {
            "type": Constants.EVENT_MESSAGE_START,
            "message": {
                "id": message_id,
                "type": "message",
                "role": Constants.ROLE_ASSISTANT,
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _claude_usage(estimated_input_tokens),
            },
        },
    )

    # Block layout: [thinking?] [text] [tool_use...]. Indices are handed out lazily
    # because whether reasoning shows up is only known once the first chunk lands.
    next_index = 0
    thinking_index = None
    thinking_open = False
    text_index = None
    text_open = False
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = _claude_usage(estimated_input_tokens)

    def start_thinking():
        nonlocal next_index, thinking_index
        if thinking_index is None:
            thinking_index = next_index
            next_index += 1
        return thinking_index

    def start_text():
        nonlocal next_index, text_index
        if text_index is None:
            text_index = next_index
            next_index += 1
        return text_index

    def close_thinking_events():
        nonlocal thinking_index, thinking_open
        if thinking_index is not None and thinking_open:
            thinking_open = False
            index = thinking_index
            thinking_index = None
            return [
                sse(
                    Constants.EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                        "index": index,
                        "delta": {
                            "type": Constants.DELTA_SIGNATURE,
                            "signature": message_id,
                        },
                    },
                ),
                sse(
                    Constants.EVENT_CONTENT_BLOCK_STOP,
                    {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": index},
                ),
            ]
        return []

    try:
        async for line in openai_stream:
            if http_request is not None and await http_request.is_disconnected():
                logger.info(f"Client disconnected, cancelling request {request_id}")
                if openai_client is not None:
                    openai_client.cancel_request(request_id)
                break

            if not line.strip():
                continue

            if not line.startswith("data: "):
                continue

            chunk_data = line[6:]
            if chunk_data.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(chunk_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chunk: {chunk_data}, error: {e}")
                continue

            usage = chunk.get("usage")
            if usage:
                prompt_details = usage.get("prompt_tokens_details") or {}
                # Explicit nulls must not blow up the subtraction mid-stream.
                prompt_tokens = usage.get("prompt_tokens") or 0
                cached_tokens = prompt_details.get("cached_tokens") or 0
                # Anthropic 语义：input_tokens 不含缓存读取，而 prompt_tokens 已含
                usage_data = _claude_usage(
                    max(prompt_tokens - cached_tokens, 0),
                    cached_tokens,
                    usage.get("completion_tokens") or 0,
                )

            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_delta:
                if text_open:
                    # Keep a single block open at a time: reasoning arriving
                    # after text needs the text block closed first.
                    text_open = False
                    yield sse(
                        Constants.EVENT_CONTENT_BLOCK_STOP,
                        {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": text_index},
                    )
                    text_index = None
                index = start_thinking()
                if not thinking_open:
                    thinking_open = True
                    yield sse(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": index,
                            "content_block": {"type": Constants.CONTENT_THINKING, "thinking": ""},
                        },
                    )
                yield sse(
                    Constants.EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                        "index": index,
                        "delta": {"type": Constants.DELTA_THINKING, "thinking": reasoning_delta},
                    },
                )

            content_delta = delta.get("content")
            if content_delta:
                for event in close_thinking_events():
                    yield event
                index = start_text()
                if not text_open:
                    text_open = True
                    yield sse(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": index,
                            "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
                        },
                    )
                yield sse(
                    Constants.EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                        "index": index,
                        "delta": {"type": Constants.DELTA_TEXT, "text": content_delta},
                    },
                )

            for tc_delta in delta.get("tool_calls") or []:
                tc_index = tc_delta.get("index", 0)
                tool_call = current_tool_calls.setdefault(
                    tc_index,
                    {"id": None, "name": None, "args": "", "index": None},
                )
                if tc_delta.get("id"):
                    tool_call["id"] = tc_delta["id"]

                function_data = tc_delta.get(Constants.TOOL_FUNCTION) or {}
                if function_data.get("name"):
                    tool_call["name"] = function_data["name"]

                if tool_call["index"] is None and tool_call["id"] and tool_call["name"]:
                    for event in close_thinking_events():
                        yield event
                    if text_open:
                        text_open = False
                        yield sse(
                            Constants.EVENT_CONTENT_BLOCK_STOP,
                            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": text_index},
                        )
                        # Release the index: a later text delta needs a fresh
                        # block instead of restarting an already-closed one.
                        text_index = None
                    tool_call["index"] = next_index
                    next_index += 1
                    yield sse(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": tool_call["index"],
                            "content_block": {
                                "type": Constants.CONTENT_TOOL_USE,
                                "id": tool_call["id"],
                                "name": tool_call["name"],
                                "input": {},
                            },
                        },
                    )

                arguments = function_data.get("arguments")
                if arguments and tool_call["index"] is not None:
                    tool_call["args"] += arguments
                    yield sse(
                        Constants.EVENT_CONTENT_BLOCK_DELTA,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                            "index": tool_call["index"],
                            "delta": {"type": Constants.DELTA_INPUT_JSON, "partial_json": arguments},
                        },
                    )

            if finish_reason:
                final_stop_reason = _map_stop_reason(finish_reason)

    except HTTPException as e:
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            yield sse(
                "error",
                {"type": "error", "error": {"type": "cancelled", "message": "Request was cancelled by client"}},
            )
            return
        overflow_message = prompt_too_long_message(e.detail)
        if overflow_message:
            logger.warning(f"Upstream context overflow: {overflow_message}")
            yield sse("error", anthropic_error_body(overflow_message))
            return
        # The response has already started, so an HTTP status can no longer be
        # sent. Report the failure inside the stream instead of raising, which
        # would make Starlette log "Caught handled exception, but response
        # already started." and leave the client with a silent truncation.
        logger.error(
            f"Upstream streaming error {e.status_code}: {e.detail} | "
            f"request_params={claude_request_params(original_request)}"
        )
        yield sse(
            "error",
            anthropic_error_body(str(e.detail), anthropic_error_type(e.status_code)),
        )
        return
    except Exception as e:
        overflow_message = prompt_too_long_message(e)
        if overflow_message:
            logger.warning(f"Upstream context overflow: {overflow_message}")
            yield sse("error", anthropic_error_body(overflow_message))
            return
        logger.error(
            f"Streaming error: {e} | "
            f"request_params={claude_request_params(original_request)}"
        )
        import traceback

        logger.error(traceback.format_exc())
        yield sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"}},
        )
        return

    # Close whatever is still open; blocks already closed mid-stream are skipped
    for event in close_thinking_events():
        yield event
    if text_index is not None and text_open:
        yield sse(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": text_index},
        )
    for tool_call in current_tool_calls.values():
        if tool_call["index"] is not None:
            yield sse(
                Constants.EVENT_CONTENT_BLOCK_STOP,
                {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": tool_call["index"]},
            )

    if final_stop_reason == Constants.STOP_END_TURN and current_tool_calls:
        final_stop_reason = Constants.STOP_TOOL_USE

    message_delta = {"stop_reason": final_stop_reason, "stop_sequence": None}
    if original_request.safeguards:
        message_delta["safeguard_results"] = _unsupported_safeguard_results(
            original_request
        )
    yield sse(
        Constants.EVENT_MESSAGE_DELTA,
        {
            "type": Constants.EVENT_MESSAGE_DELTA,
            "delta": message_delta,
            "usage": usage_data,
        },
    )
    yield sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})
