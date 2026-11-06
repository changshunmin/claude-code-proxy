import json
import uuid
from fastapi import HTTPException, Request
from src.core.constants import Constants
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

    # Build Claude content blocks
    content_blocks = []

    # Add reasoning first: Claude expects thinking ahead of text/tool_use
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        content_blocks.append({"type": Constants.CONTENT_THINKING, "thinking": reasoning})

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

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""
    async for chunk in convert_openai_streaming_to_claude_with_cancellation(
        openai_stream, original_request, logger, None, None, None
    ):
        yield chunk


def _map_stop_reason(finish_reason):
    if finish_reason == "length":
        return Constants.STOP_MAX_TOKENS
    if finish_reason in ("tool_calls", "function_call"):
        return Constants.STOP_TOOL_USE
    return Constants.STOP_END_TURN


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
                "usage": {"input_tokens": 0, "output_tokens": 0},
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
    usage_data = {"input_tokens": 0, "output_tokens": 0}

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

    def close_thinking_if_open():
        nonlocal thinking_open
        if thinking_index is not None and thinking_open:
            thinking_open = False
            return sse(
                Constants.EVENT_CONTENT_BLOCK_STOP,
                {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": thinking_index},
            )
        return None

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
                usage_data = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_details.get("cached_tokens", 0),
                }

            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_delta:
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
                closed = close_thinking_if_open()
                if closed:
                    yield closed
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
                    closed = close_thinking_if_open()
                    if closed:
                        yield closed
                    if text_open:
                        text_open = False
                        yield sse(
                            Constants.EVENT_CONTENT_BLOCK_STOP,
                            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": text_index},
                        )
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
        raise
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        yield sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"}},
        )
        return

    # Close whatever is still open; blocks already closed mid-stream are skipped
    if thinking_index is not None and thinking_open:
        yield sse(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": thinking_index},
        )
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

    yield sse(
        Constants.EVENT_MESSAGE_DELTA,
        {
            "type": Constants.EVENT_MESSAGE_DELTA,
            "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
            "usage": usage_data,
        },
    )
    yield sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})
