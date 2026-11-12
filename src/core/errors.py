import re
from typing import Optional, Tuple

_OVERFLOW_HINTS = (
    "maximum context length",
    "prompt contains at least",
    "prompt is too long",
    "input is too long for requested model",
    "in the messages,",
    "exceeds this limit",
)


def _extract_counts(text: str) -> Tuple[Optional[int], Optional[int]]:
    actual = None
    for pattern in (
        r"prompt contains at least (\d+)\s*(?:input )?tokens",
        r"\((\d+)\s+in the messages,\s*\d+\s+in the completion\)",
        r"\(parameter=input_tokens, value=(\d+)\)",
        r"prompt is too long:?\s*(\d+)\s*tokens?",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            actual = int(match.group(1))
            break

    limit = None
    for pattern in (
        r"maximum context length is (\d+)\s*tokens",
        r"prompt is too long:?\s*\d+\s*tokens?\s*>\s*(\d+)",
        r"context length is (\d+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            limit = int(match.group(1))
            break

    return actual, limit


def prompt_too_long_message(error_detail) -> Optional[str]:
    """Map an upstream context-overflow error to the message Claude Code
    recognizes for reactive auto-compaction (`prompt is too long: N tokens > M maximum`)."""
    text = str(error_detail)
    lowered = text.lower()
    if not any(hint in lowered for hint in _OVERFLOW_HINTS):
        return None

    actual, limit = _extract_counts(text)
    if actual is not None and limit is not None:
        return f"prompt is too long: {actual} tokens > {limit} maximum"
    return "prompt is too long"


def anthropic_error_body(message: str, error_type: str = "invalid_request_error") -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


_HTTP_STATUS_TO_ANTHROPIC_ERROR = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def anthropic_error_type(status_code: int) -> str:
    """Map an upstream HTTP status to an Anthropic error type."""
    return _HTTP_STATUS_TO_ANTHROPIC_ERROR.get(status_code, "api_error")
