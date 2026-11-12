from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal


class ClaudeContentBlockText(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class ClaudeContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ClaudeContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str = ""


class ClaudeContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: str = ""


class ClaudeContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ClaudeContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], None] = None
    is_error: Optional[bool] = None


class ClaudeContentBlockToolAddition(BaseModel):
    # Claude Code 2.1.x announces deferred tools mid-conversation with this
    # metadata block; it carries no model-facing content of its own.
    type: Literal["tool_addition"]
    tool: Optional[Dict[str, Any]] = None
    cache_control: Optional[Dict[str, Any]] = None


_KNOWN_CONTENT_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "thinking",
        "redacted_thinking",
        "tool_use",
        "tool_result",
        "tool_addition",
    }
)


class ClaudeContentBlockUnknown(BaseModel):
    """Catch-all for metadata blocks this relay does not model yet.

    Claude Code keeps introducing client-side block types (e.g. tool_removal).
    Accepting them here keeps a single unknown block from 400-ing the whole
    request; the converter drops them because they carry no model-facing
    content. Known types are rejected so a malformed text/tool_use block
    still fails validation instead of silently passing as "unknown".
    """

    model_config = ConfigDict(extra="allow")

    type: str

    @field_validator("type")
    @classmethod
    def _reject_known_types(cls, value: str) -> str:
        if value in _KNOWN_CONTENT_BLOCK_TYPES:
            raise ValueError(f"malformed known content block: {value}")
        return value


class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class ClaudeMessage(BaseModel):
    # Claude Code 2.1.x probes the beta mid-conversation-system shape and sends
    # SessionStart hook context as role="system" inside `messages`.
    role: Literal["user", "assistant", "system"]
    content: Union[
        str,
        List[
            Union[
                ClaudeContentBlockText,
                ClaudeContentBlockImage,
                ClaudeContentBlockThinking,
                ClaudeContentBlockRedactedThinking,
                ClaudeContentBlockToolUse,
                ClaudeContentBlockToolResult,
                ClaudeContentBlockToolAddition,
                ClaudeContentBlockUnknown,
            ]
        ],
    ]


class ClaudeTool(BaseModel):
    # Client tools omit `type`; server tools (web_search_20250305, ...) set it.
    type: Optional[str] = None
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ClaudeThinkingConfig(BaseModel):
    type: Optional[str] = None
    budget_tokens: Optional[int] = None
    display: Optional[str] = None


class ClaudeOutputConfig(BaseModel):
    effort: Optional[str] = None
    format: Optional[Dict[str, Any]] = None


class ClaudeMessagesRequest(BaseModel):
    model: str
    # Optional for raw HTTP requests, like the official DeepSeek endpoint.
    max_tokens: Optional[int] = None
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    output_config: Optional[ClaudeOutputConfig] = None
    context_management: Optional[Dict[str, Any]] = None
    # Auto mode server-side classifier request (paired with the
    # dangerous-tool-use beta header). Kept so the relay can answer with a
    # truthful "unsupported" result; it is never forwarded upstream.
    safeguards: Optional[List[Dict[str, Any]]] = None


class ClaudeTokenCountRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    output_config: Optional[ClaudeOutputConfig] = None
