from pydantic import BaseModel, Field
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


class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
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
            ]
        ],
    ]


class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ClaudeThinkingConfig(BaseModel):
    type: Optional[str] = None
    budget_tokens: Optional[int] = None
    display: Optional[str] = None


class ClaudeOutputConfig(BaseModel):
    effort: Optional[str] = None
    format: Optional[Dict[str, Any]] = None


class ClaudeMessagesRequest(BaseModel):
    model: str
    max_tokens: int
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


class ClaudeTokenCountRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    output_config: Optional[ClaudeOutputConfig] = None
