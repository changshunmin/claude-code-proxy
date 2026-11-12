from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import uuid
from typing import Optional

from src.core.config import config
from src.core.logging import claude_request_params, logger
from src.core.client import OpenAIClient
from src.core.errors import anthropic_error_body, prompt_too_long_message
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.conversion.request_converter import (
    convert_claude_to_openai,
    estimate_claude_token_count,
)
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.model_manager import model_manager, UnknownModelError

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)

async def validate_api_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    client_api_key = None
    
    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")
    
    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return
        
    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key."
        )

@router.post("/v1/messages")
async def create_message(request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)):
    try:
        logger.debug(
            f"Processing Claude request: model={request.model}, stream={request.stream}"
        )

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Convert Claude request to OpenAI format
        try:
            openai_request = convert_claude_to_openai(request, model_manager)
        except UnknownModelError as e:
            logger.warning(f"Rejected unsupported model: {request.model}")
            return JSONResponse(
                status_code=400, content=anthropic_error_body(str(e))
            )

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        openai_client,
                        request_id,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(
                    f"Streaming error: {e.detail} | "
                    f"request_params={claude_request_params(request)}"
                )
                import traceback

                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await openai_client.create_chat_completion(
                openai_request, request_id
            )
            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )
            return claude_response
    except HTTPException as e:
        overflow_message = prompt_too_long_message(e.detail)
        if overflow_message:
            logger.warning(f"Upstream context overflow: {overflow_message}")
            return JSONResponse(
                status_code=400, content=anthropic_error_body(overflow_message)
            )
        raise
    except Exception as e:
        import traceback

        logger.error(
            f"Unexpected error processing request: {e} | "
            f"request_params={claude_request_params(request)}"
        )
        logger.error(traceback.format_exc())
        overflow_message = prompt_too_long_message(e)
        if overflow_message:
            return JSONResponse(
                status_code=400, content=anthropic_error_body(overflow_message)
            )
        error_message = openai_client.classify_openai_error(str(e))
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    try:
        estimated_tokens = estimate_claude_token_count(
            request.system, request.messages, request.tools
        )
        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(
            f"Error counting tokens: {e} | "
            f"request_params={claude_request_params(request)}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """DeepSeek-compatible model catalog.

    Mirrors ``GET https://api.deepseek.com/models`` but reports the real
    window of the checkpoint this relay serves (262144 context / 128000
    output by default) instead of the hosted 1M-context values.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-flash",
                "object": "model",
                "owned_by": "deepseek",
                "name": config.big_model,
                "context_window": config.model_max_context,
                "max_output_tokens": config.model_max_output,
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "effort": {
                    "supported_levels": ["low", "high", "max"],
                    "default_level": "high",
                },
                "api_capabilities": {
                    "anthropic_messages": {
                        "system_prompt_update": "in-history",
                        "tool_update": "addition-only",
                    }
                },
            }
        ],
    }


@router.api_route("/api/hello", methods=["GET", "HEAD"])
async def hello_stub():
    """Connectivity stub for Claude Code's /api/hello preconnect probe."""
    return {"status": "ok"}


@router.post("/api/event_logging/v2/batch")
async def event_logging_batch(request: Request):
    """Accept Claude Code's first-party telemetry batches.

    CC posts ``{"events": [...]}`` here and treats a 2xx response carrying
    ``{"accepted_count": N}`` as success; non-2xx responses trigger retries
    with backoff. Answering like the real API keeps the telemetry path
    healthy when api.anthropic.com resolves to this proxy.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = None
    events = payload.get("events") if isinstance(payload, dict) else None
    count = len(events) if isinstance(events, list) else 0
    logger.debug(f"Accepted event logging batch: {count} events")
    return {"accepted_count": count}


@router.api_route("/api/health", methods=["GET", "HEAD"])
async def api_health_stub():
    """Health probe at the path Claude Code actually calls."""
    return {"status": "ok"}


@router.post("/api/eval/{client_key}")
@router.post("/api/eval-authed/{client_key}")
async def growthbook_eval_stub(client_key: str):
    """GrowthBook remote-eval endpoint used for feature flags.

    Returning an empty feature set makes Claude Code fall back to its
    built-in defaults, as it would for a workspace without overrides.
    """
    return {
        "features": {},
        "experiments": [],
        "savedGroups": {},
        "date": datetime.now().isoformat(),
    }


@router.get("/api/directory/plugins")
async def plugin_directory_stub():
    """Plugin marketplace catalog: an empty first page."""
    return {"data": []}


@router.get("/mcp-registry/v0/servers")
@router.get("/api/directory/servers")
async def mcp_registry_stub():
    """MCP server directory: an empty list with no continuation cursor."""
    return {"servers": [], "metadata": {}}


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }
