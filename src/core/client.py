import asyncio
import json
import re
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError
from src.core.logging import logger

_DATA_URL_RE = re.compile(r"data:[^;\"\\]+;base64,[A-Za-z0-9+/=]+")
_MAX_LOGGED_REQUEST_CHARS = 4000


def _bounded_request_params(request: Dict[str, Any]) -> str:
    """Serialize a request for logging without base64 blobs or unbounded size."""
    try:
        params = json.dumps(request, ensure_ascii=False)
    except (TypeError, ValueError):
        params = repr(request)
    params = _DATA_URL_RE.sub(
        lambda match: f"data:...;base64,<{len(match.group(0))} chars omitted>", params
    )
    if len(params) > _MAX_LOGGED_REQUEST_CHARS:
        params = (
            params[:_MAX_LOGGED_REQUEST_CHARS]
            + f"... <truncated, {len(params)} chars total>"
        )
    return params


class OpenAIClient:
    """Async OpenAI client with cancellation support."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 90, api_version: Optional[str] = None, custom_headers: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        
        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-proxy/1.0.0"
        }
        
        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}
        
        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=all_headers
            )
        self.active_requests: Dict[str, asyncio.Event] = {}

    def _log_upstream_error(
        self,
        error: Exception,
        request: Dict[str, Any],
        request_id: Optional[str],
    ) -> None:
        """Log the exception chain plus a bounded, redacted request summary.

        ``str(APIConnectionError)`` is only "Connection error.", so without
        the cause the log gives no hint about what actually failed
        (ConnectError, ReadError, RemoteProtocolError, ...).  The request
        parameters are kept because an upstream 400 can only be traced back
        to the payload that triggered it — but base64 image payloads are
        redacted and the serialized body is truncated so a single failure
        cannot write megabytes of conversation content into the log.
        """
        cause = error.__cause__
        params = _bounded_request_params(request)
        logger.error(
            "Upstream error: %s: %s (cause: %s: %s) | request_id=%s | request_params=%s",
            type(error).__name__,
            error,
            type(cause).__name__ if cause is not None else "none",
            cause if cause is not None else "",
            request_id or "none",
            params,
        )

    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            # Create task that can be cancelled
            completion_task = asyncio.create_task(
                self.client.chat.completions.create(**request)
            )
            
            if request_id:
                # Wait for either completion or cancellation
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if request was cancelled
                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                completion = await completion_task
            else:
                completion = await completion_task
            
            # Convert to dict format that matches the original interface
            return completion.model_dump()
        
        except AuthenticationError as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            self._log_upstream_error(e, request, request_id)
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except HTTPException:
            # Cancellation (499) and other internally raised HTTP errors must
            # pass through unchanged, not be re-wrapped as a 500.
            raise
        except Exception as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            # Ensure stream is enabled
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True
            
            # Create the streaming completion
            streaming_completion = await self.client.chat.completions.create(**request)
            
            async for chunk in streaming_completion:
                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                # Convert chunk to SSE format matching original HTTP client format
                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"
            
            # Signal end of stream
            yield "data: [DONE]"
                
        except AuthenticationError as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            self._log_upstream_error(e, request, request_id)
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except HTTPException:
            raise
        except Exception as e:
            self._log_upstream_error(e, request, request_id)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False