"""Cancellation semantics of the upstream client."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.core.client import OpenAIClient


class _Chunk:
    def model_dump(self):
        return {"choices": [{"delta": {"content": "hi"}}]}


class _FakeStreamCompletions:
    async def create(self, **kwargs):
        async def stream():
            yield _Chunk()
            yield _Chunk()

        return stream()


class _HangingCompletions:
    def __init__(self):
        self.started = asyncio.Event()

    async def create(self, **kwargs):
        self.started.set()
        await asyncio.Event().wait()


def make_client():
    return OpenAIClient(api_key="sk-test", base_url="http://upstream.example/v1")


async def test_streaming_cancellation_499_is_not_rewrapped_as_500():
    client = make_client()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeStreamCompletions())
    )
    request_id = "cancel-stream"
    stream = client.create_chat_completion_stream(
        {"model": "deepseek-flash", "messages": []}, request_id
    )
    # Consume one chunk so the generator has registered its cancel event,
    # then cancel before the next chunk is emitted.
    await stream.__anext__()
    client.cancel_request(request_id)

    with pytest.raises(HTTPException) as exc_info:
        await stream.__anext__()

    assert exc_info.value.status_code == 499


async def test_non_streaming_cancellation_499_is_not_rewrapped_as_500():
    client = make_client()
    hanging = _HangingCompletions()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=hanging))
    request_id = "cancel-plain"
    task = asyncio.create_task(
        client.create_chat_completion(
            {"model": "deepseek-flash", "messages": []}, request_id
        )
    )
    await hanging.started.wait()
    client.cancel_request(request_id)

    with pytest.raises(HTTPException) as exc_info:
        await task

    assert exc_info.value.status_code == 499
