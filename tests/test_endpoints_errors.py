"""Non-streaming upstream failures must use the Anthropic error envelope."""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import httpx
import pytest
from fastapi import HTTPException

import src.api.endpoints as endpoints
from src.core.config import config
from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def auth_headers():
    if config.anthropic_api_key:
        return {"x-api-key": config.anthropic_api_key}
    return {}


async def post_message(client):
    return await client.post(
        "/v1/messages",
        json={
            "model": "deepseek-flash",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=auth_headers(),
    )


@pytest.mark.parametrize(
    "status_code,error_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (529, "overloaded_error"),
    ],
)
async def test_non_streaming_upstream_error_uses_anthropic_envelope(
    client, monkeypatch, status_code, error_type
):
    async def fail(*args, **kwargs):
        raise HTTPException(status_code=status_code, detail=f"upstream {status_code}")

    monkeypatch.setattr(endpoints.openai_client, "create_chat_completion", fail)

    response = await post_message(client)

    assert response.status_code == status_code
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == error_type
    assert f"upstream {status_code}" in payload["error"]["message"]


async def test_unexpected_non_streaming_error_uses_anthropic_envelope(
    client, monkeypatch
):
    async def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(endpoints.openai_client, "create_chat_completion", fail)

    response = await post_message(client)

    assert response.status_code == 500
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "api_error"
    assert "boom" in payload["error"]["message"]
