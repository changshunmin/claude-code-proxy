"""Parity tests for the official DeepSeek Anthropic surface.

The relay is mounted twice: at the legacy root and under ``/anthropic``,
which is the base path of ``https://api.deepseek.com/anthropic``. These
tests pin the model catalog and the prefixed routes down.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import httpx
import pytest

from src.core.config import config
from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def test_anthropic_prefix_serves_count_tokens(client):
    response = await client.post(
        "/anthropic/v1/messages/count_tokens",
        json={
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "Say hi"}],
        },
        headers={"x-api-key": config.anthropic_api_key or "test"},
    )
    assert response.status_code == 200
    assert response.json()["input_tokens"] >= 1


async def test_anthropic_prefix_rejects_unknown_model(client):
    response = await client.post(
        "/anthropic/v1/messages",
        json={
            "model": "nope",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"x-api-key": config.anthropic_api_key or "test"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert "deepseek-flash" in body["error"]["message"]


async def test_models_catalog_reports_local_window(client):
    response = await client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    model = payload["data"][0]
    assert model["id"] == "deepseek-flash"
    assert model["context_window"] == config.model_max_context
    assert model["max_output_tokens"] == config.model_max_output
    assert model["effort"]["supported_levels"] == ["low", "high", "max"]
    assert model["effort"]["default_level"] == "high"


async def test_models_catalog_is_also_available_under_anthropic(client):
    response = await client.get("/anthropic/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "deepseek-flash"
