"""Error logs must stay bounded and must not persist base64 payloads."""

import logging
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import httpx
import pytest

from src.core.client import OpenAIClient
from src.core.config import config
from src.main import app


def auth_headers():
    if config.anthropic_api_key:
        return {"x-api-key": config.anthropic_api_key}
    return {}


def client_with_huge_image():
    payload = "A" * 1_000_000
    return {
        "model": "deepseek-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{payload}"},
                    }
                ],
            }
        ],
    }


def test_upstream_error_log_truncates_and_redacts_request_body(caplog):
    client = OpenAIClient(api_key="sk-test", base_url="http://upstream.example/v1")

    with caplog.at_level(logging.ERROR):
        client._log_upstream_error(
            RuntimeError("boom"), client_with_huge_image(), "req-1"
        )

    text = caplog.text
    assert len(text) < 10_000
    assert "A" * 5_000 not in text
    assert "base64" in text
    assert "deepseek-flash" in text


@pytest.fixture
async def asgi_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def test_validation_log_does_not_embed_full_invalid_input(
    asgi_client, caplog
):
    payload = "A" * 200_000
    body = {
        "model": "deepseek-flash",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": {"x": payload}}],
    }

    with caplog.at_level(logging.ERROR):
        response = await asgi_client.post(
            "/v1/messages", json=body, headers=auth_headers()
        )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    text = caplog.text
    assert len(text) < 20_000
    assert "A" * 100_000 not in text
