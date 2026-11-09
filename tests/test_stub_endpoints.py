"""Tests for the Claude Code companion endpoint stubs.

When api.anthropic.com resolves to this proxy, Claude Code also calls the
feature-flag, health and marketplace discovery endpoints. Serving their
shapes keeps CC behaving exactly as it does against the official API.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import httpx
import pytest

from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def test_api_health_get_and_head(client):
    get_response = await client.get("/api/health")
    head_response = await client.head("/api/health")
    assert get_response.status_code == 200
    assert get_response.json() == {"status": "ok"}
    assert head_response.status_code == 200


async def test_growthbook_remote_eval(client):
    for path in ("/api/eval/sdk-testkey", "/api/eval-authed/sdk-testkey"):
        response = await client.post(path, json={"attributes": {}})
        assert response.status_code == 200, path
        payload = response.json()
        assert payload["features"] == {}
        assert payload["experiments"] == []


async def test_plugin_directory_returns_empty_page(client):
    response = await client.get(
        "/api/directory/plugins",
        params={"limit": "500", "visibility": "public", "include": "description"},
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


async def test_mcp_registry_returns_empty_list(client):
    for path in ("/mcp-registry/v0/servers", "/api/directory/servers"):
        response = await client.get(
            path, params={"version": "latest", "limit": "100", "visibility": "public"}
        )
        assert response.status_code == 200, path
        payload = response.json()
        assert payload["servers"] == []
        assert payload["metadata"] == {}
