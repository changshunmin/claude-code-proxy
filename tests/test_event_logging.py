"""Tests for the Claude Code telemetry endpoint stub.

Claude Code posts first-party telemetry batches to
``https://api.anthropic.com/api/event_logging/v2/batch`` and treats a 2xx
response carrying ``{"accepted_count": N}`` as success; non-2xx triggers
retries with backoff. When api.anthropic.com points at this proxy we must
answer the same way so the client does not retry/drop events.
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


async def test_event_logging_batch_accepts_events(client):
    response = await client.post(
        "/api/event_logging/v2/batch",
        json={"events": [{"event_type": "a"}, {"event_type": "b"}]},
    )
    assert response.status_code == 200
    assert response.json() == {"accepted_count": 2}


async def test_event_logging_batch_tolerates_malformed_body(client):
    response = await client.post(
        "/api/event_logging/v2/batch",
        content=b"not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json()["accepted_count"] == 0
