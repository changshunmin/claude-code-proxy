"""Unit tests for effort mapping (official DeepSeek supports low/high/max only)."""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from src.core.config import Config


def _config(**overrides):
    c = Config()
    c.reasoning_effort = "auto"
    c.default_effort = "high"
    for key, value in overrides.items():
        setattr(c, key, value)
    return c


def test_client_xhigh_is_mapped_to_max():
    assert _config().resolve_effort("xhigh") == "max"


def test_env_override_xhigh_is_mapped_to_max():
    assert _config(reasoning_effort="xhigh").resolve_effort(None) == "max"


def test_default_xhigh_is_mapped_to_max():
    assert _config(default_effort="xhigh").resolve_effort("bogus") == "max"


def test_supported_levels_pass_through():
    c = _config()
    assert c.resolve_effort("low") == "low"
    assert c.resolve_effort("high") == "high"
    assert c.resolve_effort("max") == "max"


def test_medium_minimal_none_and_unknown():
    c = _config()
    assert c.resolve_effort("medium") == "high"
    assert c.resolve_effort("minimal") == "low"
    assert c.resolve_effort("none") == "none"
    assert c.resolve_effort("bogus") == "high"
