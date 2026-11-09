"""Tests for model name mapping.

The local relay upstream only serves the model configured in .env
(DeepSeek-V4.1-Flash-1017). Claude Code sessions may carry DeepSeek alias
names (ANTHROPIC_MODEL=deepseek-flash, /model deepseek-v4-pro, ...), which
must be mapped onto the configured BIG/MIDDLE/SMALL models instead of being
forwarded verbatim (the upstream answers 404 for unknown model ids).
"""

import pytest

from src.core.model_manager import ModelManager, UnknownModelError


class FakeConfig:
    big_model = "BIG"
    middle_model = "MIDDLE"
    small_model = "SMALL"


def test_deepseek_aliases_map_to_configured_models():
    m = ModelManager(FakeConfig())
    assert m.map_claude_model_to_openai("deepseek-v4-pro") == "BIG"
    assert m.map_claude_model_to_openai("deepseek-flash") == "SMALL"
    assert m.map_claude_model_to_openai("deepseek-v4-flash") == "SMALL"
    assert m.map_claude_model_to_openai("deepseek-chat") == "BIG"


def test_claude_models_map_by_tier():
    m = ModelManager(FakeConfig())
    assert m.map_claude_model_to_openai("claude-opus-5-5") == "BIG"
    assert m.map_claude_model_to_openai("claude-sonnet-5-5") == "MIDDLE"
    assert m.map_claude_model_to_openai("claude-haiku-5-5") == "SMALL"


def test_openai_and_ark_models_pass_through():
    m = ModelManager(FakeConfig())
    assert m.map_claude_model_to_openai("gpt-4o") == "gpt-4o"
    assert m.map_claude_model_to_openai("ep-20250101") == "ep-20250101"
    assert m.map_claude_model_to_openai("doubao-pro") == "doubao-pro"


def test_context_tags_are_stripped():
    """Claude Code marks 1M-context models with a [1m] suffix."""
    m = ModelManager(FakeConfig())
    assert m.map_claude_model_to_openai("deepseek-flash[1m]") == "SMALL"
    assert m.map_claude_model_to_openai("claude-opus-5-5[1m]") == "BIG"


def test_other_claude_names_follow_the_official_mapping():
    """Official mapping: opus -> v4-pro, every other claude* -> flash."""
    m = ModelManager(FakeConfig())
    assert m.map_claude_model_to_openai("claude-fable-5") == "SMALL"
    assert m.map_claude_model_to_openai("claude-3-opus-20240229") == "BIG"


def test_unknown_non_claude_model_is_rejected():
    m = ModelManager(FakeConfig())
    with pytest.raises(UnknownModelError):
        m.map_claude_model_to_openai("not-a-real-model")
