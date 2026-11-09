import re

from src.core.config import config


class UnknownModelError(ValueError):
    """Raised when the client asks for a model this relay cannot serve."""


# Claude Code context tags such as `deepseek-flash[1m]` are client-side
# metadata; the upstream model id never carries them.
_CONTEXT_TAG_RE = re.compile(r"\s*\[[^\]]*\]\s*$")


class ModelManager:
    def __init__(self, config):
        self.config = config

    def normalize_model_name(self, claude_model: str) -> str:
        """Strip Claude Code context tags from a model id."""
        return _CONTEXT_TAG_RE.sub("", (claude_model or "").strip()).strip()

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Anthropic/Claude Code model ids onto the configured upstream model.

        The official DeepSeek Anthropic endpoint accepts ``deepseek-flash`` and
        ``deepseek-v4-pro`` plus every ``claude*`` alias (opus -> v4-pro,
        everything else -> flash). This relay serves one local checkpoint, so
        all of those ids collapse onto the configured BIG/MIDDLE/SMALL tiers.
        """
        name = self.normalize_model_name(claude_model)
        lower = name.lower()

        # OpenAI-compatible ids belong to the legacy pass-through surface.
        if lower.startswith(("gpt-", "o1-", "ep-", "doubao-")):
            return name

        # Official DeepSeek model ids.
        if lower == "deepseek-flash":
            return self.config.small_model
        if lower == "deepseek-v4-pro":
            return self.config.big_model

        # Aliases that Claude Code / DeepSeek docs use (deepseek-v4-flash,
        # deepseek-chat, the local checkpoint id, ...).
        if lower.startswith("deepseek-"):
            if "flash" in lower or "haiku" in lower or "mini" in lower:
                return self.config.small_model
            if "sonnet" in lower:
                return self.config.middle_model
            return self.config.big_model

        # Official Claude mapping: opus -> v4-pro, other claude* -> flash.
        if lower.startswith("claude"):
            if "opus" in lower:
                return self.config.big_model
            if "sonnet" in lower:
                return self.config.middle_model
            if "haiku" in lower:
                return self.config.small_model
            return self.config.small_model

        raise UnknownModelError(
            "The supported model names are deepseek-flash, deepseek-v4-pro "
            f"(claude-* names are accepted), but you passed {name}."
        )


model_manager = ModelManager(config)
