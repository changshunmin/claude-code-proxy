import json
import logging
from src.core.config import config

# Parse log level - extract just the first word to handle comments
log_level = config.log_level.split()[0].upper()

# Validate and set default if invalid
valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
if log_level not in valid_levels:
    log_level = 'INFO'

# Logging Configuration
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def claude_request_params(request) -> str:
    """Compact client request parameters for error log entries."""
    return json.dumps(
        {
            "model": getattr(request, "model", None),
            "stream": getattr(request, "stream", None),
            "max_tokens": getattr(request, "max_tokens", None),
            "temperature": getattr(request, "temperature", None),
            "message_count": len(getattr(request, "messages", None) or []),
            "tools": [
                tool.name for tool in (getattr(request, "tools", None) or [])
            ],
            "tool_choice": getattr(request, "tool_choice", None),
        },
        ensure_ascii=False,
    )


# Configure uvicorn to be quieter
for uvicorn_logger in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(uvicorn_logger).setLevel(logging.WARNING)