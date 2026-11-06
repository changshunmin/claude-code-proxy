import os
import sys
from typing import Optional

# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")
        
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))

        # Total window (input + output) of the upstream model
        self.model_max_context = int(os.environ.get("MODEL_MAX_CONTEXT", "262144"))
        # Hard cap on completion tokens; also clamped by model_max_context - prompt
        self.model_max_output = int(os.environ.get("MODEL_MAX_OUTPUT", str(self.max_tokens_limit)))

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Model settings - BIG and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

        # Thinking / reasoning
        # Upstream accepts: none, low, high, xhigh, max (DeepSeek-V4.1 rejects medium)
        self.thinking_mode = os.environ.get("THINKING_MODE", "auto").strip().lower()
        self.reasoning_effort = os.environ.get("REASONING_EFFORT", "auto").strip().lower()
        self.thinking_display = os.environ.get("THINKING_DISPLAY", "summarized").strip().lower()
        self.default_effort = os.environ.get("DEFAULT_EFFORT", "high").strip().lower()

    def resolve_effort(self, client_effort: Optional[str]) -> str:
        """Map Claude Code's effort values onto what the upstream accepts."""
        ladder = ["low", "high", "xhigh", "max"]
        if self.reasoning_effort != "auto":
            return self.reasoning_effort
        effort = (client_effort or "").strip().lower()
        if effort in ladder:
            return effort
        if effort == "minimal":
            return "low"
        if effort == "none":
            return "none"
        # "medium" is not supported upstream; bump to the nearest accepted tier
        if effort == "medium":
            return "high"
        return self.default_effort
        
    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith('sk-'):
            return False
        return True
        
    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True
            
        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key
    
    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}
        
        # Get all environment variables
        env_vars = dict(os.environ)
        
        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith('CUSTOM_HEADER_'):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix
                
                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace('_', '-')
                    custom_headers[header_name] = env_value
        
        return custom_headers

try:
    config = Config()
    print(f" Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'")
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
