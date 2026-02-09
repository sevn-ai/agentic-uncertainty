"""Settings loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # Provider configuration
    provider: Literal["anthropic", "openai", "gemini", "grok", "litellm"] = "anthropic"
    api_key: str = ""
    base_url: str | None = None  # For Azure/Foundry endpoints
    model: str = ""

    # Gemini-specific configuration
    project: str | None = None  # Google Cloud project ID
    location: str = ""  # Google Cloud location

    # Elicitation parameters
    temperature: float = 0.0  # For primary estimates
    sampling_temperature: float = 0.9  # For sampling-based methods (higher for more variance)
    num_samples: int = 10  # K for self-consistency and outcome sampling
    max_concurrency: int = 10  # Max concurrent API calls for batch processing
    rate_limit_max_retries: int = 5  # Max retries for rate-limited requests
    rate_limit_base_delay: float = 1.0  # Initial backoff delay (seconds)
    rate_limit_max_delay: float = 30.0  # Max backoff delay (seconds)

    # Prompt parameters
    max_problem_chars: int = 10000  # Max chars for problem statement in prompts
    max_trajectory_chars: int = 10000  # Max chars for trajectory context in checkpoint prompts

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Settings":
        """Load settings from .env file and environment variables."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # Determine provider and get appropriate API key
        explicit_provider = os.getenv("PROVIDER", "").lower()

        # Default values
        project = None
        location = "us-central1"

        if explicit_provider == "grok" or (
            not explicit_provider and os.getenv("GROK_API_KEY")
        ):
            provider = "grok"
            api_key = os.getenv("GROK_API_KEY")
            base_url = os.getenv("GROK_BASE_URL")
            default_model = "grok-3"
        elif explicit_provider == "litellm" or (
            not explicit_provider and os.getenv("OPENAI_API_KEY")
        ):
            # Use LiteLLM for OpenAI-compatible models (standard chat completions API)
            provider = "litellm"
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            default_model = "gpt-4o"
        elif explicit_provider == "openai":
            # Explicit openai provider uses Responses API
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            default_model = "gpt-4o"
        elif explicit_provider == "gemini" or (
            not explicit_provider and os.getenv("GOOGLE_CLOUD_PROJECT")
        ):
            provider = "gemini"
            api_key = ""  # Gemini uses ADC, not API key
            base_url = None
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            default_model = "gemini-2.5-flash"
        else:
            provider = "anthropic"
            foundry_key = os.getenv("FOUNDRY_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            api_key = foundry_key or anthropic_key
            # Only use FOUNDRY_BASE_URL if using Foundry key
            # Standard Anthropic API doesn't need base_url (uses api.anthropic.com)
            base_url = os.getenv("FOUNDRY_BASE_URL") if foundry_key else None
            default_model = "claude-sonnet-4-20250514"

        # Validate credentials
        if provider == "gemini":
            if not project:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT not found. Set it in your .env file."
                )
        elif provider == "grok":
            if not api_key:
                raise ValueError("GROK_API_KEY not found. Set it in your .env file.")
            if not base_url:
                raise ValueError("GROK_BASE_URL not found. Set it in your .env file.")
        elif not api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or FOUNDRY_API_KEY in your .env file."
            )

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("MODEL", default_model),
            project=project,
            location=location,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance, loading from environment if needed."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Clear cached settings so they will be reloaded from environment."""
    global _settings
    _settings = None
