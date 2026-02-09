"""Factory for creating model clients based on configuration."""

import os
from enum import Enum

from .anthropic import AnthropicClient
from .base import ModelClient
from .gemini import GeminiClient
from .litellm_client import LiteLLMClient
from .openai import OpenAIClient


class Provider(Enum):
    """Supported model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    GROK = "grok"  # Grok via Azure Foundry (OpenAI-compatible)
    LITELLM = "litellm"  # Model-agnostic via LiteLLM


def detect_provider() -> Provider:
    """Detect the provider from environment variables.

    Priority order:
    1. Explicit PROVIDER env var if set
    2. GROK_API_KEY -> Grok
    3. OPENAI_API_KEY -> LiteLLM (uses standard chat completions API)
    4. GOOGLE_CLOUD_PROJECT -> Gemini
    5. ANTHROPIC_API_KEY or FOUNDRY_API_KEY -> Anthropic

    Returns:
        The detected Provider enum value.

    Raises:
        ValueError: If no API key/project is found.
    """
    # Allow explicit override
    explicit = os.getenv("PROVIDER", "").lower()
    if explicit == "openai":
        return Provider.OPENAI
    if explicit == "litellm":
        return Provider.LITELLM
    if explicit == "gemini":
        return Provider.GEMINI
    if explicit == "grok":
        return Provider.GROK
    if explicit in ("anthropic", "foundry"):
        return Provider.ANTHROPIC

    # Auto-detect from API keys/project
    if os.getenv("GROK_API_KEY"):
        return Provider.GROK
    if os.getenv("OPENAI_API_KEY"):
        # Use LiteLLM for OpenAI models - it uses standard chat completions API
        # which is more widely supported than the Responses API
        return Provider.LITELLM
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_CLOUD_PROJECT"):
        return Provider.GEMINI
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("FOUNDRY_API_KEY"):
        return Provider.ANTHROPIC

    raise ValueError(
        "No API key/project found. Set GROK_API_KEY, OPENAI_API_KEY, "
        "GOOGLE_CLOUD_PROJECT, ANTHROPIC_API_KEY, or FOUNDRY_API_KEY."
    )


def create_client(
    provider: Provider | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    project: str | None = None,
    location: str | None = None,
) -> ModelClient:
    """Create a model client for the specified or detected provider.

    Args:
        provider: Explicit provider choice. Auto-detected if None.
        api_key: API key. Read from environment if None.
        base_url: Optional custom base URL.
        project: Google Cloud project ID (for Gemini).
        location: Google Cloud location (for Gemini).

    Returns:
        A ModelClient implementation for the provider.

    Raises:
        ValueError: If provider is unknown or API key/project missing.
    """
    if provider is None:
        provider = detect_provider()

    if provider == Provider.ANTHROPIC:
        # Check for Foundry first, then standard Anthropic API
        foundry_key = os.getenv("FOUNDRY_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if api_key is None:
            api_key = foundry_key or anthropic_key
        if not api_key:
            raise ValueError("Anthropic API key not found")

        # Only use FOUNDRY_BASE_URL if using Foundry key (not standard Anthropic API)
        if base_url is None and foundry_key:
            base_url = os.getenv("FOUNDRY_BASE_URL")
        # If using ANTHROPIC_API_KEY without FOUNDRY_API_KEY, base_url stays None
        # (uses standard api.anthropic.com)

        return AnthropicClient(api_key=api_key, base_url=base_url)

    elif provider == Provider.OPENAI:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAIClient(api_key=api_key, base_url=base_url)

    elif provider == Provider.GEMINI:
        # Prefer API key (simpler), fall back to Vertex AI
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            return GeminiClient(api_key=gemini_api_key)
        # Vertex AI mode
        if project is None:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError("GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT not found")
        if location is None:
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        return GeminiClient(project=project, location=location)

    elif provider == Provider.GROK:
        # Grok via Azure Foundry uses OpenAI-compatible API
        if api_key is None:
            api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY not found")
        if base_url is None:
            base_url = os.getenv("GROK_BASE_URL")
        if not base_url:
            raise ValueError("GROK_BASE_URL not found")
        return OpenAIClient(api_key=api_key, base_url=base_url)

    elif provider == Provider.LITELLM:
        # LiteLLM provides model-agnostic access via standard chat completions API
        # It reads API keys from environment (OPENAI_API_KEY, etc.)
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
        return LiteLLMClient(api_key=api_key, base_url=base_url)

    else:
        raise ValueError(f"Unknown provider: {provider}")
