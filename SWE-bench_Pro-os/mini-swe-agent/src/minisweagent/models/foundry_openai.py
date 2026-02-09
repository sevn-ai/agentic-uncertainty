"""Foundry OpenAI-compatible model implementation (for Grok, etc.)."""

import os

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig


class FoundryOpenAIModelConfig(LitellmModelConfig):
    """Configuration for Foundry OpenAI-compatible models."""

    pass


class FoundryOpenAIModel(LitellmModel):
    """Model class for OpenAI-compatible models via Microsoft Foundry (Azure).

    This class automatically configures LiteLLM to use the Foundry OpenAI endpoint
    for models like Grok that use OpenAI-compatible APIs.

    Required environment variables:
        FOUNDRY_OPENAI_BASE_URL: The Foundry OpenAI endpoint (e.g., https://your-foundry.azure.com/openai/v1/)
        FOUNDRY_API_KEY: The Foundry API key (shared with Anthropic endpoint)

    Example usage:
        # Command line
        python -m minisweagent.run.extra.swebench \\
            --model "grok-4-fast-reasoning" \\
            --model-class foundry_openai \\
            ...

        # Python
        from minisweagent.models.foundry_openai import FoundryOpenAIModel
        model = FoundryOpenAIModel(model_name="grok-4-fast-reasoning")
    """

    def __init__(self, *, config_class: type = FoundryOpenAIModelConfig, **kwargs):
        """Initialize the Foundry OpenAI-compatible model.

        Args:
            config_class: Configuration class to use (default: FoundryOpenAIModelConfig)
            **kwargs: Additional arguments passed to LitellmModel

        Raises:
            ValueError: If FOUNDRY_OPENAI_BASE_URL or FOUNDRY_API_KEY are not set
        """
        kwargs.setdefault("model_kwargs", {})

        # Get Foundry configuration from environment
        base_url = os.getenv("FOUNDRY_OPENAI_BASE_URL")
        api_key = os.getenv("FOUNDRY_API_KEY")

        if not base_url or not api_key:
            raise ValueError(
                "Foundry OpenAI requires FOUNDRY_OPENAI_BASE_URL and FOUNDRY_API_KEY environment variables. "
                "See code/agentic_uncertainty/.env.example for configuration details."
            )

        # Configure LiteLLM to use Foundry OpenAI endpoint
        kwargs["model_kwargs"]["api_base"] = base_url
        kwargs["model_kwargs"]["api_key"] = api_key

        # Prefix model name with openai/ for LiteLLM to use OpenAI-compatible API
        if "model_name" in kwargs and not kwargs["model_name"].startswith("openai/"):
            kwargs["model_name"] = f"openai/{kwargs['model_name']}"

        super().__init__(config_class=config_class, **kwargs)
