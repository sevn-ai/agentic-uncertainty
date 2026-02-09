"""Foundry (Azure-hosted Anthropic) model implementation."""

import os

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.cache_control import set_cache_control


class FoundryModelConfig(LitellmModelConfig):
    """Configuration for Foundry model with Anthropic cache control."""

    set_cache_control: str | None = "default_end"


class FoundryModel(LitellmModel):
    """Model class for Anthropic models via Microsoft Foundry (Azure).

    This class automatically configures LiteLLM to use the Foundry endpoint
    specified in environment variables. It also enables cache control markers
    for Anthropic models.

    Required environment variables:
        FOUNDRY_BASE_URL: The Foundry endpoint URL (e.g., https://your-foundry.azure.com)
        FOUNDRY_API_KEY: The Foundry API key

    Example usage:
        # Command line
        python -m minisweagent.run.extra.swebench \\
            --model "claude-sonnet-4-5-20250514" \\
            --model-class foundry \\
            ...

        # Python
        from minisweagent.models.foundry import FoundryModel
        model = FoundryModel(model_name="claude-sonnet-4-5-20250514")
    """

    def __init__(self, *, config_class: type = FoundryModelConfig, **kwargs):
        """Initialize the Foundry model.

        Args:
            config_class: Configuration class to use (default: FoundryModelConfig)
            **kwargs: Additional arguments passed to LitellmModel

        Raises:
            ValueError: If FOUNDRY_BASE_URL or FOUNDRY_API_KEY are not set
        """
        kwargs.setdefault("model_kwargs", {})

        # Get Foundry configuration from environment
        base_url = os.getenv("FOUNDRY_BASE_URL")
        api_key = os.getenv("FOUNDRY_API_KEY")

        if not base_url or not api_key:
            raise ValueError(
                "Foundry requires FOUNDRY_BASE_URL and FOUNDRY_API_KEY environment variables. "
                "See code/agentic_uncertainty/.env.example for configuration details."
            )

        # Configure LiteLLM to use Foundry endpoint
        kwargs["model_kwargs"]["api_base"] = base_url
        kwargs["model_kwargs"]["api_key"] = api_key

        # Prefix model name with anthropic/ for LiteLLM to use Anthropic provider
        if "model_name" in kwargs and not kwargs["model_name"].startswith("anthropic/"):
            kwargs["model_name"] = f"anthropic/{kwargs['model_name']}"

        super().__init__(config_class=config_class, **kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        """Query the model with cache control for Anthropic.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments passed to the model

        Returns:
            Response dictionary with 'content' and 'extra' keys
        """
        messages = set_cache_control(messages, mode="default_end")
        return super().query(messages, **kwargs)
