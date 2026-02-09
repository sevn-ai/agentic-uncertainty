import os
import warnings
from typing import Literal

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.key_per_thread import get_key_per_thread


class AnthropicModelConfig(LitellmModelConfig):
    set_cache_control: Literal["default_end"] | None = "default_end"
    """Set explicit cache control markers, for example for Anthropic models"""


class AnthropicModel(LitellmModel):
    """Model class for Anthropic models via the standard Anthropic API (api.anthropic.com).

    This class configures LiteLLM to use the standard Anthropic API endpoint.
    It automatically prefixes model names with 'anthropic/' for LiteLLM compatibility
    and enables cache control markers.

    Required environment variables:
        ANTHROPIC_API_KEY: Your Anthropic API key from https://console.anthropic.com/

    Example usage:
        # Command line
        mini --model "claude-sonnet-4-5-20250514" --model-class anthropic

        # Python
        from minisweagent.models.anthropic import AnthropicModel
        model = AnthropicModel(model_name="claude-sonnet-4-5-20250514")

    Note: For Azure Foundry deployments, use FoundryModel instead.
    """

    def __init__(self, *, config_class: type = AnthropicModelConfig, **kwargs):
        # Prefix model name with anthropic/ for LiteLLM to use the Anthropic provider
        if "model_name" in kwargs and not kwargs["model_name"].startswith("anthropic/"):
            kwargs["model_name"] = f"anthropic/{kwargs['model_name']}"

        super().__init__(config_class=config_class, **kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        api_key = None
        # Legacy only
        if rotating_keys := os.getenv("ANTHROPIC_API_KEYS"):
            warnings.warn(
                "ANTHROPIC_API_KEYS is deprecated and will be removed in the future. "
                "Simply use the ANTHROPIC_API_KEY environment variable instead. "
                "Key rotation is no longer required."
            )
            api_key = get_key_per_thread(rotating_keys.split("::"))
        messages = set_cache_control(messages, mode="default_end")
        return super().query(messages, api_key=api_key, **kwargs)
