"""LiteLLM client implementation for model-agnostic API access."""

import litellm
from litellm import RateLimitError


def is_reasoning_model(model: str | None) -> bool:
    """Check if model is a reasoning model that doesn't support temperature."""
    if not model:
        return False
    reasoning_patterns = ["o1", "o3", "codex", "reasoning"]
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in reasoning_patterns)


class LiteLLMClient:
    """LiteLLM client implementing ModelClient protocol.

    Uses LiteLLM for model-agnostic API access, supporting OpenAI, Anthropic,
    and many other providers through a unified interface.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the LiteLLM client.

        Args:
            api_key: Optional API key (can also be set via environment).
            base_url: Optional custom base URL.
        """
        self._api_key = api_key
        self._base_url = base_url

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make an async completion request via LiteLLM.

        Uses the standard chat completions format which is widely supported.
        """
        messages = [{"role": "user", "content": prompt}]

        # Build request kwargs - don't include max_tokens for codex models
        # as they may not support it or use different parameter names
        kwargs = {
            "model": model,
            "messages": messages,
        }

        # Only add max_tokens for non-codex models
        if model and "codex" not in model.lower():
            kwargs["max_tokens"] = max_tokens

        # Skip temperature for reasoning models
        if not is_reasoning_model(model):
            kwargs["temperature"] = temperature

        # Add optional parameters
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url

        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        if isinstance(exc, RateLimitError):
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        return False

    def get_retry_after(self, exc: Exception) -> float | None:
        """Extract retry delay from headers when available."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    return None
        return None
