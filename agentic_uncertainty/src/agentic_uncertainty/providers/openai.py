"""OpenAI Responses API client implementation."""

import openai
from openai import AsyncOpenAI


def is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model that doesn't support temperature."""
    reasoning_patterns = ["o1", "o3", "codex", "reasoning"]
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in reasoning_patterns)


class OpenAIClient:
    """OpenAI Responses API client implementing ModelClient protocol."""

    def __init__(self, api_key: str, base_url: str | None = None):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key.
            base_url: Optional custom base URL.
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make an async completion request via OpenAI Responses API.

        Uses the Responses API format:
        client.responses.create(model, input, ...) -> response.output_text
        """
        # Build request kwargs - skip temperature for reasoning models
        kwargs = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if not is_reasoning_model(model):
            kwargs["temperature"] = temperature

        response = await self._client.responses.create(**kwargs)
        return response.output_text

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        if isinstance(exc, openai.RateLimitError):
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
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
