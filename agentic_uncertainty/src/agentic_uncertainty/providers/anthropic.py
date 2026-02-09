"""Anthropic client implementation."""

import anthropic


class AnthropicClient:
    """Anthropic API client implementing ModelClient protocol."""

    def __init__(self, api_key: str, base_url: str | None = None):
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key.
            base_url: Optional custom base URL (for Foundry/Azure).
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make an async completion request via Anthropic Messages API."""
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        if isinstance(exc, anthropic.RateLimitError):
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
