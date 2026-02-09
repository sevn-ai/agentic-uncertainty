"""Base protocol for model clients."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelClient(Protocol):
    """Protocol defining the interface for model clients.

    This protocol ensures all provider implementations expose
    the same async interface for making model calls.
    """

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make an async completion request.

        Args:
            prompt: The user prompt to send.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            The model's text response.
        """
        ...

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error.

        Args:
            exc: The exception to check.

        Returns:
            True if this is a rate limit error.
        """
        ...

    def get_retry_after(self, exc: Exception) -> float | None:
        """Extract retry-after delay from exception headers.

        Args:
            exc: The exception with response headers.

        Returns:
            Delay in seconds, or None if not available.
        """
        ...
