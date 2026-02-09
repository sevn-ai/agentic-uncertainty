"""Base class for uncertainty estimators."""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_uncertainty.config import Settings, get_settings
from agentic_uncertainty.data import Task
from agentic_uncertainty.providers import ModelClient, Provider, create_client

if TYPE_CHECKING:
    pass


@dataclass
class EstimationResult:
    """Result of an uncertainty estimation."""

    probability: float | None  # p(resolved) in [0, 1], or None if extraction failed
    raw_response: str  # The model's raw response
    metadata: dict  # Additional info (e.g., sampled responses for aggregation)

    @property
    def has_valid_confidence(self) -> bool:
        """Check if this result has a valid confidence value."""
        return self.probability is not None


class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimators."""

    def __init__(
        self,
        client: ModelClient | None = None,
        settings: Settings | None = None,
    ):
        """Initialize the estimator with optional explicit client and settings.

        Args:
            client: Optional ModelClient instance. Created from settings if None.
            settings: Optional Settings instance. Uses get_settings() if None.
        """
        self.settings = settings if settings is not None else get_settings()

        if client is not None:
            self.client = client
        else:
            # Create provider-appropriate client via factory
            # Factory reads credentials from environment when not provided
            provider = Provider(self.settings.provider)
            self.client = create_client(provider=provider)

    @abstractmethod
    async def estimate(self, task: Task) -> EstimationResult:
        """Estimate the probability of successfully resolving the task.

        Args:
            task: The SWE-bench Pro task to estimate success for.

        Returns:
            EstimationResult with probability in [0, 1].
        """
        pass

    async def _call_model_async(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Call the API with a prompt, retrying on rate limits."""
        if temperature is None:
            temperature = self.settings.temperature

        attempt = 0
        delay = self.settings.rate_limit_base_delay

        while True:
            try:
                # Use the abstracted client interface
                return await self.client.complete(
                    prompt=prompt,
                    model=self.settings.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                # Delegate rate limit detection to client
                if not self.client.is_rate_limit_error(exc):
                    raise
                if attempt >= self.settings.rate_limit_max_retries:
                    raise

                retry_after = self.client.get_retry_after(exc)
                if retry_after is None:
                    sleep_for = min(delay, self.settings.rate_limit_max_delay)
                    sleep_for += random.uniform(0, sleep_for * 0.1)
                    delay = min(delay * 2, self.settings.rate_limit_max_delay)
                else:
                    sleep_for = max(retry_after, self.settings.rate_limit_base_delay)

                attempt += 1
                await asyncio.sleep(sleep_for)

    async def _call_model_batch(
        self,
        prompts: list[str],
        temperature: float | None = None,
        max_tokens: int = 1024,
        max_concurrency: int | None = None,
    ) -> list[str]:
        """Call model on multiple prompts concurrently.

        Args:
            prompts: List of prompts to process.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per response.
            max_concurrency: Maximum concurrent requests.

        Returns:
            List of responses in same order as prompts.
        """
        if max_concurrency is None:
            max_concurrency = self.settings.max_concurrency

        semaphore = asyncio.Semaphore(max_concurrency)

        async def call_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self._call_model_async(prompt, temperature, max_tokens)

        tasks = [call_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def _parse_confidence(self, response: str) -> float | None:
        """Parse a confidence value from the model's response.

        Looks for patterns like "75%", "0.75", or "75" and converts to [0, 1].

        Returns:
            Confidence value in [0, 1], or None if parsing fails.
        """
        import logging
        import re

        logger = logging.getLogger(__name__)

        # Try to find percentage pattern (e.g., "75%")
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", response)
        if percent_match:
            return float(percent_match.group(1)) / 100.0

        # Try to find decimal pattern (e.g., "0.75")
        decimal_match = re.search(r"\b0?\.\d+\b", response)
        if decimal_match:
            return float(decimal_match.group())

        # Try to find standalone number (interpret as percentage if > 1)
        number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
        if number_match:
            value = float(number_match.group(1))
            if value > 1:
                return min(value / 100.0, 1.0)
            return value

        # Return None if parsing fails - caller should handle/filter this case
        logger.warning(
            "Failed to parse confidence from response. "
            "This result will be filtered out. Response preview: %s",
            response[:200] if len(response) > 200 else response,
        )
        return None


class PromptDrivenEstimator(UncertaintyEstimator):
    """Generic estimator driven by prompt files.

    Prompt-template methods were removed from the paper-only public release.
    """

    def __init__(
        self,
        prompt_name: str,
        client: ModelClient | None = None,
        settings: Settings | None = None,
    ):
        """Initialize the estimator.

        Args:
            prompt_name: Name of the prompt file (e.g., "direct.md").
            client: Optional ModelClient instance.
            settings: Optional Settings instance.
        """
        super().__init__(client=client, settings=settings)
        self.prompt_name = prompt_name

    async def estimate(self, task: Task) -> EstimationResult:
        """Prompt-driven estimation is unavailable in this release."""
        raise RuntimeError(
            "PromptDrivenEstimator is not available in this paper-only public release. "
            "Use exploration/review/mid_execution/checkpoint methods instead."
        )
