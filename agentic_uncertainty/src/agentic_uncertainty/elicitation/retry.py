"""Shared retry utilities for agent elicitation."""

from __future__ import annotations

import asyncio
import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Try to import LLM API error types
try:
    from litellm.exceptions import APIError as LiteLLMAPIError
except ImportError:
    LiteLLMAPIError = None

try:
    from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError
except ImportError:
    LiteLLMAPIConnectionError = None

try:
    from openai import APIError as OpenAIAPIError
except ImportError:
    OpenAIAPIError = None

try:
    from openai import APIConnectionError as OpenAIAPIConnectionError
except ImportError:
    OpenAIAPIConnectionError = None

try:
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
except ImportError:
    LiteLLMBadRequestError = None

try:
    from litellm.exceptions import ServiceUnavailableError as LiteLLMServiceUnavailableError
except ImportError:
    LiteLLMServiceUnavailableError = None

try:
    from litellm import BadGatewayError as LiteLLMBadGatewayError
except ImportError:
    try:
        from litellm.exceptions import BadGatewayError as LiteLLMBadGatewayError
    except ImportError:
        LiteLLMBadGatewayError = None

try:
    from litellm import RateLimitError as LiteLLMRateLimitError
except ImportError:
    try:
        from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
    except ImportError:
        LiteLLMRateLimitError = None

# Google API errors (for Gemini models)
try:
    from google.api_core.exceptions import ServiceUnavailable as GoogleServiceUnavailable
except ImportError:
    GoogleServiceUnavailable = None

try:
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
except ImportError:
    GoogleResourceExhausted = None

try:
    from google.api_core.exceptions import InternalServerError as GoogleInternalServerError
except ImportError:
    GoogleInternalServerError = None

try:
    from google.api_core.exceptions import DeadlineExceeded as GoogleDeadlineExceeded
except ImportError:
    GoogleDeadlineExceeded = None

# Exceptions that should trigger a retry across agents.
_base_exceptions = [
    asyncio.TimeoutError,
    ConnectionError,
    TimeoutError,
    OSError,  # Includes network errors
]

# Add LLM API errors if available
if LiteLLMAPIError:
    _base_exceptions.append(LiteLLMAPIError)
if LiteLLMAPIConnectionError:
    _base_exceptions.append(LiteLLMAPIConnectionError)
if OpenAIAPIError:
    _base_exceptions.append(OpenAIAPIError)
if OpenAIAPIConnectionError:
    _base_exceptions.append(OpenAIAPIConnectionError)
if LiteLLMBadGatewayError:
    _base_exceptions.append(LiteLLMBadGatewayError)
if LiteLLMRateLimitError:
    _base_exceptions.append(LiteLLMRateLimitError)
if LiteLLMServiceUnavailableError:
    _base_exceptions.append(LiteLLMServiceUnavailableError)

# Add Google API errors if available
if GoogleServiceUnavailable:
    _base_exceptions.append(GoogleServiceUnavailable)
if GoogleResourceExhausted:
    _base_exceptions.append(GoogleResourceExhausted)
if GoogleInternalServerError:
    _base_exceptions.append(GoogleInternalServerError)
if GoogleDeadlineExceeded:
    _base_exceptions.append(GoogleDeadlineExceeded)

RETRYABLE_EXCEPTIONS = tuple(_base_exceptions)


def retry_async(
    logger: logging.Logger,
    *,
    attempts: int = 5,  # Increased for API reliability
    min_wait: int = 4,
    max_wait: int = 60,
):
    """Return a tenacity decorator for async agent runs."""
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
