"""Native Gemini model implementation using google-genai SDK.

This bypasses LiteLLM to avoid issues with Gemini's thinking tokens.
"""

import logging
import os
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("gemini_model")


# Pricing per 1M tokens (as of Jan 2025)
GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


def _is_retryable_error(exc: BaseException) -> bool:
    """Check if exception should be retried."""
    exc_str = str(exc).upper()
    exc_type = type(exc).__name__

    # Retry on rate limits and transient errors
    if "RESOURCE_EXHAUSTED" in exc_str or "RATE" in exc_str:
        return True
    if "429" in exc_str or "503" in exc_str or "500" in exc_str or "502" in exc_str:
        return True
    if exc_type in ("ResourceExhausted", "ServiceUnavailable", "InternalServerError"):
        return True

    # Retry on connection/protocol errors
    if "REMOTEPROTOCOLERROR" in exc_str or "PROTOCOL" in exc_str:
        return True
    if "CONNECTION" in exc_str or "TIMEOUT" in exc_str:
        return True
    if exc_type in ("RemoteProtocolError", "ConnectionError", "TimeoutError", "ReadTimeout",
                    "ConnectTimeout", "ReadError", "WriteTimeout", "PoolTimeout"):
        return True

    # Retry on ClientError (generic API errors) - these are often transient
    if exc_type == "ClientError" or "CLIENTERROR" in exc_str:
        return True

    # Retry on API errors that might be transient
    if "API" in exc_str and ("ERROR" in exc_str or "FAILED" in exc_str):
        return True

    # Retry on SSL/TLS errors
    if "SSL" in exc_str or "TLS" in exc_str or "CERTIFICATE" in exc_str:
        return True

    return False


class GeminiModelConfig(BaseModel):
    model_name: str
    temperature: float = 0.0
    max_output_tokens: int = 16384  # Higher default for models with thinking tokens (e.g., gemini-3-pro-preview)
    api_key: str | None = None


class GeminiModel:
    """Native Gemini model using google-genai SDK.

    This avoids LiteLLM issues with Gemini's thinking tokens.

    Required environment variables:
        GEMINI_API_KEY: Your Gemini API key from https://aistudio.google.com/apikey

    Example usage:
        # Command line
        mini --model "gemini-2.5-flash" --model-class minisweagent.models.gemini.GeminiModel

        # Python
        from minisweagent.models.gemini import GeminiModel
        model = GeminiModel(model_name="gemini-2.5-flash")
    """

    def __init__(self, *, config_class: type = GeminiModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in your environment or pass api_key parameter. "
                "Get your key from https://aistudio.google.com/apikey"
            )

        # Configure client with timeout to prevent hanging
        http_options = types.HttpOptions(
            timeout=120000,  # 120 seconds in milliseconds
        )
        self._client = genai.Client(api_key=api_key, http_options=http_options)

    def _convert_messages(self, messages: list[dict]) -> list[types.Content]:
        """Convert OpenAI-style messages to Gemini format."""
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # Gemini handles system as a separate parameter, but we can prepend it
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

        # Prepend system instruction to first user message if present
        if system_instruction and contents:
            first_content = contents[0]
            if first_content.role == "user":
                combined_text = f"{system_instruction}\n\n---\n\n{first_content.parts[0].text}"
                contents[0] = types.Content(role="user", parts=[types.Part(text=combined_text)])

        return contents

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        # Extract base model name (remove gemini/ prefix if present)
        base_model = model.replace("gemini/", "")

        pricing = GEMINI_PRICING.get(base_model, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "15"))),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_exception(_is_retryable_error),
    )
    def _query(self, contents: list[types.Content]) -> types.GenerateContentResponse:
        """Make the API call with retry logic."""
        return self._client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            ),
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the model and return response in expected format."""
        contents = self._convert_messages(messages)
        response = self._query(contents)

        # Extract text from response
        text = ""
        if (
            response.candidates
            and len(response.candidates) > 0
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            text = response.candidates[0].content.parts[0].text or ""

        # Calculate cost from usage metadata
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        cost = self._calculate_cost(self.config.model_name, input_tokens, output_tokens)
        self.cost += cost
        self.n_calls += 1
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": text,
            "extra": {
                "response": {
                    "model": self.config.model_name,
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": text,
                                "tool_calls": None,
                            },
                            "finish_reason": (
                                response.candidates[0].finish_reason.name
                                if (
                                    response.candidates
                                    and len(response.candidates) > 0
                                    and response.candidates[0].finish_reason is not None
                                )
                                else None
                            ),
                        }
                    ],
                },
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for prompt rendering."""
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}
