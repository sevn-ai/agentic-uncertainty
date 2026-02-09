"""Google Gemini client implementation via API key or Vertex AI."""

from google import genai
from google.genai import types


class GeminiClient:
    """Gemini API client implementing ModelClient protocol.

    Supports two modes:
    - API key: Simple authentication with GEMINI_API_KEY
    - Vertex AI: GCP project-based auth (requires gcloud setup)
    """

    def __init__(
        self,
        api_key: str | None = None,
        project: str | None = None,
        location: str = "us-central1",
    ):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key (preferred, simpler setup).
            project: Google Cloud project ID (for Vertex AI mode).
            location: Google Cloud location (default: us-central1).

        If api_key is provided, uses direct Gemini API.
        Otherwise, uses Vertex AI with project/location.
        """
        if api_key:
            self._client = genai.Client(api_key=api_key)
        elif project:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            raise ValueError("Either api_key or project must be provided")

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make an async completion request via Gemini API.

        Uses the Vertex AI Gemini API:
        client.aio.models.generate_content(...) -> response.text
        """
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        # Check for Google API resource exhausted error
        exc_type = type(exc).__name__
        if exc_type == "ResourceExhausted":
            return True

        # Check status code
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True

        # Check for code attribute (grpc style)
        code = getattr(exc, "code", None)
        if code is not None:
            code_name = getattr(code, "name", str(code))
            if "RESOURCE_EXHAUSTED" in str(code_name).upper():
                return True

        # Check exception message
        exc_str = str(exc).upper()
        if "RESOURCE_EXHAUSTED" in exc_str or "RATE" in exc_str:
            return True

        return False

    def get_retry_after(self, exc: Exception) -> float | None:
        """Extract retry delay from exception metadata when available."""
        # Try to get retry info from exception metadata
        metadata = getattr(exc, "metadata", None)
        if metadata:
            for key, value in metadata:
                if key.lower() == "retry-after":
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        pass

        # Check for details with retry info
        details = getattr(exc, "details", None)
        if details and callable(details):
            try:
                detail_str = details()
                if "retry" in detail_str.lower():
                    # Try to extract numeric value
                    import re

                    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", detail_str)
                    if match:
                        return float(match.group(1))
            except Exception:
                pass

        return None
