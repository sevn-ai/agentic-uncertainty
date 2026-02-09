"""Smart parser for extracting confidence values from model responses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a model response."""

    probability: float | None  # p(resolved) in [0, 1], or None if extraction failed
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_valid_confidence(self) -> bool:
        """Check if this result has a valid confidence value."""
        return self.probability is not None


# Special parsing rules for specific prompts
SPECIAL_PARSING = {
    "failure_confidence.md": {"invert": True},
    "terminal_failure_confidence.md": {"invert": True},
}


def _extract_all_xml_tags(response: str) -> dict[str, str]:
    """Extract all XML-style tags from response."""
    pattern = r"<(\w+)>\s*(.*?)\s*</\1>"
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
    return {tag.lower(): value for tag, value in matches}


def _parse_number(text: str) -> float | None:
    """Parse a number from text, handling percentages."""
    # Try percentage
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1)) / 100.0

    # Try decimal
    match = re.search(r"\b0?\.\d+\b", text)
    if match:
        return float(match.group())

    # Try integer/float
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        value = float(match.group(1))
        return value / 100.0 if value > 1 else value

    return None


def _parse_confidence_fuzzy(response: str) -> float | None:
    """Fuzzy parse a confidence value from anywhere in the response.

    Returns:
        Confidence value in [0, 1], or None if no number found.
    """
    return _parse_number(response)


def parse_response(response: str, prompt_name: str = "") -> ParseResult:
    """Parse a model response to extract probability and metadata.

    This is a smart parser that:
    1. Extracts all XML tags found in the response into metadata
    2. Uses <confidence> tag value if present, otherwise fuzzy regex
    3. Applies special rules (like invert) based on prompt_name

    Args:
        response: The model's raw response text.
        prompt_name: Optional prompt filename for special handling.

    Returns:
        ParseResult with probability in [0,1] (or None if extraction failed)
        and extracted metadata.
    """
    # Extract all XML tags
    tags = _extract_all_xml_tags(response)
    metadata: dict[str, Any] = {}

    # Convert tag values to numbers where possible and add to metadata
    for tag, value in tags.items():
        num = _parse_number(value)
        if num is not None:
            metadata[tag] = num
        else:
            metadata[tag] = value

    # Determine probability: prefer <confidence>, then <failure_confidence>, then fuzzy
    if "confidence" in tags:
        probability = _parse_number(tags["confidence"])
        if probability is None:
            probability = _parse_confidence_fuzzy(response)
    elif "failure_confidence" in tags:
        probability = _parse_number(tags["failure_confidence"])
        if probability is None:
            probability = _parse_confidence_fuzzy(response)
    else:
        probability = _parse_confidence_fuzzy(response)

    # If we couldn't extract a probability, log a warning
    if probability is None:
        logger.warning(
            "Failed to parse confidence from model response. "
            "This result should be filtered out. Response preview: %s",
            response[:200] if len(response) > 200 else response,
        )
        return ParseResult(probability=None, metadata=metadata)

    # Apply special rules
    special = SPECIAL_PARSING.get(prompt_name, {})
    if special.get("invert", False):
        probability = 1.0 - probability

    return ParseResult(
        probability=min(max(probability, 0.0), 1.0),
        metadata=metadata,
    )
