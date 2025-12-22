"""
Duration string parser utility for converting Elasticsearch-style duration strings to seconds.

Supports duration formats like "1d" (days) and "6h" (hours).
"""
import re
from typing import Union


# Regex pattern for duration strings: {number}{unit}
# Supports: d (days), h (hours)
# Examples: "1d", "7d", "0.5d", "6h", "12h", "1.5h"
DURATION_PATTERN = re.compile(r'^([0-9]+(?:\.[0-9]+)?)(d|h)$')

# Conversion factors to seconds
UNIT_TO_SECONDS = {
    'd': 24 * 60 * 60,  # days to seconds
    'h': 60 * 60,       # hours to seconds
}


def parse_duration_to_seconds(duration: str) -> float:
    """
    Parse an Elasticsearch-style duration string to seconds.

    Supported units:
    - d: days
    - h: hours

    Args:
        duration: Duration string in format "{number}{unit}" (e.g., "1d", "6h", "0.5d")

    Returns:
        Duration in seconds as a float

    Raises:
        ValueError: If duration string format is invalid or value is negative

    Examples:
        >>> parse_duration_to_seconds("1d")
        86400.0
        >>> parse_duration_to_seconds("6h")
        21600.0
        >>> parse_duration_to_seconds("0.5d")
        43200.0
        >>> parse_duration_to_seconds("1.5h")
        5400.0
    """
    if not isinstance(duration, str):
        raise ValueError(f"Duration must be a string, got {type(duration).__name__}")

    match = DURATION_PATTERN.match(duration)
    if not match:
        raise ValueError(
            f"Invalid duration format: '{duration}'. "
            f"Expected format: {{number}}{{unit}} where unit is 'd' (days) or 'h' (hours). "
            f"Examples: '1d', '6h', '0.5d'"
        )

    value_str, unit = match.groups()
    value = float(value_str)

    if value < 0:
        raise ValueError(f"Duration value cannot be negative: {duration}")

    seconds = value * UNIT_TO_SECONDS[unit]
    return seconds
