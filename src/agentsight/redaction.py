"""
Payload hygiene: redaction, truncation, and safe serialization.

Controls what ends up in span attributes and log payloads.
Production systems MUST configure redaction patterns for PII,
credentials, and large payloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Default patterns that match common secret formats
_DEFAULT_REDACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(secret|token)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(authorization)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(bearer)\s+\S+"),
    re.compile(r"(?i)(aws_secret_access_key|aws_access_key_id)\s*[:=]\s*\S+"),
    # Credit card numbers (basic pattern)
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    # SSN
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
]

_REDACTED = "[REDACTED]"


@dataclass
class PayloadPolicy:
    """
    Controls what goes into OTel attributes.

    Attributes:
        max_str_len:       Truncate string values beyond this length.
        max_attr_count:    Maximum number of attributes per span/event.
        redact_patterns:   Regex patterns; matching substrings are replaced.
        redact_keys:       Attribute key names to fully redact (case-insensitive).
        allow_keys:        If non-empty, ONLY these keys pass through (allowlist mode).
        drop_keys:         Keys to silently drop (never recorded).
    """

    max_str_len: int = 4096
    max_attr_count: int = 64
    redact_patterns: list[re.Pattern[str]] = field(default_factory=lambda: list(_DEFAULT_REDACT_PATTERNS))
    redact_keys: set[str] = field(
        default_factory=lambda: {
            "password",
            "passwd",
            "secret",
            "token",
            "api_key",
            "apikey",
            "authorization",
            "credentials",
        }
    )
    allow_keys: set[str] = field(default_factory=set)
    drop_keys: set[str] = field(default_factory=set)


def sanitize_attributes(attrs: dict[str, Any], policy: PayloadPolicy) -> dict[str, Any]:
    """
    Apply redaction, truncation, and filtering to an attribute dict.

    Returns a new dict; never mutates the input.
    """
    result: dict[str, Any] = {}
    count = 0

    for key, value in attrs.items():
        if count >= policy.max_attr_count:
            break

        key_lower = key.lower()

        # Drop list
        if key_lower in {k.lower() for k in policy.drop_keys}:
            continue

        # Allowlist mode
        if policy.allow_keys and key_lower not in {k.lower() for k in policy.allow_keys}:
            continue

        # Full redaction by key name
        if key_lower in {k.lower() for k in policy.redact_keys}:
            result[key] = _REDACTED
            count += 1
            continue

        # Sanitize value
        result[key] = _sanitize_value(value, policy)
        count += 1

    return result


def _sanitize_value(value: Any, policy: PayloadPolicy) -> Any:
    """Recursively sanitize a single value."""
    if isinstance(value, str):
        return _sanitize_string(value, policy)
    if isinstance(value, dict):
        return sanitize_attributes(value, policy)
    if isinstance(value, (list, tuple)):
        sanitized = [_sanitize_value(v, policy) for v in value[:policy.max_attr_count]]
        return type(value)(sanitized) if isinstance(value, tuple) else sanitized
    if isinstance(value, (int, float, bool)):
        return value
    # Fallback: convert to string and sanitize
    return _sanitize_string(str(value), policy)


def _sanitize_string(value: str, policy: PayloadPolicy) -> str:
    """Apply pattern redaction and truncation to a string."""
    result = value
    for pattern in policy.redact_patterns:
        result = pattern.sub(_REDACTED, result)
    if len(result) > policy.max_str_len:
        result = result[: policy.max_str_len] + f"... [truncated, original length: {len(value)}]"
    return result
