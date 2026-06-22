"""Configuration for Cognee instrumentation."""

from __future__ import annotations

import os


def get_bool_env(key: str, default: bool = False) -> bool:
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def get_optional_bool_env(key: str) -> "bool | None":
    raw = os.getenv(key)
    if raw is None:
        return None
    raw_lower = raw.lower()
    if raw_lower in ("true", "1", "yes", "on"):
        return True
    if raw_lower in ("false", "0", "no", "off"):
        return False
    return None


def first_present_bool(keys: list[str], default: bool) -> bool:
    for key in keys:
        value = get_optional_bool_env(key)
        if value is not None:
            return value
    return default


_CAPTURE_MESSAGE_CONTENT_KEYS: list[str] = [
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
    "otel.instrumentation.genai.capture_message_content",
    "COGNEE_CAPTURE_MESSAGE_CONTENT",
]


def is_capture_message_content_enabled() -> bool:
    return first_present_bool(_CAPTURE_MESSAGE_CONTENT_KEYS, False)


_INTERNAL_PHASES_KEYS: list[str] = [
    "OTEL_INSTRUMENTATION_COGNEE_INTERNAL_ENABLED",
    "otel.instrumentation.cognee.internal.enabled",
]


def is_internal_phases_enabled() -> bool:
    return first_present_bool(_INTERNAL_PHASES_KEYS, False)


_REACT_STEP_KEYS: list[str] = [
    "OTEL_INSTRUMENTATION_COGNEE_REACT_STEP_ENABLED",
    "otel.instrumentation.cognee.react_step.enabled",
]


def is_react_step_enabled() -> bool:
    return first_present_bool(_REACT_STEP_KEYS, True)


MAX_PAYLOAD_BYTES = 4096
