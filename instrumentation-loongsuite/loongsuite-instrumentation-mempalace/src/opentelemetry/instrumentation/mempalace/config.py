"""
Configuration for MemPalace instrumentation.
"""

from __future__ import annotations

import os


def get_bool_env(key: str, default: bool = False) -> bool:
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def get_float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


class MemPalaceInstrumentationConfig:
    _INNER_KEYS: list[str] = [
        "OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED",
        "otel.instrumentation.mempalace.inner.enabled",
    ]
    INTERNAL_PHASES_ENABLED: bool = False


def is_internal_phases_enabled() -> bool:
    for key in MemPalaceInstrumentationConfig._INNER_KEYS:
        raw = os.getenv(key)
        if raw is None:
            continue
        rl = raw.lower()
        if rl in ("true", "1", "yes", "on"):
            return True
        if rl in ("false", "0", "no", "off"):
            return False
    return MemPalaceInstrumentationConfig.INTERNAL_PHASES_ENABLED


def is_capture_message_content_enabled() -> bool:
    return get_bool_env(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", False
    )


def get_embedding_sample_rate() -> float:
    return get_float_env(
        "OTEL_INSTRUMENTATION_MEMPALACE_EMBEDDING_SAMPLE_RATE", 0.1
    )


def get_slow_search_threshold() -> float:
    return get_float_env(
        "OTEL_INSTRUMENTATION_MEMPALACE_SLOW_SEARCH_THRESHOLD_S", 2.0
    )


def get_slow_add_threshold() -> float:
    return get_float_env(
        "OTEL_INSTRUMENTATION_MEMPALACE_SLOW_ADD_THRESHOLD_S", 1.0
    )


def get_llm_slow_threshold() -> float:
    return get_float_env(
        "OTEL_INSTRUMENTATION_MEMPALACE_LLM_SLOW_THRESHOLD_S", 10.0
    )


def get_attr_max_bytes() -> int:
    return get_int_env(
        "OTEL_INSTRUMENTATION_MEMPALACE_ATTR_MAX_BYTES", 4096
    )


def get_user_id() -> str | None:
    return os.getenv("MEMPALACE_USER_ID")
