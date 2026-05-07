"""Environment-variable driven configuration for the OpenHands instrumentation."""

from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"true", "1", "yes", "on"}


OTEL_INSTRUMENTATION_OPENHANDS_ENABLED = _bool_env(
    "OTEL_INSTRUMENTATION_OPENHANDS_ENABLED", True
)

OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS = _bool_env(
    "OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS", True
)

OTEL_INSTRUMENTATION_OPENHANDS_AUTO_INSTRUMENT_LITELLM = _bool_env(
    "OTEL_INSTRUMENTATION_OPENHANDS_AUTO_INSTRUMENT_LITELLM", True
)
