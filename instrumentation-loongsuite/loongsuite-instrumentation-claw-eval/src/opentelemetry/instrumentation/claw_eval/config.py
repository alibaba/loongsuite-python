"""Configuration via environment variables."""

from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"true", "1", "yes", "on"}


OTEL_INSTRUMENTATION_CLAW_EVAL_ENABLED = _bool_env(
    "OTEL_INSTRUMENTATION_CLAW_EVAL_ENABLED", True
)

OTEL_CLAW_EVAL_CAPTURE_CONTENT = _bool_env(
    "OTEL_CLAW_EVAL_CAPTURE_CONTENT", False
)

OTEL_CLAW_EVAL_PROPAGATE_TO_WORKER = _bool_env(
    "OTEL_CLAW_EVAL_PROPAGATE_TO_WORKER", False
)
