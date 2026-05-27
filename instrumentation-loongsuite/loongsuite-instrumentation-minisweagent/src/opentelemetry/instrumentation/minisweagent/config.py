"""Configuration via environment variables."""

from __future__ import annotations

import contextvars
import os


def _int_env(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN = _int_env(
    "OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN", "256"
)
OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN = _int_env(
    "OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN", "256"
)

ENTRY_SPAN_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_minisweagent_entry_active", default=False
)
