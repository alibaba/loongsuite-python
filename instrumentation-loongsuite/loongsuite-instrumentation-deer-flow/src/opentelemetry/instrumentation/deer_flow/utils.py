# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for building util-genai invocations from DeerFlow objects."""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry.util.genai.utils import (
    get_content_capturing_mode,
    is_experimental_mode,
)
from opentelemetry.util.genai.types import ContentCapturingMode

logger = logging.getLogger(__name__)

DEER_FLOW_PROVIDER = "deer-flow"
DEER_FLOW_COMPONENT = "gen_ai.deer_flow.component"
DEER_FLOW_OPERATION = "gen_ai.deer_flow.operation"
DEER_FLOW_TASK_NAME = "gen_ai.deer_flow.task.name"

ENV_CAPTURE_MEMORY_CONTENT = "OTEL_DEER_FLOW_CAPTURE_MEMORY_CONTENT"


def _non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _should_capture_content() -> bool:
    if not is_experimental_mode():
        return False
    try:
        return get_content_capturing_mode() in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    except ValueError:
        return False


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _should_capture_memory_content() -> bool:
    return _bool_env(ENV_CAPTURE_MEMORY_CONTENT, False)


def _safe_call(action: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning(
            "DeerFlow instrumentation %s failed: %s",
            action,
            exc,
            exc_info=True,
        )
        return None


def _call_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
    default: Any = None,
) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _normalize_subagent_name(name: str | None) -> str:
    if not name:
        return "subagent"
    return name.strip().lower().replace("_", "-")


def _extract_human_content(graph_input: Any) -> str | None:
    """Best-effort extraction of the human message content from a graph input."""
    if graph_input is None:
        return None
    if isinstance(graph_input, str):
        return graph_input
    messages = None
    if isinstance(graph_input, dict):
        messages = graph_input.get("messages")
    else:
        messages = getattr(graph_input, "messages", None)
    if not messages:
        return None
    if isinstance(messages, list):
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if content and role in ("human", "user", None):
                return _non_empty_string(content)
    return None


def _token_values_from_usage_metadata(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    values: dict[str, int] = {}
    input_tokens = _int_value(
        getattr(usage, "input_tokens", None)
        or (usage.get("input_tokens") if isinstance(usage, dict) else None)
    )
    output_tokens = _int_value(
        getattr(usage, "output_tokens", None)
        or (usage.get("output_tokens") if isinstance(usage, dict) else None)
    )
    total_tokens = _int_value(
        getattr(usage, "total_tokens", None)
        or (usage.get("total_tokens") if isinstance(usage, dict) else None)
    )
    if input_tokens is not None:
        values["input_tokens"] = input_tokens
    if output_tokens is not None:
        values["output_tokens"] = output_tokens
    if total_tokens is not None:
        values["total_tokens"] = total_tokens
    return values


def _snapshot_token_records(collector: Any) -> dict[str, int]:
    """Pull usage records from a SubagentTokenCollector-like object."""
    if collector is None:
        return {}
    for method_name in ("snapshot_records", "snapshot", "get_usage"):
        method = getattr(collector, method_name, None)
        if method is None:
            continue
        try:
            records = method()
        except Exception:
            continue
        if not records:
            continue
        if isinstance(records, dict):
            usage = records
        elif isinstance(records, list) and records:
            usage = records[0] if isinstance(records[0], dict) else vars(records[0])
        else:
            usage = vars(records)
        values = _token_values_from_usage_metadata(usage)
        if values:
            return values
    return {}
