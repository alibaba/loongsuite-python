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

"""Attribute / span-name constants and helpers for WebArena spans."""

from __future__ import annotations

import json
from typing import Any, Iterable

from opentelemetry.instrumentation.webarena.config import (
    WEBARENA_OTEL_MAX_ATTR_LENGTH,
    WEBARENA_OTEL_PROMPT_PREVIEW_MAX_LEN,
)

# --- vendor-extended attribute names -----------------------------------

GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
GEN_AI_REACT_ROUND = "gen_ai.react.round"
GEN_AI_REACT_FINISH_REASON = "gen_ai.react.finish_reason"

# WebArena-specific attribute names
WEBARENA_TASK_ID = "webarena.task.id"
WEBARENA_SITES = "webarena.sites"
WEBARENA_REQUIRE_LOGIN = "webarena.require_login"
WEBARENA_OBSERVATION_TYPE = "webarena.observation_type"
WEBARENA_ACTION_SET_TAG = "webarena.action_set_tag"
WEBARENA_ACTION_TYPE = "webarena.action.type"
WEBARENA_FAIL_ERROR = "webarena.fail_error"
WEBARENA_PAGE_URL_BEFORE = "webarena.page.url.before"
WEBARENA_PAGE_URL_AFTER = "webarena.page.url.after"
WEBARENA_BROWSER_ELEMENT_ID = "webarena.browser.element_id"
WEBARENA_OBSERVATION_MAIN_TYPE = "webarena.observation.main_type"
WEBARENA_STEP_COUNT = "webarena.step.count"
WEBARENA_TOOL_COUNT = "webarena.tool.count"
WEBARENA_PARSING_FAILURE_COUNT = "webarena.parsing_failure.count"
WEBARENA_PREVIOUS_ACTION = "webarena.previous_action"
WEBARENA_MEMORY_TRAJECTORY_LENGTH = "webarena.memory.trajectory_length"
WEBARENA_MEMORY_OBS_TEXT_LENGTH = "webarena.memory.obs_text_length"

FRAMEWORK_NAME = "webarena"


def truncate(value: str, max_len: int = WEBARENA_OTEL_MAX_ATTR_LENGTH) -> str:
    """Trim a string attribute to ``max_len`` characters with an ellipsis."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def truncate_content(value: str) -> str:
    """Trim a body / message-style attribute (longer cap than truncate())."""
    return truncate(value, WEBARENA_OTEL_PROMPT_PREVIEW_MAX_LEN)


def safe_json_dumps(value: Any, max_len: int | None = None) -> str:
    """JSON-encode ``value`` with best-effort fallback to ``str``."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = str(value)
    if max_len is None:
        return truncate(text)
    return truncate(text, max_len)


def action_type_name(action: Any) -> str:
    """Resolve an Action dict's ``action_type`` to its enum name."""
    if not isinstance(action, dict):
        return "UNKNOWN"
    raw = action.get("action_type")
    if raw is None:
        return "UNKNOWN"
    name = getattr(raw, "name", None)
    if name:
        return str(name)
    try:
        from browser_env.actions import ActionTypes  # noqa: PLC0415

        return ActionTypes(raw).name
    except Exception:  # noqa: BLE001
        return str(raw)


def action_arguments(action: Any) -> dict[str, Any]:
    """Extract a small JSON-friendly subset of an Action dict.

    We deliberately drop high-volume / binary-ish fields like
    ``coords``, ``raw_prediction`` and ``page_screenshot`` so the
    serialised value stays under the attribute length cap.
    """
    if not isinstance(action, dict):
        return {}
    keep_keys: Iterable[str] = (
        "element_id",
        "element_role",
        "element_name",
        "url",
        "text",
        "key_comb",
        "direction",
        "amount",
        "answer",
        "pw_code",
        "nth",
    )
    out: dict[str, Any] = {"action_type": action_type_name(action)}
    for k in keep_keys:
        v = action.get(k)
        if v in (None, "", [], {}):
            continue
        out[k] = v
    return out


def messages_to_input_value(messages: Any) -> str:
    """Compact representation of an LLM/agent prompt for ``input.value``."""
    if isinstance(messages, str):
        return truncate_content(messages)
    if isinstance(messages, list):
        try:
            return safe_json_dumps(
                messages, max_len=WEBARENA_OTEL_PROMPT_PREVIEW_MAX_LEN
            )
        except Exception:  # noqa: BLE001
            return truncate_content(str(messages))
    return truncate_content(str(messages))
