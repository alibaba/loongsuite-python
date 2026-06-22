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

"""Attribute / span-name constants and helpers for Fara spans."""

from __future__ import annotations

import json
from typing import Any

from opentelemetry.instrumentation.fara.config import (
    FARA_OTEL_MAX_ATTR_LENGTH,
    FARA_OTEL_PROMPT_PREVIEW_MAX_LEN,
)

FRAMEWORK_NAME = "fara"

# Tool name -> human-readable description. Source: Fara
# ``_prompts.py`` FARA_ACTION_DEFINITIONS action space (matches
# ``FaraAgent.execute_action`` if-elif branches).
_TOOL_DESCRIPTIONS: dict[str, str] = {
    "key": "Press keyboard keys",
    "type": "Type text into a field",
    "mouse_move": "Move cursor to coordinates",
    "left_click": "Left click at coordinates",
    "scroll": "Scroll the page",
    "visit_url": "Navigate to a URL",
    "web_search": "Search the web",
    "history_back": "Go back in browser history",
    "pause_and_memorize_fact": "Memorize a fact",
    "wait": "Wait for a duration",
    "terminate": "End the task",
    # Fara aliases seen in execute_action if-elif branches
    "click": "Left click at coordinates",
    "hover": "Hover over coordinates",
    "keypress": "Press keyboard keys",
    "input_text": "Type text into a field",
    "sleep": "Wait for a duration",
    "stop": "End the task",
}


def tool_description(action_name: str) -> str | None:
    if not action_name:
        return None
    return _TOOL_DESCRIPTIONS.get(action_name)


# Canonical Fara action names (from FARA_ACTION_DEFINITIONS). Used for
# ``gen_ai.tool.definitions`` on the AGENT span. Per gen-ai semantic
# convention: default only ``type`` and ``name``; ``description`` is
# added only when capture-message-content is on.
_CANONICAL_TOOL_NAMES = (
    "key",
    "type",
    "mouse_move",
    "left_click",
    "scroll",
    "visit_url",
    "web_search",
    "history_back",
    "pause_and_memorize_fact",
    "wait",
    "terminate",
)

_TOOL_DEFINITIONS_NAME_ONLY = [
    {"type": "function", "name": name} for name in _CANONICAL_TOOL_NAMES
]

_TOOL_DEFINITIONS_FULL = [
    {
        "type": "function",
        "name": name,
        "description": _TOOL_DESCRIPTIONS[name],
    }
    for name in _CANONICAL_TOOL_NAMES
]


def tool_definitions(capture: bool) -> list[dict[str, Any]]:
    return list(_TOOL_DEFINITIONS_FULL if capture else _TOOL_DEFINITIONS_NAME_ONLY)


def truncate(value: str, max_len: int = FARA_OTEL_MAX_ATTR_LENGTH) -> str:
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
    return truncate(value, FARA_OTEL_PROMPT_PREVIEW_MAX_LEN)


def safe_json_dumps(value: Any, max_len: int | None = None) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = str(value)
    if max_len is None:
        return truncate(text)
    return truncate(text, max_len)
