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

"""Small attribute / argument extraction helpers shared by the wrappers."""

from __future__ import annotations

import json
from typing import Any


def safe_str(value: Any) -> str:
    """Best-effort string conversion that never raises."""
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def preview(text: Any, max_len: int | None = None) -> str:
    """Return a string preview of *text* (kept for API compatibility).

    Truncation is no longer applied — captured content is emitted in
    full so dashboards never lose information. ``max_len`` is accepted
    but ignored.
    """
    return safe_str(text)


def maybe_preview(text: Any) -> str:
    """Alias for :func:`preview` — kept for API compatibility."""
    return preview(text)


def safe_get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first non-None attribute among *names* on *obj*."""
    for name in names:
        if obj is None:
            return default
        try:
            v = getattr(obj, name, None)
        except Exception:
            v = None
        if v is not None:
            return v
    return default


def serialize_message(message: Any) -> str:
    """Best-effort serialize an OpenHands message-like object to text."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    text_parts: list[str] = []
    for attr in ("text", "content", "value"):
        v = safe_get_attr(message, attr)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list):
            for item in v:
                t = safe_get_attr(item, "text", "content")
                if isinstance(t, str) and t:
                    text_parts.append(t)
    if text_parts:
        return "\n".join(text_parts)
    return safe_str(message)


def extract_uuid_str(value: Any) -> str:
    """Convert a UUID-like value to its hex/string form, returning ''."""
    if value is None:
        return ""
    hex_attr = getattr(value, "hex", None)
    if isinstance(hex_attr, str) and hex_attr:
        return hex_attr
    return safe_str(value)


# ---------------------------------------------------------------------------
# Semconv I/O serialization (input.value / output.value)
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any, depth: int = 0, max_depth: int = 8) -> Any:
    """Best-effort convert ``obj`` into something json.dumps can serialize.

    ``max_depth`` is generous enough to keep the GenAI message schema
    ``[{role, parts:[{type, ..., arguments:{...}}]}]`` fully expanded —
    falling through to ``safe_str`` mid-structure produces
    Python-repr-style single-quoted dict literals that aren't valid JSON.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if depth >= max_depth:
        return safe_str(obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            try:
                out[safe_str(k)] = _to_jsonable(v, depth + 1, max_depth)
            except Exception:
                out[safe_str(k)] = safe_str(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v, depth + 1, max_depth) for v in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump(), depth + 1, max_depth)
        except Exception:
            pass
    # Dataclass / generic object
    if hasattr(obj, "__dict__"):
        try:
            d = {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("_") and not callable(v)
            }
            if d:
                return _to_jsonable(d, depth + 1, max_depth)
        except Exception:
            pass
    return safe_str(obj)


def to_json_str(obj: Any, max_len: int | None = None) -> str:
    """Convert ``obj`` to a JSON string. Empty string on failure.

    No truncation is applied — captured content is emitted in full.
    ``max_len`` is accepted but ignored (kept for API compatibility).
    """
    try:
        jsonable = _to_jsonable(obj)
        s = json.dumps(jsonable, ensure_ascii=False, default=safe_str)
    except Exception:
        s = safe_str(obj)
    return s or ""


def maybe_to_json_str(obj: Any, max_len: int | None = None) -> str:
    """Alias for :func:`to_json_str` — kept for API compatibility."""
    return to_json_str(obj, max_len)


def messages_to_genai_input(messages: Any) -> str:
    """Serialize a chat-style ``messages`` list for ``gen_ai.input.messages``.

    Each item is normalized into ``{"role": ..., "content": ...}``. Keeps
    ``tool_calls`` when present.
    """
    if not isinstance(messages, list):
        return ""
    norm: list[dict[str, Any]] = []
    for m in messages:
        role = safe_get_attr(m, "role")
        content = safe_get_attr(m, "content")
        if role is None and content is None and isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        if isinstance(content, list):
            content = "".join(
                safe_str(
                    safe_get_attr(c, "text")
                    or safe_get_attr(c, "content")
                    or c
                )
                for c in content
            )
        item: dict[str, Any] = {
            "role": safe_str(role) or "user",
            "content": safe_str(content),
        }
        tool_calls = safe_get_attr(m, "tool_calls")
        if tool_calls:
            item["tool_calls"] = _to_jsonable(tool_calls)
        norm.append(item)
    return to_json_str(norm)


def action_to_genai_output(action: Any) -> str:
    """Serialize an OpenHands V0 ``Action`` into a GenAI-style assistant message."""
    if action is None:
        return ""
    action_type = safe_str(safe_get_attr(action, "action") or "")
    thought = safe_str(safe_get_attr(action, "thought") or "")
    item: dict[str, Any] = {"role": "assistant"}
    if thought:
        item["content"] = thought
    args: dict[str, Any] = {}
    for key in (
        "command",
        "code",
        "path",
        "url",
        "content",
        "task_list",
        "name",
        "arguments",
    ):
        v = safe_get_attr(action, key)
        if v not in (None, "", []):
            args[key] = _to_jsonable(v)
    if action_type or args:
        item["tool_calls"] = [
            {
                "type": "function",
                "function": {
                    "name": action_type or "agent.action",
                    "arguments": args,
                },
            }
        ]
    return to_json_str([item])
