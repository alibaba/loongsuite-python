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

"""Helpers shared by Agent-Reach wrappers."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _safe_json_dumps(value: Any, *, max_chars: Optional[int] = None) -> Optional[str]:
    """Best-effort JSON serialization that never raises.

    Returns ``None`` when the value cannot be serialized. When ``max_chars``
    is set and the serialized form exceeds it, the string is truncated and a
    marker is appended so downstream consumers can tell truncation happened.
    """
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "...(truncated)"
    return text


def _is_content_capture_enabled() -> bool:
    """Return True when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
    requests content to be written to span attributes.

    The util-genai flag accepts ``NO_CONTENT`` / ``SPAN_ONLY`` /
    ``EVENT_ONLY`` / ``SPAN_AND_EVENT``. Content is captured unless the value
    is explicitly ``NO_CONTENT`` or empty.
    """
    value = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", ""
    ).strip().upper()
    return value not in ("", "NO_CONTENT", "FALSE", "0")


def _truncate(value: Optional[str], max_chars: int) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...(truncated)"


def _get_attr(instance: Any, name: str, default: Any = None) -> Any:
    """Safe attribute access that swallows exceptions from user objects."""
    try:
        return getattr(instance, name, default)
    except Exception:  # noqa: BLE001
        return default
