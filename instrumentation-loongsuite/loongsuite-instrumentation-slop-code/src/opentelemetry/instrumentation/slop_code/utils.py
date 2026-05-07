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

"""Utility functions for slop-code instrumentation."""

from typing import Any, Optional

from opentelemetry.trace import Span

SYSTEM_NAME = "slop-code"
MAX_ATTR_LEN = 1024


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute from an object, returning default on failure."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def safe_get_nested(obj: Any, *attrs: str, default: Any = None) -> Any:
    """Safely traverse nested attributes."""
    current = obj
    for attr in attrs:
        try:
            current = getattr(current, attr)
            if current is None:
                return default
        except (AttributeError, TypeError):
            return default
    return current


def set_optional_attr(span: Span, key: str, value: Optional[Any]) -> None:
    """Set a span attribute only if value is not None."""
    if value is not None:
        if isinstance(value, str) and len(value) > MAX_ATTR_LEN:
            value = value[:MAX_ATTR_LEN]
        span.set_attribute(key, value)
