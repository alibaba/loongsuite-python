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

"""Configuration via environment variables."""

from __future__ import annotations

import os


def _int_env(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Cap on non-content string attribute values (URLs, tool names, etc.)
WEBARENA_OTEL_MAX_ATTR_LENGTH = _int_env(
    "WEBARENA_OTEL_MAX_ATTR_LENGTH", "1024"
)

# Cap on prompt / message preview length when capture-message-content is on
WEBARENA_OTEL_PROMPT_PREVIEW_MAX_LEN = _int_env(
    "WEBARENA_OTEL_PROMPT_PREVIEW_MAX_LEN", "4096"
)


def capture_message_content() -> bool:
    """Whether to record prompt / completion / tool argument bodies.

    Honours the standard semantic-conventions opt-in flag.
    Accepts SPAN_ONLY / SPAN_AND_EVENT / EVENT_ONLY as truthy values.
    """
    val = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    if val is None:
        return False
    return val.strip().upper() in {
        "TRUE",
        "1",
        "YES",
        "ON",
        "SPAN_ONLY",
        "SPAN_AND_EVENT",
        "EVENT_ONLY",
    }
