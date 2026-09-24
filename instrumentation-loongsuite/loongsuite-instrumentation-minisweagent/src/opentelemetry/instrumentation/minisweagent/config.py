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
