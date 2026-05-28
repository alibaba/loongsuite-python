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
