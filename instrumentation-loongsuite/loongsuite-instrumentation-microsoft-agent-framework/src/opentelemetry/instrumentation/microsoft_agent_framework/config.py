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

"""Environment configuration for Microsoft Agent Framework instrumentation."""

from __future__ import annotations

import os
from typing import Any


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return default


ENV_INSTRUMENTATION_ENABLED = "ARMS_MAF_INSTRUMENTATION_ENABLED"
ENV_SENSITIVE_DATA_ENABLED = "ARMS_MAF_SENSITIVE_DATA_ENABLED"
ENV_REACT_STEP_ENABLED = "ARMS_MAF_REACT_STEP_ENABLED"
ENV_SLOW_THRESHOLD_MS = "ARMS_MAF_SLOW_THRESHOLD_MS"
ENV_METRICS_ENABLED = "ARMS_MAF_METRICS_ENABLED"


def is_instrumentation_enabled(default: bool = True) -> bool:
    return _resolve_bool(
        os.getenv(ENV_INSTRUMENTATION_ENABLED), default=default
    )


def is_sensitive_data_enabled(default: bool = False) -> bool:
    return _resolve_bool(
        os.getenv(ENV_SENSITIVE_DATA_ENABLED), default=default
    )


def is_react_step_enabled(default: bool = False) -> bool:
    return _resolve_bool(os.getenv(ENV_REACT_STEP_ENABLED), default=default)


def is_metrics_enabled(default: bool = True) -> bool:
    return _resolve_bool(os.getenv(ENV_METRICS_ENABLED), default=default)


def get_slow_threshold_ms(default: int = 1000) -> int:
    raw = os.getenv(ENV_SLOW_THRESHOLD_MS)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default
