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

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def is_instrumentation_enabled(default: bool = True) -> bool:
    return _env_bool("ARMS_AUTOGEN_INSTRUMENTATION_ENABLED", default)


def is_agent_span_enabled(default: bool = True) -> bool:
    return _env_bool("ARMS_AUTOGEN_AGENT_SPAN_ENABLED", default)


def is_llm_span_enabled(default: bool = True) -> bool:
    return _env_bool("ARMS_AUTOGEN_LLM_SPAN_ENABLED", default)


def is_native_span_processor_enabled(default: bool = True) -> bool:
    return _env_bool("ARMS_AUTOGEN_NATIVE_SPAN_PROCESSOR_ENABLED", default)
