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

"""Configuration via environment variables for AlgoTune instrumentation."""

from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"true", "1", "yes", "on"}


def _int_env(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


def _float_env(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _genai_capture_enabled() -> bool:
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


# Master enable switch
OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED = _bool_env(
    "OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED", True
)

# Whether to capture potentially sensitive content (tool args/results).
# LLM message content is controlled by the LiteLLM instrumentor itself.
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = _genai_capture_enabled()

# Maximum length of any string attribute the instrumentor produces.
ALGOTUNE_OTEL_MAX_CONTENT_LENGTH = _int_env(
    "ALGOTUNE_OTEL_MAX_CONTENT_LENGTH", "4096"
)

# Slow-call thresholds (seconds) used by the Span-to-Metrics processor.
ALGOTUNE_OTEL_SLOW_TOOL_SECONDS = _float_env(
    "ALGOTUNE_OTEL_SLOW_TOOL_SECONDS", "30"
)
ALGOTUNE_OTEL_SLOW_TASK_SECONDS = _float_env(
    "ALGOTUNE_OTEL_SLOW_TASK_SECONDS", "60"
)
ALGOTUNE_OTEL_SLOW_AGENT_SECONDS = _float_env(
    "ALGOTUNE_OTEL_SLOW_AGENT_SECONDS", "300"
)

# Whether to wrap TogetherModel.query() with a manual LLM span.
# TogetherModel hits the Together API directly via requests.post and is NOT
# covered by the LiteLLM instrumentor. Default off so the LiteLLM-only
# environments stay clean.
ALGOTUNE_OTEL_INSTRUMENT_TOGETHER = _bool_env(
    "ALGOTUNE_OTEL_INSTRUMENT_TOGETHER", False
)
