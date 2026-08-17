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

"""Configuration switches for DSPy instrumentation.

Values are read from the environment on every access so that they can be
changed at runtime (and in tests) without re-instrumenting.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

OTEL_INSTRUMENTATION_DSPY_CAPTURE_ENTRY_SPAN = (
    "OTEL_INSTRUMENTATION_DSPY_CAPTURE_ENTRY_SPAN"
)
OTEL_INSTRUMENTATION_DSPY_CAPTURE_MODEL_NAME = (
    "OTEL_INSTRUMENTATION_DSPY_CAPTURE_MODEL_NAME"
)
OTEL_INSTRUMENTATION_DSPY_REACT_STEP_ENABLED = (
    "OTEL_INSTRUMENTATION_DSPY_REACT_STEP_ENABLED"
)
OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO = (
    "OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO"
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def entry_span_enabled() -> bool:
    """Whether an ENTRY span wraps the outermost DSPy module call."""
    return _bool_env(OTEL_INSTRUMENTATION_DSPY_CAPTURE_ENTRY_SPAN, True)


def model_name_enabled() -> bool:
    """Whether framework spans carry a best-effort ``gen_ai.request.model``."""
    return _bool_env(OTEL_INSTRUMENTATION_DSPY_CAPTURE_MODEL_NAME, True)


def react_step_enabled() -> bool:
    """Whether the ``ReAct`` STEP patch emits spans."""
    return _bool_env(OTEL_INSTRUMENTATION_DSPY_REACT_STEP_ENABLED, True)


def root_sample_ratio() -> float:
    """Fraction of outermost DSPy calls that produce framework spans.

    Optimizer compilation replays a program hundreds of times; lowering this
    ratio keeps the span volume bounded. The decision is taken once per
    outermost call and applies to its whole subtree, so a sampled-out program
    never emits a partial span tree.
    """
    raw = os.environ.get(OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO)
    if raw is None or not raw.strip():
        return 1.0
    try:
        ratio = float(raw)
    except ValueError:
        logger.debug(
            "Invalid %s: %r", OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO, raw
        )
        return 1.0
    return min(max(ratio, 0.0), 1.0)
