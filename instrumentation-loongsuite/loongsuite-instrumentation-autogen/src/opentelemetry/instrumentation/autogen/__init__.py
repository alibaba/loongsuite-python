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

"""OpenTelemetry instrumentation for Microsoft AutoGen AgentChat."""

from __future__ import annotations

import logging
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer_provider
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from .config import (
    is_instrumentation_enabled,
    is_native_span_processor_enabled,
)
from .package import _instruments
from .patch import apply_agentchat_patch, revert_agentchat_patch
from .span_processor import AutoGenSemanticProcessor
from .version import __version__

__all__ = ["AutoGenInstrumentor", "__version__"]

logger = logging.getLogger(__name__)


class AutoGenInstrumentor(BaseInstrumentor):
    """Instrument Microsoft AutoGen 0.7.x AgentChat flows."""

    def __init__(self) -> None:
        super().__init__()
        self._processor: Optional[AutoGenSemanticProcessor] = None
        self._handler: Optional[ExtendedTelemetryHandler] = None
        self._tracer_provider: Any = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not is_instrumentation_enabled(default=True):
            logger.info("AutoGen instrumentation disabled by env.")
            return
        if self._handler is not None:
            return

        tracer_provider = (
            kwargs.get("tracer_provider") or get_tracer_provider()
        )
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        self._handler = handler
        self._tracer_provider = tracer_provider

        if is_native_span_processor_enabled(default=True):
            processor = AutoGenSemanticProcessor()
            tracer_provider.add_span_processor(processor)
            _prepend_span_processor(tracer_provider, processor)
            self._processor = processor

        apply_agentchat_patch(handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        revert_agentchat_patch()
        if self._processor is not None:
            try:
                if self._tracer_provider is not None:
                    _remove_span_processor(
                        self._tracer_provider, self._processor
                    )
                self._processor.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("AutoGen processor shutdown failed: %s", exc)
            self._processor = None
        self._handler = None
        self._tracer_provider = None


def _span_processors(tracer_provider: Any) -> tuple[Any, Any]:
    asp = getattr(tracer_provider, "_active_span_processor", None)
    span_processors = (
        getattr(asp, "_span_processors", None)
        if asp is not None
        else getattr(tracer_provider, "_span_processors", None)
    )
    return asp, span_processors


def _prepend_span_processor(tracer_provider: Any, processor: Any) -> None:
    asp, span_processors = _span_processors(tracer_provider)
    try:
        if isinstance(span_processors, tuple):
            others = tuple(p for p in span_processors if p is not processor)
            new_procs = (processor,) + others
            if asp is not None:
                asp._span_processors = new_procs  # type: ignore[attr-defined]
            else:
                tracer_provider._span_processors = new_procs  # type: ignore[attr-defined]
        elif isinstance(span_processors, list):
            if processor in span_processors:
                span_processors.remove(processor)
            span_processors.insert(0, processor)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("AutoGen processor prepend failed: %s", exc)


def _remove_span_processor(tracer_provider: Any, processor: Any) -> None:
    asp, span_processors = _span_processors(tracer_provider)
    if isinstance(span_processors, tuple):
        new_procs = tuple(p for p in span_processors if p is not processor)
        if asp is not None:
            asp._span_processors = new_procs  # type: ignore[attr-defined]
        else:
            tracer_provider._span_processors = new_procs  # type: ignore[attr-defined]
    elif isinstance(span_processors, list):
        span_processors[:] = [p for p in span_processors if p is not processor]
