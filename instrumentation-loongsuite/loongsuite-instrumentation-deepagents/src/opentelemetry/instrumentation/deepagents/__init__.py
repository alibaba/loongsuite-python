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

"""LoongSuite instrumentation for langchain-ai deepagents."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import metrics, trace
from opentelemetry.instrumentation.deepagents.internal._enricher import (
    install_enricher_callback,
    uninstall_enricher_callback,
)
from opentelemetry.instrumentation.deepagents.internal._entry_patch import (
    instrument_entry_patch,
    uninstrument_entry_patch,
)
from opentelemetry.instrumentation.deepagents.internal._metrics_processor import (  # noqa: E501
    install_metrics_processor,
    shutdown_metrics_processors,
)
from opentelemetry.instrumentation.deepagents.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

__all__ = ["DeepAgentsInstrumentor"]

_logger = logging.getLogger(__name__)


def _instrument_dependency(
    module_name: str,
    class_name: str,
    **kwargs: Any,
) -> None:
    """Instrument a required base package when it is installed."""
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or (
            exc.name is not None and module_name.startswith(f"{exc.name}.")
        ):
            _logger.warning(
                "deepagents instrumentation requires %s; continuing with "
                "ENTRY/metrics only.",
                module_name,
            )
            return
        raise

    instrumentor_type = getattr(module, class_name, None)
    if instrumentor_type is None:
        _logger.warning(
            "deepagents instrumentation could not find %s.%s",
            module_name,
            class_name,
        )
        return

    instrumentor = instrumentor_type()
    if instrumentor.is_instrumented_by_opentelemetry:
        return
    instrumentor.instrument(**kwargs)


class DeepAgentsInstrumentor(BaseInstrumentor):
    """Instrumentation for deepagents.

    The plugin is intentionally additive: LangChain and LangGraph keep owning
    AGENT/CHAIN/STEP/LLM/TOOL spans, while this plugin contributes the ENTRY
    wrapper, a sidecar metadata enricher, and GenAI metrics from finished spans.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        _instrument_dependency(
            "opentelemetry.instrumentation.langchain",
            "LangChainInstrumentor",
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        _instrument_dependency(
            "opentelemetry.instrumentation.langgraph",
            "LangGraphInstrumentor",
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        instrument_entry_patch(handler)
        install_enricher_callback()

        if tracer_provider is None:
            tracer_provider = trace.get_tracer_provider()
        if meter_provider is None:
            _logger.warning(
                "deepagents instrumentation meter_provider was not supplied; "
                "using the global MeterProvider for GenAI metrics."
            )
            meter_provider = metrics.get_meter_provider()
        install_metrics_processor(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        uninstrument_entry_patch()
        uninstall_enricher_callback()
        shutdown_metrics_processors()
