# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import logging
from typing import Any, Collection

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from opentelemetry.instrumentation.pydantic_ai.capability import (
    LoongSuiteInstrumentationCapability,
)
from opentelemetry.instrumentation.pydantic_ai.package import _instruments
from opentelemetry.instrumentation.pydantic_ai.span_processor import (
    LoongSuiteSpanProcessor,
)
from opentelemetry.instrumentation.pydantic_ai.version import __version__

logger = logging.getLogger(__name__)

__all__ = [
    "LoongSuiteInstrumentationCapability",
    "LoongSuiteSpanProcessor",
    "PydanticAIInstrumentor",
    "__version__",
]


class PydanticAIInstrumentor(BaseInstrumentor):
    """Enable Pydantic AI built-in instrumentation with LoongSuite normalization."""

    def __init__(self) -> None:
        super().__init__()
        self._span_processor: LoongSuiteSpanProcessor | None = None
        self._previous_agent_instrument: Any = None
        self._previous_embedder_instrument: Any = None
        self._previous_auto_capability_types: Any = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider") or trace.get_tracer_provider()
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._register_span_processor(tracer_provider, meter_provider)
        settings = self._build_instrumentation_settings(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            kwargs=kwargs,
        )
        self._enable_pydantic_ai_defaults(settings)

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._span_processor is not None:
            self._span_processor.disable()
            self._span_processor = None
        self._restore_pydantic_ai_defaults()

    def _register_span_processor(
        self,
        tracer_provider: Any,
        meter_provider: Any,
    ) -> None:
        if not hasattr(tracer_provider, "add_span_processor"):
            logger.warning(
                "Current tracer provider does not support span processors; "
                "pydantic-ai span normalization skipped."
            )
            return
        self._span_processor = LoongSuiteSpanProcessor(
            meter_provider=meter_provider,
        )
        tracer_provider.add_span_processor(self._span_processor)

    def _build_instrumentation_settings(
        self,
        *,
        tracer_provider: Any,
        meter_provider: Any,
        logger_provider: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        from pydantic_ai.models.instrumented import InstrumentationSettings

        return InstrumentationSettings(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            include_binary_content=kwargs.get("include_binary_content", True),
            include_content=kwargs.get("include_content", False),
            version=kwargs.get("version", 5),
            event_mode=kwargs.get("event_mode", "attributes"),
            use_aggregated_usage_attribute_names=kwargs.get(
                "use_aggregated_usage_attribute_names",
                False,
            ),
        )

    def _enable_pydantic_ai_defaults(self, settings: Any) -> None:
        from pydantic_ai import Agent
        import pydantic_ai.agent as agent_module
        from pydantic_ai.embeddings import Embedder

        self._previous_agent_instrument = Agent._instrument_default
        self._previous_embedder_instrument = Embedder._instrument_default
        self._previous_auto_capability_types = getattr(
            agent_module,
            "_AUTO_INJECT_CAPABILITY_TYPES",
            None,
        )
        if self._previous_auto_capability_types is not None:
            auto_capability_types = self._previous_auto_capability_types
            if LoongSuiteInstrumentationCapability not in auto_capability_types:
                agent_module._AUTO_INJECT_CAPABILITY_TYPES = (
                    *auto_capability_types,
                    LoongSuiteInstrumentationCapability,
                )
        Agent.instrument_all(settings)
        Embedder.instrument_all(settings)

    def _restore_pydantic_ai_defaults(self) -> None:
        try:
            from pydantic_ai import Agent
            import pydantic_ai.agent as agent_module
            from pydantic_ai.embeddings import Embedder

            Agent.instrument_all(self._previous_agent_instrument or False)
            Embedder.instrument_all(self._previous_embedder_instrument or False)
            if self._previous_auto_capability_types is not None:
                agent_module._AUTO_INJECT_CAPABILITY_TYPES = (
                    self._previous_auto_capability_types
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to restore pydantic-ai instrumentation: %s", exc)
        finally:
            self._previous_auto_capability_types = None
