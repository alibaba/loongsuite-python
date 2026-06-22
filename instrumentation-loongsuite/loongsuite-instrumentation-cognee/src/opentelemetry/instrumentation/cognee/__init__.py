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

"""OpenTelemetry Cognee Instrumentation.

Implements gen-ai semantic-convention spans for the Cognee AI memory platform
(https://github.com/topoteretes/cognee).

Span coverage:
    - ENTRY    — wraps cognee.add/cognify/search/recall/remember
    - CHAIN    — Cognee-native ``cognee.api.*`` spans, normalized by
                 ``CogneeAttributeSpanProcessor``
    - TASK     — Cognee-native ``cognee.pipeline.task.*`` spans
    - RETRIEVER — Cognee-native ``cognee.retrieval.*`` spans
    - AGENT    — wraps ``AgenticRetriever._run_tool_loop``
    - STEP     — wraps ``generate_completion`` inside the ReAct loop
    - TOOL     — wraps ``cognee.modules.tools.execute_tool``
    - EMBEDDING — wraps non-LiteLLM ``EmbeddingEngine.embed_text`` adapters
    - LLM      — delegated to ``loongsuite-instrumentation-litellm``

LLM spans (including token usage metrics) are produced by
``loongsuite-instrumentation-litellm``; this instrumentor does not wrap
``litellm.acompletion`` / ``LLMGateway.acreate_structured_output``.
"""

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.cognee.internal._agent_wrapper import (
    install_agent_wrapper,
    uninstall_agent_wrapper,
)
from opentelemetry.instrumentation.cognee.internal._embedding_wrapper import (
    install_embedding_wrappers,
    uninstall_embedding_wrappers,
)
from opentelemetry.instrumentation.cognee.internal._entry_wrapper import (
    install_entry_wrappers,
    uninstall_entry_wrappers,
)
from opentelemetry.instrumentation.cognee.internal._llm_compat_wrapper import (
    install_llm_compat_wrapper,
    uninstall_llm_compat_wrapper,
)
from opentelemetry.instrumentation.cognee.internal._span_processor import (
    CogneeAttributeSpanProcessor,
)
from opentelemetry.instrumentation.cognee.internal._step_wrapper import (
    install_step_wrapper,
    uninstall_step_wrapper,
)
from opentelemetry.instrumentation.cognee.internal._tool_wrapper import (
    install_tool_wrapper,
    uninstall_tool_wrapper,
)
from opentelemetry.instrumentation.cognee.package import _instruments
from opentelemetry.instrumentation.cognee.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

__all__ = ["CogneeInstrumentor", "__version__"]


def _enable_cognee_tracing(tracer_provider: Any) -> None:
    """Best-effort enable Cognee's own tracing so its ``new_span`` calls produce spans.

    ``cognee.modules.observability.trace_context.enable_tracing`` flips on the
    module-level flag and calls ``setup_tracing()``. ``setup_tracing`` detects
    an externally-installed TracerProvider via ``_is_auto_instrumented`` and
    reuses it (so no provider conflict). If Cognee's OTEL extras are not
    installed we log a warning but keep the wrapt-based ENTRY/AGENT/TOOL/STEP/
    EMBEDDING spans alive.
    """
    try:
        import cognee.modules.observability.trace_context as cognee_tc  # type: ignore

        if cognee_tc.is_tracing_enabled():
            # Cognee already initialized — refresh its tracer reference so
            # subsequent new_span() calls land on the probe's TracerProvider.
            try:
                from cognee.modules.observability import (
                    tracing as cognee_tracing,  # type: ignore
                )

                cognee_tracing._tracer = tracer_provider.get_tracer(
                    "cognee", "1.2.1"
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(
                    "Cognee tracing already enabled but tracer refresh failed: %s",
                    e,
                )
        else:
            cognee_tc.enable_tracing()
    except Exception as e:
        logger.warning(
            "Failed to enable Cognee tracing: %s. Cognee-native spans "
            "(CHAIN/TASK/RETRIEVER) will be missing; probe-created spans "
            "(ENTRY/AGENT/TOOL/STEP/EMBEDDING) still work.",
            e,
        )


class CogneeInstrumentor(BaseInstrumentor):
    """Instrumentor for the Cognee AI memory platform."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if getattr(self, "_is_instrumented", False):
            return
        self._is_instrumented = True

        tracer_provider = kwargs.get("tracer_provider") or trace_api.get_tracer_provider()
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        telemetry_handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        self._telemetry_handler = telemetry_handler

        # Step 1: enable Cognee's own tracing so cognee.* spans get emitted.
        _enable_cognee_tracing(tracer_provider)

        # Step 2: attach SpanProcessor for cognee.* span normalization.
        # The processor rewrites span name/kind/operation in on_start and
        # wraps span.set_attribute so cognee.* attributes set during the span
        # body are mirrored to gen_ai.* on the same span.
        self._cognee_processor = CogneeAttributeSpanProcessor()
        if hasattr(tracer_provider, "add_span_processor"):
            try:
                tracer_provider.add_span_processor(self._cognee_processor)
            except Exception as e:
                logger.debug("add_span_processor(CogneeAttributeSpanProcessor) failed: %s", e)

        # Step 3: install wrappers (each in its own try/except — single failure
        # does not break the rest).
        install_entry_wrappers(telemetry_handler)
        install_agent_wrapper(telemetry_handler)
        install_step_wrapper(telemetry_handler)
        install_tool_wrapper(telemetry_handler)
        install_embedding_wrappers(telemetry_handler)

        # Step 3b: GenericAPIAdapter async-compat wrap — see
        # internal._llm_compat_wrapper for the root-cause analysis. Without
        # this, Cognee's `instructor.from_litellm(litellm.acompletion, ...)`
        # picks the sync retry path (because LiteLLMInstrumentor replaces
        # `litellm.acompletion` with a class instance that
        # `inspect.iscoroutinefunction` does not recognize as async), and
        # every LLM call raises InstructorRetryException. The wrap rebuilds
        # `self.aclient` as an explicit AsyncInstructor — purely a runtime
        # compat fix, no telemetry impact.
        install_llm_compat_wrapper()

        # Step 4: probe LiteLLM instrumentor availability — LLM spans depend on it.
        try:
            from opentelemetry.instrumentation.litellm import (
                LiteLLMInstrumentor,  # noqa: F401
            )
        except ImportError:
            logger.warning(
                "loongsuite-instrumentation-litellm not installed; "
                "LLM spans and token usage metrics will be missing."
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        if not getattr(self, "_is_instrumented", False):
            return
        uninstall_entry_wrappers()
        uninstall_agent_wrapper()
        uninstall_step_wrapper()
        uninstall_tool_wrapper()
        uninstall_embedding_wrappers()
        uninstall_llm_compat_wrapper()
        # Note: we cannot detach SpanProcessor from a TracerProvider that does
        # not expose a removal API. The processor becomes a no-op once
        # _is_instrumented flips back to False.
        self._is_instrumented = False
