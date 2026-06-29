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

"""OpenTelemetry instrumentation for Microsoft Agent Framework.

This instrumentor enables MAF's built-in OTel telemetry (``enable_instrumentation``
with ``force=True`` so a sticky user-disable does not block us), bridges MAF's
native span helpers through ``opentelemetry-util-genai`` finish helpers, and
registers :class:`~.span_processor.MAFSemanticProcessor` for workflow/MCP
normalization plus metrics aggregation.

The optional ReAct step patch (``ARMS_MAF_REACT_STEP_ENABLED=true``) wraps the
``FunctionInvocationLayer.get_response`` ReAct loop with
``ExtendedTelemetryHandler.react_step()`` spans — direct ``start_as_current_span``
calls are not used, per the plugin's hard constraints.

Usage::

    from opentelemetry.instrumentation.microsoft_agent_framework import (
        MicrosoftAgentFrameworkInstrumentor,
    )
    MicrosoftAgentFrameworkInstrumentor().instrument()
"""

from __future__ import annotations

import logging
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer_provider

from .config import (
    get_slow_threshold_ms,
    is_instrumentation_enabled,
    is_metrics_enabled,
    is_react_step_enabled,
    is_sensitive_data_enabled,
)
from .package import _instruments
from .react_step_patch import apply_react_step_patch, revert_react_step_patch
from .span_processor import MAFSemanticProcessor
from .util_genai_bridge import (
    apply_util_genai_bridge,
    revert_util_genai_bridge,
)
from .version import __version__

__all__ = ["MicrosoftAgentFrameworkInstrumentor", "__version__"]

logger = logging.getLogger(__name__)


class MicrosoftAgentFrameworkInstrumentor(BaseInstrumentor):
    """Instrumentor for Microsoft Agent Framework (``agent-framework-core``)."""

    def __init__(self) -> None:
        super().__init__()
        self._processor: Optional[MAFSemanticProcessor] = None
        self._tracer_provider: Any = None
        self._react_applied: bool = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not is_instrumentation_enabled(default=True):
            logger.info(
                "Microsoft Agent Framework instrumentation disabled by env "
                "(ARMS_MAF_INSTRUMENTATION_ENABLED=false)."
            )
            return
        if self._processor is not None:
            return

        tracer_provider = (
            kwargs.get("tracer_provider") or get_tracer_provider()
        )
        meter_provider = kwargs.get("meter_provider")

        # 1) Enable MAF's built-in OTel instrumentation. ``force=True`` clears
        #    any sticky disable previously set by ``disable_instrumentation()``
        #    so our instrumentation always takes effect. ``enable_sensitive_data``
        #    is wired to the ARMS_MAF_SENSITIVE_DATA_ENABLED env (default False —
        #    PII/data redaction by default, per ARMS privacy guardrails).
        sensitive = bool(
            kwargs.get(
                "enable_sensitive_data",
                is_sensitive_data_enabled(default=False),
            )
        )
        try:
            from agent_framework.observability import enable_instrumentation

            enable_instrumentation(enable_sensitive_data=sensitive, force=True)
        except (ImportError, AttributeError, TypeError) as exc:
            logger.warning(
                "Could not enable MAF native instrumentation: %s. "
                "Spans will not be emitted by MAF; the SpanProcessor will "
                "still be registered but inert.",
                exc,
            )

        # 2) Bridge MAF's native span helper functions through util-genai's
        #    invocation finish helpers. This keeps MAF's span lifetime and
        #    streaming cleanup behavior, but writes AGENT/LLM/TOOL semantic
        #    attributes before span.end() creates the exporter snapshot.
        apply_util_genai_bridge()

        # 3) Register the semantic SpanProcessor. MAF uses the standard OTel
        #    TracerProvider (it does not have its own multi-processor), so
        #    ``add_span_processor`` is the right hook.
        processor = MAFSemanticProcessor(
            meter_provider=meter_provider,
            slow_threshold_ms=get_slow_threshold_ms(),
            metrics_enabled=is_metrics_enabled(default=True),
            capture_sensitive_data=sensitive,
        )
        try:
            tracer_provider.add_span_processor(processor)
        except Exception as exc:
            logger.warning("add_span_processor failed: %s", exc)
            raise

        # Ensure our processor runs FIRST in the pipeline so its ``on_end``
        # enrichments (gen_ai.span.kind / operation.name / framework / rename
        # map / provider normalization) are visible to any exporter processors
        # that were registered before us (e.g. by user bootstrap scripts that
        # add ``ConsoleSpanExporter`` / ``OTLPSpanExporter`` before
        # ``instrument()``). ``add_span_processor`` appends; processors are
        # invoked in registration order on ``on_end``, so we move ourselves
        # to index 0. The SDK stores the list as a tuple on
        # ``_active_span_processor._span_processors`` (defensive — falls back
        # to a list-style attribute on alternative provider layouts).
        try:
            asp = getattr(tracer_provider, "_active_span_processor", None)
            span_processors = (
                getattr(asp, "_span_processors", None)
                if asp is not None
                else None
            )
            if span_processors is None:
                span_processors = getattr(
                    tracer_provider, "_span_processors", None
                )
            # ``span_processors`` may be a tuple (current SDK) or a list.
            if isinstance(span_processors, tuple):
                others = tuple(
                    p for p in span_processors if p is not processor
                )
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
            logger.debug("prepend processor failed: %s", exc)

        self._processor = processor
        self._tracer_provider = tracer_provider

        # 4) Optional ReAct step patch (default OFF).
        react_enabled = bool(
            kwargs.get(
                "react_step_enabled", is_react_step_enabled(default=False)
            )
        )
        if react_enabled:
            try:
                apply_react_step_patch(tracer_provider)
                self._react_applied = True
            except (AttributeError, ImportError, TypeError) as exc:
                logger.warning("ReAct step patch skipped: %s", exc)

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._react_applied:
            revert_react_step_patch()
            self._react_applied = False
        revert_util_genai_bridge()
        if self._processor is not None:
            try:
                if self._tracer_provider is not None:
                    _remove_span_processor(
                        self._tracer_provider, self._processor
                    )
                self._processor.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("processor shutdown error: %s", exc)
            self._processor = None
            self._tracer_provider = None

        # We intentionally do NOT call ``disable_instrumentation()`` — that
        # would set MAF's sticky ``_user_disabled`` flag and prevent the user
        # from re-enabling later without ``force=True``. Respects the user's
        # own MAF observability state.


def _remove_span_processor(tracer_provider: Any, processor: Any) -> None:
    """Best-effort removal of the processor this instrumentor registered."""
    asp = getattr(tracer_provider, "_active_span_processor", None)
    span_processors = (
        getattr(asp, "_span_processors", None)
        if asp is not None
        else getattr(tracer_provider, "_span_processors", None)
    )
    if isinstance(span_processors, tuple):
        new_procs = tuple(p for p in span_processors if p is not processor)
        if asp is not None:
            asp._span_processors = new_procs  # type: ignore[attr-defined]
        else:
            tracer_provider._span_processors = new_procs  # type: ignore[attr-defined]
    elif isinstance(span_processors, list):
        span_processors[:] = [p for p in span_processors if p is not processor]
