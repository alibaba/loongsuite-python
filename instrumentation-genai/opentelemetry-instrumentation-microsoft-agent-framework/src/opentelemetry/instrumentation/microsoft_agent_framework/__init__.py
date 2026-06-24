"""OpenTelemetry instrumentation for Microsoft Agent Framework.

This instrumentor enables MAF's built-in OTel telemetry (``enable_instrumentation``
with ``force=True`` so a sticky user-disable does not block us) and registers a
:class:`~.span_processor.MAFSemanticProcessor` that enriches MAF's native
spans to align with the ARMS GenAI semantic conventions.

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
from .version import __version__

__all__ = ["MicrosoftAgentFrameworkInstrumentor", "__version__"]

logger = logging.getLogger(__name__)


class MicrosoftAgentFrameworkInstrumentor(BaseInstrumentor):
    """Instrumentor for Microsoft Agent Framework (``agent-framework-core``)."""

    def __init__(self) -> None:
        super().__init__()
        self._processor: Optional[MAFSemanticProcessor] = None
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

        tracer_provider = kwargs.get("tracer_provider") or get_tracer_provider()
        meter_provider = kwargs.get("meter_provider")

        # 1) Enable MAF's built-in OTel instrumentation. ``force=True`` clears
        #    any sticky disable previously set by ``disable_instrumentation()``
        #    so our instrumentation always takes effect. ``enable_sensitive_data``
        #    is wired to the ARMS_MAF_SENSITIVE_DATA_ENABLED env (default False —
        #    PII/data redaction by default, per ARMS privacy guardrails).
        sensitive = bool(
            kwargs.get(
                "enable_sensitive_data", is_sensitive_data_enabled(default=False)
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

        # 2) Register the semantic SpanProcessor. MAF uses the standard OTel
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
        self._processor = processor

        # 3) Optional ReAct step patch (default OFF).
        react_enabled = bool(
            kwargs.get("react_step_enabled", is_react_step_enabled(default=False))
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
        if self._processor is not None:
            try:
                self._processor.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("processor shutdown error: %s", exc)
            self._processor = None

        # We intentionally do NOT call ``disable_instrumentation()`` — that
        # would set MAF's sticky ``_user_disabled`` flag and prevent the user
        # from re-enabling later without ``force=True``. Respects the user's
        # own MAF observability state.
