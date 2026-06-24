"""Optional ReAct step monkey patch for Microsoft Agent Framework.

Emits one ``react step`` span per LLM round-trip inside the
``FunctionInvocationLayer.get_response`` ReAct loop. Uses
``ExtendedTelemetryHandler.react_step()`` from ``opentelemetry.util.genai`` —
direct ``start_as_current_span`` calls are explicitly forbidden by the
plugin's hard constraints.

Patch strategy (minimal, robust):
- Wrap ``FunctionInvocationLayer.get_response`` to set a per-call ContextVar
  ``_maf_react_loop_active=True`` and reset the per-call step counter to 0.
- Wrap ``ChatTelemetryLayer.get_response`` (the LLM-call entry) so that when
  invoked inside a react-loop scope, the call is wrapped with
  ``handler.react_step(ReactStepInvocation(round=counter))``. Each LLM call
  inside the ReAct loop = one ReAct iteration = one ``react step`` span.
  The LLM span (already emitted by MAF's ``ChatTelemetryLayer``) becomes a
  child of the react_step span, matching the spec hierarchy
  ``AGENT > STEP > LLM``.

The patch is OFF by default (``ARMS_MAF_REACT_STEP_ENABLED=false``). It only
imports ``agent_framework`` internals when actually applied, so import-time
failure of a renamed MAF internal does not break the rest of instrumentation.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Per-call react-loop state. Set on entry to FunctionInvocationLayer.get_response
# so the ChatTelemetryLayer wrapper knows to open a react_step span around the
# LLM call. Stored as a token so we can reset cleanly on exit.
_maf_react_loop_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_maf_react_loop_active", default=False
)
_maf_react_step_counter: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_maf_react_step_counter", default=0
)

_applied = False
_original_fil_get_response: Any = None
_original_chat_get_response: Any = None
_handler: Any = None


def _get_extended_handler(tracer_provider: Any = None) -> Any:
    """Return the singleton ExtendedTelemetryHandler, creating it on first use."""
    global _handler
    if _handler is None:
        from opentelemetry.util.genai.extended_handler import (
            get_extended_telemetry_handler,
        )

        _handler = get_extended_telemetry_handler(tracer_provider=tracer_provider)
    return _handler


def apply_react_step_patch(tracer_provider: Any = None) -> None:
    """Apply the ReAct step patch. Idempotent; safe to call multiple times."""
    global _applied, _original_fil_get_response, _original_chat_get_response
    if _applied:
        return

    try:
        import wrapt  # type: ignore
        from agent_framework._tools import FunctionInvocationLayer  # type: ignore
        from agent_framework.observability import ChatTelemetryLayer  # type: ignore
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "ReAct step patch skipped: MAF internals not found (%s). "
            "This is expected if agent-framework version changed.",
            exc,
        )
        return

    handler = _get_extended_handler(tracer_provider)

    @wrapt.wrap_function_wrapper(FunctionInvocationLayer, "get_response")
    async def _fil_wrapper(wrapped, self, args, kwargs):  # type: ignore[no-untyped-def]
        # Outer function is async per MAF's signature (overloads collapse to
        # one async implementation). Set the react-loop scope for the duration
        # of the call.
        token_active = _maf_react_loop_active.set(True)
        token_counter = _maf_react_step_counter.set(0)
        try:
            return await wrapped(*args, **kwargs)
        finally:
            _maf_react_loop_active.reset(token_active)
            _maf_react_step_counter.reset(token_counter)

    @wrapt.wrap_function_wrapper(ChatTelemetryLayer, "get_response")
    async def _chat_wrapper(wrapped, self, args, kwargs):  # type: ignore[no-untyped-def]
        if not _maf_react_loop_active.get():
            return await wrapped(*args, **kwargs)

        # Each LLM call within the ReAct loop = one step.
        round_num = _maf_react_step_counter.get() + 1
        token_counter = _maf_react_step_counter.set(round_num)

        from opentelemetry.util.genai.extended_types import ReactStepInvocation

        step_inv = ReactStepInvocation(round=round_num)
        try:
            with handler.react_step(step_inv) as step:
                try:
                    result = await wrapped(*args, **kwargs)
                    # Best-effort finish_reason extraction from the response.
                    finish = _extract_finish_reason(result)
                    if finish is not None:
                        step.finish_reason = finish
                    return result
                except Exception as exc:
                    step.finish_reason = "error"
                    raise
        finally:
            _maf_react_step_counter.reset(token_counter)

    _original_fil_get_response = FunctionInvocationLayer.get_response
    _original_chat_get_response = ChatTelemetryLayer.get_response
    _applied = True
    logger.info("MAF ReAct step patch applied (handler.react_step).")


def _extract_finish_reason(result: Any) -> Any:
    """Best-effort extraction of a finish_reason string from a ChatResponse."""
    try:
        # MAF ChatResponse.messages[-1].finish_reason or choices[0].finish_reason
        messages = getattr(result, "messages", None)
        if messages:
            last = messages[-1]
            fr = getattr(last, "finish_reason", None)
            if fr:
                return fr
        choices = getattr(result, "choices", None)
        if choices:
            fr = getattr(choices[0], "finish_reason", None)
            if fr:
                return fr
    except Exception:
        return None
    return None


def revert_react_step_patch() -> None:
    """Revert the ReAct step patch. Safe to call even if not applied."""
    global _applied, _original_fil_get_response, _original_chat_get_response
    if not _applied:
        return
    try:
        from agent_framework._tools import FunctionInvocationLayer  # type: ignore
        from agent_framework.observability import ChatTelemetryLayer  # type: ignore

        if _original_fil_get_response is not None:
            try:
                FunctionInvocationLayer.get_response = _original_fil_get_response  # type: ignore[assignment]
            except Exception:
                pass
        if _original_chat_get_response is not None:
            try:
                ChatTelemetryLayer.get_response = _original_chat_get_response  # type: ignore[assignment]
            except Exception:
                pass
    except (ImportError, AttributeError):
        pass
    _applied = False
    _original_fil_get_response = None
    _original_chat_get_response = None
