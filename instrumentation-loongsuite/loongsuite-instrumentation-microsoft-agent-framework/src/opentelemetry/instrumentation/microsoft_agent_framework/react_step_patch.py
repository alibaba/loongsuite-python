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
import inspect
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

        _handler = get_extended_telemetry_handler(
            tracer_provider=tracer_provider
        )
    return _handler


def apply_react_step_patch(tracer_provider: Any = None) -> None:
    """Apply the ReAct step patch. Idempotent; safe to call multiple times."""
    global _applied, _original_fil_get_response, _original_chat_get_response
    if _applied:
        return

    try:
        import wrapt  # type: ignore
        from agent_framework._tools import (
            FunctionInvocationLayer,  # type: ignore
        )
        from agent_framework.observability import (
            ChatTelemetryLayer,  # type: ignore
        )
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "ReAct step patch skipped: MAF internals not found (%s). "
            "This is expected if agent-framework version changed.",
            exc,
        )
        return

    # Capture the unwrapped originals BEFORE installing the wrapt wrappers.
    # ``wrap_function_wrapper`` replaces the class attribute in place, so
    # capturing afterwards (as the previous version did) records the wrapper
    # itself, making the legacy ``revert`` a no-op and stacking wrappers on
    # apply→revert→apply. We also unwrap any pre-existing wrappers down to
    # the underlying function so we only restore the layer we installed.
    _original_fil_get_response = _unwrap_to_function(
        FunctionInvocationLayer.get_response
    )
    _original_chat_get_response = _unwrap_to_function(
        ChatTelemetryLayer.get_response
    )

    handler = _get_extended_handler(tracer_provider)

    # These MAF methods are synchronous dispatchers: stream=False returns an
    # awaitable while stream=True returns a ResponseStream. The wrapper must
    # preserve that public contract, so it only wraps awaitables and passes
    # streaming ResponseStream values through unchanged.
    def _fil_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        result = wrapped(*args, **kwargs)
        if _is_response_stream(result) or not inspect.isawaitable(result):
            return result

        async def _scoped():  # type: ignore[no-untyped-def]
            token_active = _maf_react_loop_active.set(True)
            token_counter = _maf_react_step_counter.set(0)
            try:
                return await result
            finally:
                _maf_react_loop_active.reset(token_active)
                _maf_react_step_counter.reset(token_counter)

        return _scoped()

    def _chat_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        result = wrapped(*args, **kwargs)

        # Read the loop-active flag synchronously. When False we pass the
        # wrapped result through unchanged. When True, only awaitables are
        # wrapped; streaming ResponseStream values keep their original type.
        if not _maf_react_loop_active.get():
            return result
        if _is_response_stream(result) or not inspect.isawaitable(result):
            return result

        # Pre-compute the round number for this call (read before the
        # coroutine body so subsequent chat calls in the same loop see the
        # incremented value at the same context level — same semantics as the
        # prior implementation).
        round_num = _maf_react_step_counter.get() + 1
        local_handler = handler

        async def _step_scoped():  # type: ignore[no-untyped-def]
            from opentelemetry.util.genai.extended_types import (
                ReactStepInvocation,
            )

            step_inv = ReactStepInvocation(round=round_num)
            token_counter = _maf_react_step_counter.set(round_num)
            try:
                with local_handler.react_step(step_inv) as step:
                    try:
                        response = await result
                        # Best-effort finish_reason extraction from the response.
                        finish = _extract_finish_reason(response)
                        if finish is not None:
                            step.finish_reason = finish
                        return response
                    except Exception:
                        step.finish_reason = "error"
                        raise
            finally:
                _maf_react_step_counter.reset(token_counter)

        return _step_scoped()

    # ``wrap_function_wrapper`` takes (module_or_obj, name, wrapper). When
    # given a class object + attribute name it patches the attribute on the
    # class. Each call installs exactly one FunctionWrapper around the
    # currently-bound attribute, so apply→revert→apply does not stack
    # provided revert restores the original first.
    wrapt.wrap_function_wrapper(
        FunctionInvocationLayer, "get_response", _fil_wrapper
    )
    wrapt.wrap_function_wrapper(
        ChatTelemetryLayer, "get_response", _chat_wrapper
    )

    _applied = True
    logger.info("MAF ReAct step patch applied (handler.react_step).")


def _unwrap_to_function(func: Any) -> Any:
    """Return the underlying function, peeling any wrapt wrappers.

    ``wrapt.FunctionWrapper`` exposes the wrapped callable via ``__wrapped__``.
    We walk that chain so that, when an unrelated instrumentor has already
    wrapped the target, we restore only the layer we installed (not the
    unrelated one — that is a separate concern). When the target is not a
    wrapper, ``__wrapped__`` is absent and we return ``func`` unchanged.
    """
    seen: set[int] = set()
    cur = func
    while True:
        if id(cur) in seen:  # defensive against cycles
            break
        seen.add(id(cur))
        nxt = getattr(cur, "__wrapped__", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _is_response_stream(result: Any) -> bool:
    """Return True for MAF ResponseStream-like values.

    ``ResponseStream`` is awaitable as a convenience, so ``inspect.isawaitable``
    alone cannot distinguish it from the non-streaming coroutine path.
    """
    return (
        hasattr(result, "map")
        and hasattr(result, "__aiter__")
        and hasattr(result, "get_final_response")
    )


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
        from agent_framework._tools import (
            FunctionInvocationLayer,  # type: ignore
        )
        from agent_framework.observability import (
            ChatTelemetryLayer,  # type: ignore
        )

        if _original_fil_get_response is not None:
            try:
                FunctionInvocationLayer.get_response = (
                    _original_fil_get_response  # type: ignore[assignment]
                )
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
