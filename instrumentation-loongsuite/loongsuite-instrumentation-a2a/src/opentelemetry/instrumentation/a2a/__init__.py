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

"""
OpenTelemetry A2A (Agent2Agent) Instrumentation

Produces an ARMS gen-ai **AGENT** span around each server-side agent turn of
the official A2A Python SDK (``a2a-sdk``). The span brackets the user's
``AgentExecutor.execute`` invocation, so all work the agent does -- including
the SDK's own transport / request-handler spans and any downstream LLM /
tool instrumentation -- nests underneath a single ``invoke_agent`` span with a
shared trace id.

Scope: agent execution, not the A2A protocol
--------------------------------------------
This package instruments the **agent execution boundary** only. It deliberately
does *not* try to model the A2A wire protocol (client/server method spans such
as ``SendMessage`` / ``GetTask``); that belongs in a dedicated protocol
instrumentation and can follow separately, tracking the A2A semantic
conventions under discussion in
https://github.com/open-telemetry/semantic-conventions-genai/pull/195.

Where that draft already names stable protocol context that is cheaply
available at the execution boundary (the task id and task state), we attach it
to the AGENT span using the draft's ``a2a.*`` keys, so the execution span can
be correlated with protocol telemetry without pretending to be a protocol span.

Relationship to a2a-sdk's built-in tracing
------------------------------------------
``a2a-sdk`` already ships an OpenTelemetry tracing layer
(``a2a.utils.telemetry``) that decorates its transports and request handlers
with generic spans under the instrumenting module ``a2a-python-sdk``. Those
spans describe the *protocol plumbing*; none of them carry gen-ai semantic
conventions and none of them wraps the user's ``execute`` implementation
(``AgentExecutor.execute`` is an abstract method the application overrides).
This package is therefore complementary, not duplicative.

Instrumentation seam
--------------------
``AgentExecutor`` is an ABC whose ``execute`` coroutine is overridden by
every concrete agent. To trace all of them we:

1. Walk the existing ``AgentExecutor`` subclass tree at ``instrument`` time
   and wrap each subclass's own ``execute`` (via ``wrapt``).
2. Install an ``__init_subclass__`` hook on ``AgentExecutor`` so that agent
   classes defined *after* instrumentation are wrapped as they are created.

Both paths mark the wrapped function with a sentinel so double-wrapping is
impossible, and ``uninstrument`` unwraps every marked ``execute`` and
restores the original ``__init_subclass__``.

Content capture
---------------
The user's input message is recorded as ``gen_ai.input.messages`` only when the
shared GenAI util's content-capture switch enables span content -- i.e. when
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is ``SPAN_ONLY`` or
``SPAN_AND_EVENT``. An absent or invalid value defaults to ``NO_CONTENT`` (no
message content), consistent with every other loongsuite instrumentation.
"""

import json
import logging
from typing import Any, Collection, Optional

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.a2a.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import get_content_capturing_mode

logger = logging.getLogger(__name__)

# -- Framework identifier -----------------------------------------------------
_FRAMEWORK = "a2a"

# -- GenAI semantic-convention attribute keys (sourced from the shared util) --
_GEN_AI_SPAN_KIND = GEN_AI_SPAN_KIND
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_FRAMEWORK = "gen_ai.framework"
_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"

_SPAN_KIND_AGENT = GenAiSpanKindValues.AGENT.value
_OP_INVOKE_AGENT = "invoke_agent"

# -- A2A protocol context keys (aligned with semantic-conventions-genai #195) --
# Only the stable task context available at the execution boundary; the full
# protocol attribute set (method.name, protocol.version, message.id, ...) is
# left to a dedicated protocol instrumentation.
_A2A_TASK_ID = "a2a.task.id"
_A2A_TASK_STATE = "a2a.task.state"
_A2A_CONTEXT_ID = "a2a.context.id"  # a2a-sdk RequestContext grouping id

# -- Sentinel -----------------------------------------------------------------
_A2A_MARKER = "_otel_a2a_wrapped"

# Content-capture modes under which message text may be written onto spans.
_CONTENT_ON_SPAN_MODES = frozenset(
    {ContentCapturingMode.SPAN_ONLY, ContentCapturingMode.SPAN_AND_EVENT}
)


def _capture_content() -> bool:
    """True when message content should be written onto spans.

    Delegated to the shared util so an absent/invalid
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` defaults to
    ``NO_CONTENT`` (no capture), matching the rest of loongsuite.
    """
    try:
        return get_content_capturing_mode() in _CONTENT_ON_SPAN_MODES
    except Exception:  # pragma: no cover - defensive: never break the app
        return False


def _text_message_json(role: str, content: Any) -> str:
    message = {
        "role": role,
        "parts": [{"type": "text", "content": str(content)}],
    }
    try:
        return json.dumps([message], ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str([message])


def _extract_user_input(context: Any) -> Optional[str]:
    """Best-effort extraction of the user's text input from a RequestContext."""
    if context is None:
        return None
    getter = getattr(context, "get_user_input", None)
    if callable(getter):
        try:
            text = getter()
            if text:
                return str(text)
        except Exception:
            return None
    return None


def _safe_set_attributes(span: Any, context: Any, agent_name: str) -> None:
    """Populate span attributes; telemetry failures must never break execution."""
    try:
        span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_AGENT)
        span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_INVOKE_AGENT)
        span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
        span.set_attribute(_GEN_AI_AGENT_NAME, agent_name)

        context_id = getattr(context, "context_id", None)
        if context_id:
            span.set_attribute(_A2A_CONTEXT_ID, str(context_id))
        task_id = getattr(context, "task_id", None)
        if task_id:
            span.set_attribute(_A2A_TASK_ID, str(task_id))
        # Task state, when the SDK exposes it on the current task.
        current_task = getattr(context, "current_task", None)
        task_state = getattr(
            getattr(current_task, "status", None), "state", None
        )
        state_value = getattr(task_state, "value", task_state)
        if state_value:
            span.set_attribute(_A2A_TASK_STATE, str(state_value))

        if _capture_content():
            user_input = _extract_user_input(context)
            if user_input:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_message_json("user", user_input),
                )
    except Exception:  # pragma: no cover - defensive: never break the app
        logger.debug(
            "A2A instrumentation failed to set span attributes", exc_info=True
        )


class _ExecuteWrapper:
    """Wrap ``AgentExecutor.execute`` to produce the AGENT span."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        context = args[0] if args else kwargs.get("context")
        agent_name = (
            type(instance).__name__ if instance is not None else _FRAMEWORK
        )

        with self._tracer.start_as_current_span(
            f"{_OP_INVOKE_AGENT} {agent_name}",
            kind=SpanKind.SERVER,
        ) as span:
            _safe_set_attributes(span, context, agent_name)

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                try:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                except Exception:  # pragma: no cover - defensive
                    pass
                raise

            try:
                span.set_status(Status(StatusCode.OK))
            except Exception:  # pragma: no cover - defensive
                pass
            return result


# ===========================================================================
# Wrap / unwrap helpers
# ===========================================================================


def _wrap_execute(cls, wrapper) -> None:
    """Wrap ``cls.execute`` exactly once (idempotent via sentinel)."""
    own = cls.__dict__.get("execute")
    if own is None:
        return  # abstract / not overridden on this class
    if getattr(own, _A2A_MARKER, False):
        return
    try:
        wrap_function_wrapper(cls, "execute", wrapper)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not wrap %s.execute: %s", cls.__name__, e)
        return
    new = cls.__dict__.get("execute")
    if new is not None:
        try:
            setattr(new, _A2A_MARKER, True)
        except Exception:  # pragma: no cover - defensive
            pass


def _unwrap_execute(cls) -> None:
    own = cls.__dict__.get("execute")
    if own is None or not getattr(own, _A2A_MARKER, False):
        return
    try:
        delattr(own, _A2A_MARKER)
    except (AttributeError, TypeError):
        pass
    try:
        unwrap(cls, "execute")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not unwrap %s.execute: %s", cls.__name__, e)


def _iter_subclasses(base):
    seen = set()
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        yield cls
        stack.extend(cls.__subclasses__())


# ===========================================================================
# Instrumentor
# ===========================================================================


class A2AInstrumentor(BaseInstrumentor):
    """Instrumentor for the official A2A Python SDK (``a2a-sdk``)."""

    def __init__(self):
        super().__init__()
        # BaseInstrumentor.__new__ returns a singleton, so __init__ may run
        # again on a later A2AInstrumentor() call. Only seed the bookkeeping
        # the first time, or a stray construct-after-instrument would clear the
        # active hook/wrapper state and make uninstrument() a no-op.
        if not hasattr(self, "_a2a_initialized"):
            self._a2a_initialized = True
            self._wrapper = None
            self._base = None
            self._saved_init_subclass = None
            self._had_own_init_subclass = False
            self._wrapped_classes = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from a2a.server.agent_execution import AgentExecutor

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__, "", tracer_provider=tracer_provider
        )
        wrapper = _ExecuteWrapper(tracer)
        self._wrapper = wrapper
        self._base = AgentExecutor
        self._wrapped_classes = []

        # 1) Wrap all existing subclasses.
        for cls in _iter_subclasses(AgentExecutor):
            _wrap_execute(cls, wrapper)
            self._wrapped_classes.append(cls)

        # 2) Hook future subclasses via __init_subclass__. Record whether
        #    AgentExecutor defined its own, so uninstrument can restore exactly.
        self._had_own_init_subclass = (
            "__init_subclass__" in AgentExecutor.__dict__
        )
        saved = AgentExecutor.__dict__.get("__init_subclass__")
        self._saved_init_subclass = saved

        def _new_init_subclass(cls, **kw):
            if saved is not None:
                # Delegate to AgentExecutor's own hook.
                saved.__func__(cls, **kw)
            else:
                # No own hook: cooperate with the rest of the MRO instead of
                # silently skipping other bases' __init_subclass__.
                super(AgentExecutor, cls).__init_subclass__(**kw)
            _wrap_execute(cls, wrapper)

        AgentExecutor.__init_subclass__ = classmethod(_new_init_subclass)

    def _uninstrument(self, **kwargs: Any) -> None:
        base = self._base
        if base is not None:
            # Unwrap everything we touched, plus any subclass carrying the
            # sentinel (covers classes created via the __init_subclass__ hook).
            classes = set(self._wrapped_classes)
            classes.update(_iter_subclasses(base))
            for cls in classes:
                _unwrap_execute(cls)

            # Restore __init_subclass__ to exactly its pre-instrument state.
            if self._had_own_init_subclass:
                base.__init_subclass__ = self._saved_init_subclass
            else:
                # We added the attribute; remove it so the inherited default
                # (object.__init_subclass__) is restored rather than a no-op.
                try:
                    delattr(base, "__init_subclass__")
                except (
                    AttributeError,
                    TypeError,
                ):  # pragma: no cover - defensive
                    logger.debug(
                        "Could not restore AgentExecutor.__init_subclass__"
                    )

        self._wrapper = None
        self._base = None
        self._saved_init_subclass = None
        self._had_own_init_subclass = False
        self._wrapped_classes = []
