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

Produces an ARMS gen-ai ``AGENT`` span around each **server-side** agent
turn of the official A2A Python SDK (``a2a-sdk``), bracketing the user's
``AgentExecutor.execute`` invocation so all work the agent does -- including
the SDK's own transport / request-handler spans and any downstream LLM /
tool instrumentation -- nests underneath a single ``invoke_agent`` span with a
shared trace id.

Span ownership
---------------
The agent execution span is *owned* by the shared
``opentelemetry-util-genai`` ``ExtendedTelemetryHandler``. This instrumentation
only builds an ``InvokeAgentInvocation`` at the executor boundary and drives
it through ``start_invoke_agent`` / ``stop_invoke_agent`` /
``fail_invoke_agent``; it never starts, ends, records errors on, or sets
attributes on the AGENT span itself and adds no extra structural span, so
the handler's ``invoke_agent`` span is the single executor-boundary span
(matching the Hermes agent instrumentation).

Scope: agent execution, not the A2A protocol
--------------------------------------------
This package instruments the **agent execution boundary** only. It deliberately
does *not* model the A2A wire protocol (client/server method spans such as
``SendMessage`` / ``GetTask``); that belongs in a dedicated protocol
instrumentation and can follow separately, tracking the A2A semantic
conventions under discussion in
https://github.com/open-telemetry/semantic-conventions-genai/pull/195.

Per that draft, the A2A ``contextId`` (which groups a multi-turn agent
conversation) maps to the stable ``gen_ai.conversation.id`` attribute on the
AGENT span. The remaining task context available at the execution boundary
(task id and task state) is attached via the invocation attributes using the
draft's ``a2a.*`` keys so the execution span can be correlated with
protocol telemetry without pretending to be a protocol span.

Relationship to a2a-sdk's built-in tracing
------------------------------------------
``a2a-sdk`` already ships an OpenTelemetry tracing layer
(``a2a.utils.telemetry``) that decorates its transports and request
handlers with generic spans under the instrumenting module
``a2a-python-sdk``. Those spans describe the *protocol plumbing*; none of
them carry gen-ai semantic conventions and none wraps the user's
``execute`` implementation (``AgentExecutor.execute`` is an abstract method
the application overrides). This package is therefore complementary, not
duplicative.

Instrumentation seam
--------------------
``AgentExecutor`` is an ABC whose ``execute`` coroutine is overridden by
every concrete agent. To trace all of them we:

1. Walk the existing ``AgentExecutor`` subclass tree at ``instrument`` time
   and wrap each subclass's own ``execute`` (via ``wrapt``).
2. Install an ``__init_subclass__`` hook on ``AgentExecutor`` so that
   agent classes defined *after* instrumentation are wrapped as they are
   created.

Both paths mark the wrapped function with a sentinel so double-wrapping is
impossible, and ``uninstrument`` unwraps every marked ``execute`` and
restores the original ``__init_subclass__``.

Fail-safety
------------
Every telemetry step -- invocation construction,
handler start/stop/fail, attribute and error recording -- is wrapped so a
telemetry failure can never block the executor, alter its result, or replace
its business exception: on failure the original exception is re-raised
unchanged.

Content capture
---------------
The user's input message is handed to the shared GenAI util as
``input_messages``; whether the prompt text is exported is decided solely by
the util's content-capture switch
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` (span / event /
none), consistent with every other loongsuite instrumentation.
"""

import logging
from collections.abc import Collection
from typing import Any, Optional

from wrapt import wrap_function_wrapper

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.a2a.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_types import (
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import Error, InputMessage, Text

logger = logging.getLogger(__name__)

# -- Framework identifier -----------------------------------------------------
_FRAMEWORK = "a2a"

# -- A2A protocol context keys (aligned with semantic-conventions-genai #195) -
# Only the stable task context available at the execution boundary; the full
# protocol attribute set (method.name, protocol.version, message.id, ...) is
# left to a dedicated protocol instrumentation. The A2A ``contextId`` itself
# maps to the standard ``gen_ai.conversation.id`` (via
# ``InvokeAgentInvocation.conversation_id``), not to an a2a.* key.
_A2A_TASK_ID = "a2a.task.id"
_A2A_TASK_STATE = "a2a.task.state"

# -- Sentinel -----------------------------------------------------------------
_A2A_MARKER = "_otel_a2a_wrapped"


def _safe_get(obj: Any, name: str) -> Any:
    """Best-effort attribute access that never raises."""
    try:
        return getattr(obj, name, None)
    except Exception:  # pragma: no cover - defensive: never break the app
        logger.debug(
            "A2A instrumentation: reading %r failed", name, exc_info=True
        )
        return None


def _stringify_task_state(state: Any) -> Optional[str]:
    """Render an a2a task state as a short, stable string.

    ``a2a-sdk`` exposes the current task's state as a protobuf enum int;
    its symbolic name (e.g. ``TASK_STATE_SUBMITTED``) is obtained
    through the module-level ``TaskState.Name`` helper when available.
    Anything else is rendered defensively.
    """
    if state is None:
        return None
    try:
        # protobuf enum int -> symbolic name
        task_state = None
        try:
            from a2a.types.a2a_pb2 import TaskState  # noqa: PLC0415
        except (
            ImportError
        ):  # pragma: no cover - a2a-sdk always present at runtime
            task_state = None
        else:
            task_state = TaskState
        if task_state is not None:
            try:
                return str(task_state.Name(int(state)))
            except Exception:  # pragma: no cover - non-enum int value
                pass
        # Enums with their own .value/.name (other SDK versions/shapes).
        value = getattr(state, "value", state)
        if isinstance(value, int) and not isinstance(value, bool):
            return str(state)
        name = getattr(state, "name", None)
        return str(name if name is not None else value)
    except Exception:  # pragma: no cover - defensive: never break the app
        logger.debug("A2A instrumentation: task state render failed")
        return None


def _build_invocation(context: Any, agent_name: str) -> InvokeAgentInvocation:
    """Build the handler invocation from the server-side request context.

    Only cheaply available, stable server-side context is read. Every
    accessor is fail-safe: a hostile context must not stop the agent.
    """
    attributes: dict[str, Any] = {}

    context_id = _safe_get(context, "context_id")
    task_id = _safe_get(context, "task_id")
    if task_id:
        attributes[_A2A_TASK_ID] = str(task_id)

    current_task = _safe_get(context, "current_task")
    status = _safe_get(current_task, "status")
    state_value = _stringify_task_state(_safe_get(status, "state"))
    if state_value:
        attributes[_A2A_TASK_STATE] = state_value

    input_messages = []
    get_user_input = _safe_get(context, "get_user_input")
    if callable(get_user_input):
        try:
            user_text = get_user_input()
        except Exception:
            logger.debug(
                "A2A instrumentation: get_user_input failed",
                exc_info=True,
            )
            user_text = None
        if user_text:
            input_messages = [
                InputMessage(role="user", parts=[Text(content=str(user_text))])
            ]

    return InvokeAgentInvocation(
        provider=_FRAMEWORK,
        agent_name=agent_name,
        conversation_id=str(context_id) if context_id else None,
        input_messages=input_messages,
        attributes=attributes,
    )


class _ExecuteWrapper:
    """Wrap ``AgentExecutor.execute`` with handler-owned agent telemetry.

    The shared handler's ``invoke_agent`` AGENT span owns the full span
    lifecycle (matching the Hermes instrumentation); this wrapper adds no
    structural span of its own.
    """

    def __init__(self, handler):
        self._handler = handler

    async def __call__(self, wrapped, instance, args, kwargs):
        context = args[0] if args else kwargs.get("context")
        agent_name = (
            type(instance).__name__ if instance is not None else _FRAMEWORK
        )

        return await self._run(wrapped, args, kwargs, context, agent_name)

    async def _run(self, wrapped, args, kwargs, context, agent_name) -> Any:
        # Build the invocation (fail-safe) and let the shared
        # ExtendedTelemetryHandler own the whole AGENT span lifecycle, matching
        # the Hermes instrumentation (no extra structural span).
        invocation = None
        started = False
        try:
            invocation = _build_invocation(context, agent_name)
        except Exception:
            logger.debug(
                "A2A instrumentation: invocation build failed",
                exc_info=True,
            )
            # Keep handler telemetry alive even if context extraction
            # itself blows up.
            invocation = InvokeAgentInvocation(
                provider=_FRAMEWORK, agent_name=agent_name
            )
        try:
            self._handler.start_invoke_agent(
                invocation, context=otel_context.get_current()
            )
            started = True
        except Exception:
            logger.debug(
                "A2A instrumentation: start_invoke_agent failed",
                exc_info=True,
            )
            # The handler may have attached a span before the failure;
            # end it best-effort so nothing leaks.
            span = getattr(invocation, "span", None)
            if span is not None:
                try:
                    span.end()
                except Exception:  # pragma: no cover - defensive
                    logger.debug("A2A instrumentation: cleanup end failed")

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as business_error:
            self._fail(invocation, started, business_error)
            raise

        self._stop(invocation, started)
        return result

    def _stop(self, invocation, started: bool) -> None:
        if not started or invocation is None:
            return
        try:
            self._handler.stop_invoke_agent(invocation)
        except Exception:
            logger.debug(
                "A2A instrumentation: stop_invoke_agent failed",
                exc_info=True,
            )

    def _fail(self, invocation, started: bool, error: Exception) -> None:
        if not started or invocation is None:
            return
        try:
            self._handler.fail_invoke_agent(
                invocation, Error(message=str(error), type=type(error))
            )
        except Exception:
            logger.debug(
                "A2A instrumentation: fail_invoke_agent failed",
                exc_info=True,
            )


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
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not wrap %s.execute", cls.__name__, exc_info=True)
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
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "Could not unwrap %s.execute", cls.__name__, exc_info=True
        )


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
        # again on a later A2AInstrumentor() call. Only seed the
        # bookkeeping the first time, or a stray construct-after-instrument
        # would clear the active hook/wrapper state and make
        # uninstrument() a no-op.
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
        from a2a.server.agent_execution import (  # noqa: PLC0415
            AgentExecutor,
        )

        from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
            get_extended_telemetry_handler,
        )

        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")

        # The shared util owns the AGENT span: attributes, content capture,
        # events, metrics and fail-safe error recording.
        handler = get_extended_telemetry_handler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )
        wrapper = _ExecuteWrapper(handler)
        self._wrapper = wrapper
        self._base = AgentExecutor
        self._wrapped_classes = []

        # 1) Wrap all existing subclasses.
        for cls in _iter_subclasses(AgentExecutor):
            _wrap_execute(cls, wrapper)
            self._wrapped_classes.append(cls)

        # 2) Hook future subclasses via __init_subclass__. Record whether
        #    AgentExecutor defined its own, so uninstrument can restore it.
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
                # No own hook: cooperate with the rest of the MRO
                # instead of silently skipping other bases' hook.
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
