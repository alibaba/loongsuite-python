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

"""Wrapper classes for AgentScope instrumentation.

Concurrency
-----------
The instrumentation supports concurrent ``await agent(...)`` calls on the
**same** ``AgentBase`` instance (e.g. TUI multi-tab, ``asyncio.gather``,
ASGI singletons). Per-call telemetry state is isolated through Python
``contextvars`` so that each asyncio task / thread observes its own
``_ReactStepState``; sibling concurrent invocations on the same agent never
clobber each other's react-step span, round counter or pending acting count.

Two pieces of shared state are kept off the instance:

1. ``_REACT_STATE`` (``ContextVar[_ReactStepState | None]``) holds the
   per-call state. ReAct hooks read it via ``_REACT_STATE.get()`` and
   no-op when the slot is ``None`` (i.e. the hook fired in a context that
   did not opt into instrumentation). The state is never written to
   ``agent`` attributes.

2. ``_REACT_HOOK_REGISTRY`` is a process-global
   ``WeakKeyDictionary[agent -> ref count]`` map guarded by
   ``_REACT_HOOK_REGISTRY_LOCK``.  Using agent objects as keys (rather
   than ``id(agent)``) avoids stale entries when CPython recycles memory
   addresses; weak references let the entry be collected automatically if
   the agent is garbage-collected without a paired release.  The first
   concurrent call on an agent registers four hooks under the fixed name
   ``_REACT_HOOK_NAME``; subsequent concurrent calls only bump the ref
   count. Hooks are removed only after the **last** outstanding call on
   that agent unwinds. This is required because AgentScope's hook
   registry is per-instance, and we must avoid both (a) registering the
   same hook twice (which would double-fire ``_reasoning`` callbacks) and
   (b) removing hooks while a sibling concurrent call still depends on
   them.

Stacked ``ChatModelBase`` / ``AgentBase`` implementations (e.g. proxies where
each layer subclasses the base and ``__call__`` forwards to an inner model or
agent) share one logical invocation. A ``contextvars`` depth counter ensures
only the outermost ``__call__`` emits LLM / ``invoke_agent`` spans; inner
layers call through without duplicating telemetry.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import timeit
import weakref
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, AsyncGenerator, MutableMapping

from opentelemetry.context import get_current as _get_current_context
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import ReactStepInvocation
from opentelemetry.util.genai.types import Error, LLMInvocation

from .utils import (
    apply_entry_baggage_identity,
    convert_agent_response_to_output_messages,
    convert_chatresponse_to_output_messages,
    create_agent_invocation,
    create_embedding_invocation,
    create_llm_invocation,
)

logger = logging.getLogger(__name__)

# Fixed hook name shared by all concurrent invocations on a given agent.
# Using a stable name (rather than a per-call uuid) lets AgentScope's own
# registry naturally deduplicate, so a single set of callbacks fires for
# each ``_reasoning`` / ``_acting`` regardless of how many concurrent
# ``__call__`` invocations are in flight on the same instance.
_REACT_HOOK_NAME = "otel_react_step"

# Per-async-task nesting for stacked __call__ (proxy / decorator chains).
_CHAT_MODEL_CALL_DEPTH = contextvars.ContextVar(
    "opentelemetry_agentscope_chat_model_call_depth",
    default=0,
)
_AGENT_CALL_DEPTH = contextvars.ContextVar(
    "opentelemetry_agentscope_agent_call_depth",
    default=0,
)

# Per-call ReAct step state. Stored in a ContextVar (not on the agent
# instance) so concurrent ``await agent(...)`` calls on the same instance
# observe isolated state. Hooks read via ``_REACT_STATE.get()`` and no-op
# when unset (i.e. fired in a context that did not opt into instrumentation).
_REACT_STATE: contextvars.ContextVar["_ReactStepState | None"] = (
    contextvars.ContextVar(
        "opentelemetry_agentscope_react_state",
        default=None,
    )
)

# Per-agent ``outstanding __call__`` counter for ReAct hook lifetime
# management. Hooks are registered exactly once on the first concurrent
# call and removed only when the last outstanding call unwinds; this
# avoids both double-firing (multiple registrations under the same name)
# and premature removal while sibling calls still depend on the hooks.
#
# A ``WeakKeyDictionary`` is used (rather than ``id(agent) -> int``) so
# that an agent that is garbage-collected without a paired ``release``
# (e.g. wrapper crashed between ``acquire`` and the protected region)
# automatically drops its registry entry, preventing a different agent
# from later being mis-recognised as already instrumented just because
# CPython recycled its memory address.
_REACT_HOOK_REGISTRY_LOCK = threading.Lock()
_REACT_HOOK_REGISTRY: MutableMapping[Any, int] = weakref.WeakKeyDictionary()


def _is_react_agent(agent_instance: Any) -> bool:
    """Check if an agent instance is a ReAct agent by duck-typing."""
    return hasattr(agent_instance, "_instance_pre_reasoning_hooks")


@dataclass
class _ReactStepState:
    """Per-call state for React step span lifecycle.

    Lives in the ``_REACT_STATE`` ``ContextVar`` rather than on the agent
    instance, so concurrent ``__call__`` invocations on the same agent
    each see their own state and never overwrite each other's
    ``active_step`` / round counter / pending acting count.

    ``owner`` is the agent instance that created this state.  Every hook
    checks ``state.owner is agent_self`` before mutating state, so a child
    agent's hooks cannot accidentally read or modify a parent agent's state
    when both share the same asyncio execution context.
    """

    owner: Any = field(default=None)
    react_round: int = 0
    active_step: ReactStepInvocation | None = None
    original_context: Any = field(default=None)
    pending_acting_count: int = 0
    # Subclasses that override _reasoning / _acting and call super() stack multiple
    # AgentScope hook wrappers; only the outermost wrapper should drive spans.
    reasoning_nesting: int = 0
    acting_nesting: int = 0


def _make_pre_reasoning_hook(
    handler: ExtendedTelemetryHandler,
) -> Any:
    """Create a pre_reasoning hook that opens a new React step span.

    Also closes any leftover step from a previous iteration as a fallback
    (normal path closes via post_acting).

    When multiple ReAct hook wrappers run (subclass ``_reasoning`` calling
    ``super()._reasoning``), only the outermost pre_reasoning opens a step.
    """

    def hook(agent_self: Any, kwargs: dict) -> None:
        state = _REACT_STATE.get()
        if state is None or state.owner is not agent_self:
            return None

        state.reasoning_nesting += 1
        if state.reasoning_nesting != 1:
            return None

        if state.active_step:
            state.active_step.finish_reason = "tool_calls"
            handler.stop_react_step(state.active_step)
            state.active_step = None

        state.react_round += 1
        inv = ReactStepInvocation(round=state.react_round)
        apply_entry_baggage_identity(inv)
        handler.start_react_step(inv, context=state.original_context)
        state.active_step = inv
        state.pending_acting_count = 0
        return None

    return hook


def _make_post_reasoning_hook(
    handler: ExtendedTelemetryHandler,
) -> Any:
    """Create a post_reasoning hook that counts tool_use blocks
    to initialize the pending_acting_count for the current step.

    Inner wrappers' post_reasoning run first on unwind; tool_use counting
    must run on the outermost post_reasoning (last to execute).
    """

    def hook(agent_self: Any, kwargs: dict, output: Any) -> None:
        state = _REACT_STATE.get()
        if state is None or state.owner is not agent_self:
            return None
        try:
            if (
                state.reasoning_nesting == 1
                and output is not None
                and hasattr(output, "get_content_blocks")
            ):
                tool_blocks = output.get_content_blocks("tool_use")
                state.pending_acting_count = len(tool_blocks)
            elif state.reasoning_nesting == 1 and output is not None:
                state.pending_acting_count = 0
        finally:
            if state.reasoning_nesting > 0:
                state.reasoning_nesting -= 1
        return None

    return hook


def _make_pre_acting_hook() -> Any:
    """Track nested _acting wrappers (subclass calls ``super()._acting``)."""

    def hook(agent_self: Any, kwargs: dict) -> None:
        state = _REACT_STATE.get()
        if state is None or state.owner is not agent_self:
            return None
        state.acting_nesting += 1
        return None

    return hook


def _make_post_acting_hook(
    handler: ExtendedTelemetryHandler,
) -> Any:
    """Create a post_acting hook that decrements pending_acting_count
    and closes the step span when all acting calls are done.

    Only the outermost post_acting (last on unwind) updates pending counts.
    """

    def hook(agent_self: Any, kwargs: dict, output: Any) -> None:
        state = _REACT_STATE.get()
        if state is None or state.owner is not agent_self:
            return None
        try:
            if state.acting_nesting == 1 and state.active_step is not None:
                state.pending_acting_count -= 1
                if state.pending_acting_count <= 0:
                    state.active_step.finish_reason = "tool_calls"
                    handler.stop_react_step(state.active_step)
                    state.active_step = None
        finally:
            if state.acting_nesting > 0:
                state.acting_nesting -= 1
        return None

    return hook


_REACT_HOOK_TYPES = (
    "pre_reasoning",
    "post_reasoning",
    "pre_acting",
    "post_acting",
)


def _acquire_react_hooks(
    agent: Any, handler: ExtendedTelemetryHandler
) -> None:
    """Register ReAct step hooks on ``agent`` if no concurrent call has
    already done so, and bump the per-instance reference count.

    Hooks are registered exactly once under the fixed name
    ``_REACT_HOOK_NAME``. Subsequent concurrent invocations only bump the
    refcount; the actual ``register_instance_hook`` calls happen only
    when the count transitions from 0 to 1. This avoids registering the
    same hook twice (which would cause AgentScope to double-fire each
    ``_reasoning`` / ``_acting`` callback) while keeping the hooks alive
    for as long as any in-flight call still needs them.

    If a partial registration fails (e.g. AgentScope rejects the third of
    four ``register_instance_hook`` calls), any hooks already installed
    in this transition are rolled back and the refcount is left at zero
    so the agent ends up in a consistent, un-instrumented state and the
    exception is propagated to the caller.
    """
    with _REACT_HOOK_REGISTRY_LOCK:
        count = _REACT_HOOK_REGISTRY.get(agent, 0)
        if count == 0:
            registrations = (
                ("pre_reasoning", _make_pre_reasoning_hook(handler)),
                ("post_reasoning", _make_post_reasoning_hook(handler)),
                ("pre_acting", _make_pre_acting_hook()),
                ("post_acting", _make_post_acting_hook(handler)),
            )
            installed: list = []
            try:
                for hook_type, hook_fn in registrations:
                    agent.register_instance_hook(
                        hook_type, _REACT_HOOK_NAME, hook_fn
                    )
                    installed.append(hook_type)
            except Exception:
                # Roll back any partially installed hooks so the agent is
                # left exactly as we found it; the refcount stays at 0
                # because we never bumped it past this point.
                for hook_type in installed:
                    try:
                        agent.remove_instance_hook(hook_type, _REACT_HOOK_NAME)
                    except Exception:
                        logger.warning(
                            "AgentScope instrumentation: failed to roll "
                            "back %s hook on %s during acquire failure",
                            hook_type,
                            type(agent).__name__,
                        )
                raise
        _REACT_HOOK_REGISTRY[agent] = count + 1


def _release_react_hooks(agent: Any) -> None:
    """Decrement the ref count and remove hooks when the last in-flight
    call on ``agent`` unwinds.

    Hooks must not be removed while sibling concurrent invocations are
    still running, otherwise their ``_reasoning`` / ``_acting`` callbacks
    would silently stop firing mid-flight.

    Releases without a paired acquire (``count`` already 0) are silently
    tolerated rather than raising: the only way to reach this state is
    for a caller's ``try/finally`` to fire even though the matching
    ``_acquire_react_hooks`` aborted before bumping the refcount, in
    which case there is simply nothing to clean up.
    """
    with _REACT_HOOK_REGISTRY_LOCK:
        current = _REACT_HOOK_REGISTRY.get(agent, 0)
        if current <= 0:
            return
        count = current - 1
        if count <= 0:
            for hook_type in _REACT_HOOK_TYPES:
                try:
                    agent.remove_instance_hook(hook_type, _REACT_HOOK_NAME)
                except (ValueError, KeyError):
                    # AgentScope already lost the hook (e.g. user code
                    # called remove_instance_hook directly or rebuilt the
                    # registry). Log and continue: leaving the refcount
                    # entry around is worse than a noisy log line.
                    logger.warning(
                        "AgentScope instrumentation: %s hook missing "
                        "on %s during release; continuing cleanup",
                        hook_type,
                        type(agent).__name__,
                    )
            try:
                del _REACT_HOOK_REGISTRY[agent]
            except KeyError:
                pass
        else:
            _REACT_HOOK_REGISTRY[agent] = count


class AgentScopeChatModelWrapper:
    """Wrapper for ChatModelBase that hijacks __init__ to replace __call__."""

    _original_methods = {}

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    async def _wrap_streaming_response(
        self, generator: AsyncGenerator, invocation: LLMInvocation
    ) -> AsyncGenerator:
        """Wrap streaming response to update invocation when done."""
        try:
            last_chunk = None
            first_token_received = False
            async for chunk in generator:
                # Record time when first token is received
                if not first_token_received:
                    first_token_received = True
                    invocation.monotonic_first_token_s = timeit.default_timer()

                last_chunk = chunk
                yield chunk

            if last_chunk:
                invocation.output_messages = (
                    convert_chatresponse_to_output_messages(last_chunk)
                )

                if hasattr(last_chunk, "usage") and last_chunk.usage:
                    invocation.input_tokens = getattr(
                        last_chunk.usage, "input_tokens", None
                    )
                    invocation.output_tokens = getattr(
                        last_chunk.usage, "output_tokens", None
                    )

                if hasattr(last_chunk, "id"):
                    invocation.response_id = getattr(last_chunk, "id", None)

            self._handler.stop_llm(invocation)
        except Exception as e:
            self._handler.fail_llm(
                invocation, Error(message=str(e), type=type(e))
            )
            raise

    def __call__(self, wrapped, instance, args, kwargs):
        """
        Hijack ChatModelBase.__init__ to replace the instance's __call__ method.
        """
        model_class = type(instance)

        if model_class in self._instrumented_classes:
            result = wrapped(*args, **kwargs)
            return result

        if model_class not in self._original_methods:
            self._original_methods[model_class] = {}

        result = wrapped(*args, **kwargs)

        if not hasattr(model_class, "__call__") or not callable(
            getattr(model_class, "__call__", None)
        ):
            return

        original_call = model_class.__call__
        if model_class not in self._original_methods:
            self._original_methods[model_class] = {}
        self._original_methods[model_class]["__call__"] = original_call

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            """Async wrapper for ChatModelBase.__call__."""
            parent_depth = _CHAT_MODEL_CALL_DEPTH.get()
            depth_token = _CHAT_MODEL_CALL_DEPTH.set(parent_depth + 1)
            try:
                if parent_depth > 0:
                    return await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                invocation = create_llm_invocation(
                    call_self, call_args, call_kwargs
                )

                self._handler.start_llm(invocation)

                function_name = f"{call_self.__class__.__name__}.__call__"
                invocation.attributes["rpc"] = function_name

                try:
                    result = await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                    if isinstance(result, AsyncGenerator):
                        return self._wrap_streaming_response(
                            result, invocation
                        )

                    invocation.output_messages = (
                        convert_chatresponse_to_output_messages(result)
                    )

                    if hasattr(result, "usage") and result.usage:
                        invocation.input_tokens = getattr(
                            result.usage, "input_tokens", None
                        )
                        invocation.output_tokens = getattr(
                            result.usage, "output_tokens", None
                        )

                    invocation.response_model = invocation.request_model
                    invocation.response_finish_reasons = ["stop"]

                    if hasattr(result, "id"):
                        invocation.response_id = getattr(result, "id", None)

                    self._handler.stop_llm(invocation)
                    return result

                except Exception as e:
                    self._handler.fail_llm(
                        invocation, Error(message=str(e), type=type(e))
                    )
                    raise
            finally:
                _CHAT_MODEL_CALL_DEPTH.reset(depth_token)

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(model_class)


class AgentScopeAgentWrapper:
    """Wrapper for AgentBase that hijacks __init__ to replace __call__.

    Supports concurrent ``await agent(...)`` calls on the same instance:
    per-call state is isolated via ``_REACT_STATE`` ContextVar and hook
    lifetime is managed through ``_acquire_react_hooks``/``_release_react_hooks``.
    """

    _original_methods = {}

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(self, wrapped, instance, args, kwargs):
        """
        Hijack AgentBase.__init__ to replace the instance's __call__ method.
        """
        agent_class = type(instance)

        if agent_class in self._instrumented_classes:
            result = wrapped(*args, **kwargs)
            return result

        if agent_class not in self._original_methods:
            self._original_methods[agent_class] = {}

        result = wrapped(*args, **kwargs)

        if not hasattr(agent_class, "__call__") or not callable(
            getattr(agent_class, "__call__", None)
        ):
            return

        original_call = agent_class.__call__
        if agent_class not in self._original_methods:
            self._original_methods[agent_class] = {}
        self._original_methods[agent_class]["__call__"] = original_call

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any,
            *call_args: Any,
            **call_kwargs: Any,
        ) -> Any:
            """Async wrapper for AgentBase.__call__."""
            parent_depth = _AGENT_CALL_DEPTH.get()
            depth_token = _AGENT_CALL_DEPTH.set(parent_depth + 1)
            try:
                if parent_depth > 0:
                    return await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                try:
                    invocation = create_agent_invocation(
                        call_self, call_args, call_kwargs
                    )

                    self._handler.start_invoke_agent(invocation)

                    function_name = f"{call_self.__class__.__name__}.__call__"
                    invocation.attributes["rpc"] = function_name

                    is_react = _is_react_agent(call_self)
                    state: _ReactStepState | None = None
                    state_token = None
                    if is_react:
                        state = _ReactStepState(
                            owner=call_self,
                            original_context=_get_current_context(),
                        )
                        state_token = _REACT_STATE.set(state)

                    try:
                        if is_react:
                            _acquire_react_hooks(call_self, self._handler)
                        result = await original_call(
                            call_self, *call_args, **call_kwargs
                        )

                        if is_react and state and state.active_step:
                            state.active_step.finish_reason = "stop"
                            self._handler.stop_react_step(state.active_step)
                            state.active_step = None

                        invocation.output_messages = (
                            convert_agent_response_to_output_messages(result)
                        )

                        if hasattr(result, "id"):
                            invocation.response_id = getattr(
                                result, "id", None
                            )

                        self._handler.stop_invoke_agent(invocation)
                        return result

                    except Exception as e:
                        if is_react and state and state.active_step:
                            self._handler.fail_react_step(
                                state.active_step,
                                Error(message=str(e), type=type(e)),
                            )
                            state.active_step = None

                        self._handler.fail_invoke_agent(
                            invocation, Error(message=str(e), type=type(e))
                        )
                        raise

                    finally:
                        if is_react:
                            _release_react_hooks(call_self)
                            if state_token is not None:
                                _REACT_STATE.reset(state_token)

                except Exception as e:
                    logger.exception("Error in agent instrumentation: %s", e)
                    return await original_call(
                        call_self, *call_args, **call_kwargs
                    )
            finally:
                _AGENT_CALL_DEPTH.reset(depth_token)

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(agent_class)


class AgentScopeEmbeddingModelWrapper:
    """Wrapper for EmbeddingModelBase that hijacks __init__ to replace __call__."""

    _original_methods = {}

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(self, wrapped, instance, args, kwargs):
        """
        Hijack EmbeddingModelBase.__init__ to replace the instance's __call__ method.
        """
        embedding_class = type(instance)

        if embedding_class in self._instrumented_classes:
            result = wrapped(*args, **kwargs)
            return result

        if embedding_class not in self._original_methods:
            self._original_methods[embedding_class] = {}

        result = wrapped(*args, **kwargs)

        if not hasattr(embedding_class, "__call__") or not callable(
            getattr(embedding_class, "__call__", None)
        ):
            return

        original_call = embedding_class.__call__
        if embedding_class not in self._original_methods:
            self._original_methods[embedding_class] = {}
        self._original_methods[embedding_class]["__call__"] = original_call

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            """Async wrapper for EmbeddingModelBase.__call__."""
            invocation = create_embedding_invocation(
                call_self, call_args, call_kwargs
            )

            self._handler.start_embedding(invocation)

            try:
                result = await original_call(
                    call_self, *call_args, **call_kwargs
                )

                if hasattr(result, "embeddings") and result.embeddings:
                    invocation.dimension_count = len(result.embeddings[0])
                if hasattr(result, "usage") and result.usage:
                    tokens = getattr(result.usage, "tokens", None)
                    if tokens is not None:
                        invocation.input_tokens = tokens

                invocation.response_model_name = invocation.request_model

                self._handler.stop_embedding(invocation)
                return result

            except Exception as e:
                self._handler.fail_embedding(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(embedding_class)
