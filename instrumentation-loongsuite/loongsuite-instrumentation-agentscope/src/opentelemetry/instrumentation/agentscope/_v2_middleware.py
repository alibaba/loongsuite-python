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

"""AgentScope v2 middleware instrumentation."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import timeit
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Sequence,
)
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from agentscope.agent import Agent
from agentscope.message import Msg
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.tool import ToolResponse

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.context import Context, get_current
from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.handler import _safe_detach
from opentelemetry.util.genai.types import (
    Error,
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
)

from ._skill import _enrich_skill_metadata
from ._usage import _extract_cache_tokens

logger = logging.getLogger(__name__)

_MIDDLEWARE_PARAMETER = "middlewares"
_FIRST_TOKEN_EVENT_TYPES = {
    "text_block_delta",
    "thinking_block_delta",
    "tool_call_delta",
}


@hook_advice("agentscope", "record_cancellation")
def _record_cancellation(invocation: Any, error: BaseException | None) -> None:
    if (
        isinstance(error, asyncio.CancelledError)
        and invocation.span is not None
    ):
        invocation.span.set_attribute("agentscope.cancelled", True)


@dataclass
class _ReactState:
    handler: ExtendedTelemetryHandler
    invocation: ReactStepInvocation
    context: Context
    finish_reason: str | None = None
    error: BaseException | None = None
    finalized: bool = False


@dataclass
class _AgentState:
    handler: ExtendedTelemetryHandler
    invocation: InvokeAgentInvocation
    context: Context
    first_token_seen: bool = False
    last_msg: Msg | None = None
    active_react: _ReactState | None = None
    finalized: bool = False


@dataclass
class _LLMState:
    handler: ExtendedTelemetryHandler
    invocation: LLMInvocation
    context: Context
    first_token_seen: bool = False
    last_chunk: ChatResponse | None = None
    finalized: bool = False


@dataclass
class _ActingState:
    handler: ExtendedTelemetryHandler
    tool_invocation: ExecuteToolInvocation
    tool_context: Context
    react_state: _ReactState | None = None
    owns_react: bool = False
    last_item: Any = None
    tool_finalized: bool = False


def _abandon_invocation(invocation: Any) -> None:
    token = invocation.context_token
    invocation.context_token = None
    _safe_detach(token)
    span = invocation.span
    if span is not None and span.is_recording():
        span.end()


@hook_advice("agentscope", "start_llm")
def _start_llm(
    handler: ExtendedTelemetryHandler,
    invocation: LLMInvocation,
    context: Context,
) -> _LLMState:
    """Start LLM and capture its Context before stream ownership transfer."""

    try:
        handler.start_llm(invocation, context=context)
        llm_context = get_current()
    except Exception:
        _abandon_invocation(invocation)
        raise

    return _LLMState(
        handler=handler,
        invocation=invocation,
        context=llm_context,
    )


@hook_advice("agentscope", "release_llm_context")
def _release_llm_context(state: _LLMState) -> bool:
    """Detach the start token in its creation Task before returning a stream."""

    token = state.invocation.context_token
    state.invocation.context_token = None
    _safe_detach(token)
    return True


@hook_advice("agentscope", "record_llm_chunk")
def _record_llm_chunk(state: _LLMState, chunk: ChatResponse) -> None:
    if not state.first_token_seen:
        state.invocation.monotonic_first_token_s = timeit.default_timer()
        state.first_token_seen = True
    state.last_chunk = chunk


@hook_advice("agentscope", "finish_llm")
def _finish_llm(
    state: _LLMState,
    error: BaseException | None = None,
) -> None:
    if state.finalized:
        return
    state.finalized = True

    invocation = state.invocation
    token = None
    callback_completed = False
    span = invocation.span
    try:
        token = otel_context.attach(state.context)
        invocation.context_token = token
        _record_cancellation(invocation, error)
        if error is None or isinstance(
            error, (GeneratorExit, asyncio.CancelledError)
        ):
            if error is None or getattr(state.last_chunk, "is_last", False):
                _finish_llm_invocation(invocation, state.last_chunk)
            state.handler.stop_llm(invocation)
        else:
            state.handler.fail_llm(
                invocation,
                Error(
                    message=str(error) or type(error).__name__,
                    type=type(error),
                ),
            )
        callback_completed = True
    finally:
        invocation.context_token = None
        if get_current() is state.context:
            _safe_detach(token)
        if not callback_completed and span is not None and span.is_recording():
            span.end()


@hook_advice("agentscope", "start_invoke_agent")
def _start_agent(
    handler: ExtendedTelemetryHandler,
    invocation: InvokeAgentInvocation,
) -> _AgentState:
    """Start Agent, save its Context, then detach before yielding ownership."""

    try:
        handler.start_invoke_agent(invocation)
        agent_context = get_current()
        if invocation.conversation_id:
            agent_context = baggage.set_baggage(
                "gen_ai.session.id", invocation.conversation_id, agent_context
            )
    except Exception:
        _abandon_invocation(invocation)
        raise

    token = invocation.context_token
    invocation.context_token = None
    _safe_detach(token)
    return _AgentState(
        handler=handler,
        invocation=invocation,
        context=agent_context,
    )


@hook_advice("agentscope", "attach_stream_context")
def _attach_stream_context(context: Context) -> object:
    return otel_context.attach(context)


@hook_advice("agentscope", "detach_stream_context")
def _detach_stream_context(token: object | None) -> None:
    _safe_detach(token)


@hook_advice("agentscope", "record_agent_chunk")
def _record_agent_chunk(state: _AgentState, item: Any) -> None:
    # AgentScope may consume CancelledError and emit an interrupted reply
    # instead. Preserve that signal without changing the framework behavior.
    reason = getattr(item, "finished_reason", None)
    if getattr(reason, "value", reason) == "interrupted":
        if state.invocation.span is not None:
            state.invocation.span.set_attribute("agentscope.cancelled", True)
        state.invocation.finish_reasons = ["interrupted"]
    if not state.first_token_seen and _is_first_token_event(item):
        state.invocation.monotonic_first_token_s = timeit.default_timer()
        state.first_token_seen = True
    if isinstance(item, Msg):
        state.last_msg = item


@hook_advice("agentscope", "finish_invoke_agent")
def _finish_agent(
    state: _AgentState,
    error: BaseException | None = None,
) -> None:
    if state.finalized:
        return
    state.finalized = True

    invocation = state.invocation
    token = None
    span = invocation.span
    try:
        token = otel_context.attach(state.context)
        invocation.context_token = token
        if state.last_msg is not None:
            invocation.output_messages = [_message_to_output(state.last_msg)]
            if state.last_msg.usage is not None:
                invocation.input_tokens = state.last_msg.usage.input_tokens
                invocation.output_tokens = state.last_msg.usage.output_tokens
                _extract_cache_tokens(state.last_msg.usage, invocation)
        _record_cancellation(invocation, error)
        if error is None or isinstance(
            error, (GeneratorExit, asyncio.CancelledError)
        ):
            state.handler.stop_invoke_agent(invocation)
        else:
            state.handler.fail_invoke_agent(
                invocation,
                Error(
                    message=str(error) or type(error).__name__,
                    type=type(error),
                ),
            )
    finally:
        invocation.context_token = None
        if span is not None and span.is_recording():
            _safe_detach(token)
            span.end()


@hook_advice("agentscope", "start_react_step")
def _start_react(
    handler: ExtendedTelemetryHandler,
    agent: Agent,
    context: Context,
) -> _ReactState:
    react_invocation = ReactStepInvocation(
        round=getattr(
            getattr(agent, "state", None),
            "cur_iter",
            0,
        )
        + 1
    )
    _apply_identity(react_invocation, context=context)

    try:
        handler.start_react_step(react_invocation, context=context)
        react_context = get_current()
    except Exception:
        _abandon_invocation(react_invocation)
        raise

    react_token = react_invocation.context_token
    react_invocation.context_token = None
    _safe_detach(react_token)
    return _ReactState(
        handler=handler,
        invocation=react_invocation,
        context=react_context,
    )


@hook_advice("agentscope", "record_react_event")
def _record_react_event(state: _ReactState, item: Any) -> None:
    event_type = str(getattr(item, "type", "")).lower()
    if event_type in {
        "tool_call",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
    }:
        state.finish_reason = "tool_calls"
    elif isinstance(item, Msg) and state.finish_reason != "tool_calls":
        state.finish_reason = "stop"


@hook_advice("agentscope", "record_react_error")
def _record_react_error(
    state: _ReactState,
    error: BaseException,
) -> None:
    if not isinstance(error, GeneratorExit):
        state.error = error


def _create_tool_invocation(
    agent: Agent,
    tool_call: Any,
) -> ExecuteToolInvocation:
    tool_name = str(getattr(tool_call, "name", "unknown_tool"))
    tool_call_arguments = _loads_json(getattr(tool_call, "input", None))
    tool_invocation = ExecuteToolInvocation(
        tool_name=tool_name,
        tool_type="function",
        tool_call_id=getattr(tool_call, "id", None),
        tool_call_arguments=tool_call_arguments,
        provider="agentscope",
    )
    _apply_identity(tool_invocation)
    _apply_skill_metadata(
        tool_invocation,
        agent,
        tool_name,
        tool_call_arguments,
    )
    return tool_invocation


def _apply_skill_metadata(
    invocation: ExecuteToolInvocation,
    agent: Agent,
    tool_name: str,
    tool_call_arguments: Any,
) -> None:
    """Enrich AgentScope v2's built-in Skill viewer tool span."""

    if tool_name.lower() != "skill" or not isinstance(
        tool_call_arguments, dict
    ):
        return

    requested_name = tool_call_arguments.get("skill")
    if not isinstance(requested_name, str) or not requested_name:
        return

    metadata: dict[str, Any] = {"name": requested_name}
    toolkit = getattr(agent, "toolkit", None)

    qwenpaw_skills = getattr(toolkit, "_qp_skills", None)
    if isinstance(qwenpaw_skills, dict):
        qwenpaw_metadata = qwenpaw_skills.get(requested_name)
        if isinstance(qwenpaw_metadata, dict):
            metadata.update(qwenpaw_metadata)

    for group in getattr(toolkit, "tool_groups", ()) or ():
        for entry in getattr(group, "skills_or_loaders", ()) or ():
            candidates = [entry]
            cache = getattr(entry, "_cache", None)
            if isinstance(cache, dict):
                candidates.extend(cache.values())
            for candidate in candidates:
                if getattr(candidate, "name", None) != requested_name:
                    continue
                for field_name in ("name", "description", "dir"):
                    value = getattr(candidate, field_name, None)
                    if value not in (None, ""):
                        metadata[field_name] = value

    enriched = _enrich_skill_metadata(metadata)
    invocation.skill_name = enriched.get("name") or requested_name
    invocation.skill_id = enriched.get("id") or requested_name
    invocation.skill_description = enriched.get("description")
    invocation.skill_version = enriched.get("version")


@hook_advice("agentscope", "start_acting")
def _start_acting(
    handler: ExtendedTelemetryHandler,
    agent: Agent,
    tool_call: Any,
    react_state: _ReactState | None,
) -> _ActingState:
    tool_invocation = _create_tool_invocation(agent, tool_call)
    owns_react = react_state is None
    if react_state is None:
        react_state = _start_react(handler, agent, get_current())

    parent_context = (
        react_state.context if react_state is not None else get_current()
    )

    try:
        handler.start_execute_tool(tool_invocation, context=parent_context)
        tool_context = get_current()
    except Exception:
        _abandon_invocation(tool_invocation)
        if owns_react and react_state is not None:
            _finish_react(react_state)
        raise

    tool_token = tool_invocation.context_token
    tool_invocation.context_token = None
    _safe_detach(tool_token)
    return _ActingState(
        handler=handler,
        tool_invocation=tool_invocation,
        tool_context=tool_context,
        react_state=react_state,
        owns_react=owns_react,
    )


@hook_advice("agentscope", "finish_execute_tool")
def _finish_tool(
    state: _ActingState,
    error: BaseException | None = None,
) -> None:
    if state.tool_finalized:
        return
    state.tool_finalized = True

    invocation = state.tool_invocation
    if error is None or isinstance(
        error, (GeneratorExit, asyncio.CancelledError)
    ):
        if isinstance(state.last_item, ToolResponse):
            invocation.tool_call_result = _jsonable(
                _blocks_to_parts(state.last_item.content)
            )
        elif state.last_item is not None:
            invocation.tool_call_result = str(state.last_item)

    token = None
    span = invocation.span
    try:
        token = otel_context.attach(state.tool_context)
        invocation.context_token = token
        _record_cancellation(invocation, error)
        if error is None or isinstance(
            error, (GeneratorExit, asyncio.CancelledError)
        ):
            state.handler.stop_execute_tool(invocation)
        else:
            state.handler.fail_execute_tool(
                invocation,
                Error(
                    message=str(error) or type(error).__name__,
                    type=type(error),
                ),
            )
    finally:
        invocation.context_token = None
        if span is not None and span.is_recording():
            _safe_detach(token)
            span.end()


@hook_advice("agentscope", "finish_react_step")
def _finish_react(
    state: _ReactState,
    error: BaseException | None = None,
    finish_reason: str | None = None,
) -> None:
    if state.finalized:
        return
    state.finalized = True

    invocation = state.invocation
    if finish_reason is not None:
        state.finish_reason = finish_reason
    invocation.finish_reason = state.finish_reason
    failure = error
    if failure is None:
        failure = state.error
    _record_cancellation(invocation, failure)
    if isinstance(failure, (GeneratorExit, asyncio.CancelledError)):
        if isinstance(failure, asyncio.CancelledError):
            invocation.finish_reason = None
        failure = None

    token = None
    span = invocation.span
    try:
        token = otel_context.attach(state.context)
        invocation.context_token = token
        if failure is None:
            state.handler.stop_react_step(invocation)
        else:
            state.handler.fail_react_step(
                invocation,
                Error(
                    message=str(failure) or type(failure).__name__,
                    type=type(failure),
                ),
            )
    finally:
        invocation.context_token = None
        if span is not None and span.is_recording():
            _safe_detach(token)
            span.end()


def _finish_acting(
    state: _ActingState,
    error: BaseException | None = None,
) -> None:
    _finish_tool(state, error)
    if state.react_state is not None:
        if error is not None:
            _record_react_error(state.react_state, error)
        if state.owns_react:
            _finish_react(
                state.react_state,
                error,
                finish_reason=(
                    "tool_calls"
                    if error is None or isinstance(error, GeneratorExit)
                    else None
                ),
            )


async def _close_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


def append_loongsuite_middleware(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    middleware: "AgentScopeV2Middleware",
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Append LoongSuite middleware to AgentScope v2 Agent.__init__ inputs."""
    if _MIDDLEWARE_PARAMETER in kwargs:
        kwargs = dict(kwargs)
        kwargs[_MIDDLEWARE_PARAMETER] = _append_once(
            kwargs.get(_MIDDLEWARE_PARAMETER), middleware
        )
        return args, kwargs

    middleware_position = _middleware_arg_position()
    if middleware_position is not None and len(args) > middleware_position:
        updated_args = list(args)
        updated_args[middleware_position] = _append_once(
            updated_args[middleware_position],
            middleware,
        )
        return tuple(updated_args), kwargs

    kwargs = dict(kwargs)
    kwargs[_MIDDLEWARE_PARAMETER] = [middleware]
    return args, kwargs


def _append_once(
    middlewares: Sequence[MiddlewareBase] | None,
    middleware: "AgentScopeV2Middleware",
) -> list[MiddlewareBase]:
    result = list(middlewares or [])
    if any(isinstance(item, AgentScopeV2Middleware) for item in result):
        return result
    result.append(middleware)
    return result


class AgentScopeV2Middleware(MiddlewareBase):
    """LoongSuite telemetry adapter for AgentScope v2 middleware hooks."""

    def __init__(
        self, handler: Callable[[], ExtendedTelemetryHandler | None]
    ) -> None:
        self._handler = handler
        # AgentScope itself stores reply_id, cur_iter, and memory on the Agent,
        # so one Agent instance supports one in-flight reply at a time.
        self._reply_states: dict[int, _AgentState] = {}

    def _reply_state(self, agent: Agent) -> _AgentState | None:
        return self._reply_states.get(id(agent))

    def _clear_reply_state(
        self,
        agent: Agent,
        state: _AgentState,
    ) -> None:
        if self._reply_states.get(id(agent)) is state:
            self._reply_states.pop(id(agent), None)

    def _finish_active_react(
        self,
        agent: Agent,
        state: _AgentState,
        error: BaseException | None = None,
        finish_reason: str | None = None,
    ) -> None:
        react_state = state.active_react
        state.active_react = None
        if react_state is not None:
            _finish_react(react_state, error, finish_reason)

    async def on_reply(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        handler = self._handler()
        if handler is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return

        invocation = _create_agent_invocation(agent, input_kwargs)
        if invocation is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return
        state = _start_agent(handler, invocation)
        try:
            business_stream = next_handler(**input_kwargs)
            iterator = business_stream.__aiter__()
        except BaseException as exc:
            if state is not None:
                _finish_agent(state, exc)
            raise
        if state is None:
            try:
                async for item in iterator:
                    yield item
            except GeneratorExit:
                await _close_iterator(iterator)
                raise
            return

        self._reply_states[id(agent)] = state
        try:
            while True:
                token = _attach_stream_context(state.context)
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                finally:
                    _detach_stream_context(token)

                _record_agent_chunk(state, item)
                yield item
        except GeneratorExit as exc:
            token = _attach_stream_context(state.context)
            try:
                await _close_iterator(iterator)
            except BaseException as close_error:
                self._finish_active_react(agent, state, close_error)
                _finish_agent(state, close_error)
                raise
            finally:
                _detach_stream_context(token)
            self._finish_active_react(agent, state, exc)
            _finish_agent(state, exc)
            raise
        except BaseException as exc:
            self._finish_active_react(agent, state, exc)
            _finish_agent(state, exc)
            raise
        finally:
            if state.active_react is not None:
                self._finish_active_react(agent, state)
            if not state.finalized:
                _finish_agent(state)
            self._clear_reply_state(agent, state)

    async def on_reasoning(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        handler = self._handler()
        reply_state = self._reply_state(agent)
        if handler is None or reply_state is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return

        self._finish_active_react(
            agent,
            reply_state,
            finish_reason="tool_calls",
        )
        state = _start_react(handler, agent, reply_state.context)
        if state is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return
        reply_state.active_react = state

        try:
            business_stream = next_handler(**input_kwargs)
            iterator = business_stream.__aiter__()
        except BaseException as exc:
            self._finish_active_react(agent, reply_state, exc)
            raise

        try:
            while True:
                token = _attach_stream_context(state.context)
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                finally:
                    _detach_stream_context(token)

                _record_react_event(state, item)
                yield item
        except GeneratorExit as exc:
            token = _attach_stream_context(state.context)
            try:
                await _close_iterator(iterator)
            except BaseException as close_error:
                self._finish_active_react(
                    agent,
                    reply_state,
                    close_error,
                )
                raise
            finally:
                _detach_stream_context(token)
            self._finish_active_react(agent, reply_state, exc)
            raise
        except BaseException as exc:
            self._finish_active_react(agent, reply_state, exc)
            raise
        finally:
            if (
                reply_state.active_react is state
                and state.finish_reason != "tool_calls"
            ):
                self._finish_active_react(
                    agent,
                    reply_state,
                    finish_reason=state.finish_reason or "stop",
                )

    async def on_model_call(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[
            ...,
            Awaitable[ChatResponse | AsyncGenerator[ChatResponse, None]],
        ],
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        model = input_kwargs.get("current_model")
        if not isinstance(model, ChatModelBase):
            return await next_handler(**input_kwargs)

        handler = self._handler()
        if handler is None:
            return await next_handler(**input_kwargs)

        invocation = _create_llm_invocation(model, input_kwargs)
        if invocation is None:
            return await next_handler(**input_kwargs)
        state = _start_llm(handler, invocation, get_current())
        try:
            result = await next_handler(**input_kwargs)
        except BaseException as exc:
            if state is not None:
                if _release_llm_context(state):
                    _finish_llm(state, exc)
                else:
                    _abandon_invocation(invocation)
            raise
        if state is None:
            return result
        if not _release_llm_context(state):
            _abandon_invocation(invocation)
            return result
        if inspect.isasyncgen(result):
            return self._wrap_model_stream(result, state)

        state.last_chunk = result
        _finish_llm(state)
        return result

    async def _wrap_model_stream(
        self,
        result: AsyncGenerator[ChatResponse, None],
        state: _LLMState,
    ) -> AsyncGenerator[ChatResponse, None]:
        iterator = result.__aiter__()
        try:
            while True:
                token = _attach_stream_context(state.context)
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                finally:
                    _detach_stream_context(token)

                _record_llm_chunk(state, chunk)
                yield chunk
        except GeneratorExit as exc:
            token = _attach_stream_context(state.context)
            try:
                await _close_iterator(iterator)
            except BaseException as close_error:
                _finish_llm(state, close_error)
                raise
            finally:
                _detach_stream_context(token)
            _finish_llm(state, exc)
            raise
        except BaseException as exc:
            _finish_llm(state, exc)
            raise
        finally:
            if not state.finalized:
                _finish_llm(state)

    async def on_acting(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        handler = self._handler()
        if handler is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return

        tool_call = input_kwargs.get("tool_call")
        reply_state = self._reply_state(agent)
        react_state = (
            reply_state.active_react if reply_state is not None else None
        )
        if reply_state is not None and react_state is None:
            react_state = _start_react(handler, agent, reply_state.context)
            reply_state.active_react = react_state

        state = _start_acting(handler, agent, tool_call, react_state)
        try:
            business_stream = next_handler(**input_kwargs)
            iterator = business_stream.__aiter__()
        except BaseException as exc:
            if state is not None:
                _finish_acting(state, exc)
            raise
        if state is None:
            try:
                async for item in iterator:
                    yield item
            except GeneratorExit:
                await _close_iterator(iterator)
                raise
            return

        try:
            while True:
                token = _attach_stream_context(state.tool_context)
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                finally:
                    _detach_stream_context(token)

                state.last_item = item
                yield item
        except GeneratorExit as exc:
            token = _attach_stream_context(state.tool_context)
            try:
                await _close_iterator(iterator)
            except BaseException as close_error:
                _finish_acting(state, close_error)
                raise
            finally:
                _detach_stream_context(token)
            _finish_acting(state, exc)
            raise
        except BaseException as exc:
            _finish_acting(state, exc)
            raise
        finally:
            if not state.tool_finalized or (
                state.owns_react
                and state.react_state is not None
                and not state.react_state.finalized
            ):
                _finish_acting(state)


def _apply_identity(
    invocation: Any, context: Context | None = None
) -> str | None:
    """Prefer the Entry's business identity over framework-local state IDs.

    Use ordinary W3C baggage, as the v1 adapter does. Robin's additional
    traffic-coloring prefix is intentionally not part of the OSS contract.
    """
    for key in ("gen_ai.session.id", "gen_ai.user.id"):
        value = baggage.get_baggage(key, context=context)
        if isinstance(value, str) and value.strip():
            invocation.attributes[key] = value
    session_id = invocation.attributes.get("gen_ai.session.id")
    if session_id:
        invocation.attributes["gen_ai.conversation.id"] = session_id
    return session_id


@hook_advice("agentscope", "create_agent_invocation")
def _create_agent_invocation(
    agent: Agent,
    input_kwargs: dict[str, Any],
) -> InvokeAgentInvocation:
    model = getattr(agent, "model", None)
    request_model = getattr(model, "model", None)
    provider = _get_provider_name(model)
    inputs = input_kwargs.get("inputs")
    invocation = InvokeAgentInvocation(
        provider=provider,
        agent_name=otel_context.get_value("qwenpaw.dream.agent.name")
        or getattr(agent, "name", "unknown_agent"),
        agent_id=getattr(getattr(agent, "state", None), "session_id", None),
        conversation_id=getattr(
            getattr(agent, "state", None), "session_id", None
        ),
        request_model=request_model,
        input_messages=_messages_to_inputs(inputs),
        system_instruction=[
            Text(content=getattr(agent, "_system_prompt", ""))
        ],
    )
    session_id = _apply_identity(invocation)
    if session_id:
        invocation.conversation_id = session_id
    return invocation


@hook_advice("agentscope", "create_llm_invocation")
def _create_llm_invocation(
    model: ChatModelBase,
    input_kwargs: dict[str, Any],
) -> LLMInvocation:
    invocation = LLMInvocation(
        request_model=getattr(model, "model", None),
        provider=_get_provider_name(model),
        input_messages=_messages_to_inputs(
            input_kwargs.get("messages"),
            include_reasoning=getattr(
                getattr(model, "formatter", None),
                "supports_thinking_input",
                True,
            ),
        ),
        tool_definitions=_tool_definitions(input_kwargs.get("tools")),
    )
    _apply_identity(invocation)
    parameters = getattr(model, "parameters", None)
    for source in (parameters, input_kwargs):
        _set_if_present(invocation, "temperature", source)
        _set_if_present(invocation, "top_p", source)
        _set_if_present(invocation, "max_tokens", source)
    return invocation


def _finish_llm_invocation(
    invocation: LLMInvocation,
    response: ChatResponse | None,
) -> None:
    if response is None:
        return
    invocation.response_id = getattr(response, "id", None)
    invocation.output_messages = [_chat_response_to_output(response)]
    usage = getattr(response, "usage", None)
    if usage is not None:
        invocation.input_tokens = getattr(usage, "input_tokens", None)
        invocation.output_tokens = getattr(usage, "output_tokens", None)
        _extract_cache_tokens(usage, invocation)


def _messages_to_inputs(
    value: Any, *, include_reasoning: bool = True
) -> list[InputMessage]:
    if value is None:
        return []
    if isinstance(value, Msg):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    messages = []
    for msg in value:
        if not isinstance(msg, Msg):
            continue
        pending = []
        # AgentScope v2 keeps multiple ReAct rounds in one assistant Msg.
        # Emit tool results as separate messages, preserving call/result order
        # without modifying the framework's persisted conversation.
        for part in _blocks_to_parts(msg.content):
            if isinstance(part, ToolCallResponse):
                if pending:
                    messages.append(InputMessage(role=msg.role, parts=pending))
                    pending = []
                messages.append(InputMessage(role="tool", parts=[part]))
            elif include_reasoning or not isinstance(part, Reasoning):
                pending.append(part)
        if pending:
            messages.append(InputMessage(role=msg.role, parts=pending))
    return messages


def _message_to_output(msg: Msg) -> OutputMessage:
    # The reply Msg is cumulative, not just the final response. Keep only
    # visible text after the last tool interaction at the AGENT boundary;
    # intermediate reasoning and tools remain on their child spans.
    parts = []
    for part in _blocks_to_parts(msg.content):
        if isinstance(part, (ToolCall, ToolCallResponse)):
            parts = []
        elif isinstance(part, Text):
            parts.append(part)
    return OutputMessage(
        role=msg.role,
        parts=parts,
        finish_reason="stop",
    )


def _chat_response_to_output(response: ChatResponse) -> OutputMessage:
    finish_reason = "stop"
    if any(
        getattr(block, "type", None) == "tool_call"
        for block in response.content
    ):
        finish_reason = "tool_calls"
    return OutputMessage(
        role="assistant",
        parts=_blocks_to_parts(response.content),
        finish_reason=finish_reason,
    )


def _blocks_to_parts(blocks: Sequence[Any]) -> list[Any]:
    if blocks is None:
        return []
    if isinstance(blocks, str):
        return [Text(content=blocks)]
    parts = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(Text(content=getattr(block, "text", "")))
        elif block_type == "thinking":
            parts.append(Reasoning(content=getattr(block, "thinking", "")))
        elif block_type == "tool_call":
            parts.append(
                ToolCall(
                    id=getattr(block, "id", None),
                    name=getattr(block, "name", ""),
                    arguments=_loads_json(getattr(block, "input", None)),
                )
            )
        elif block_type == "tool_result":
            parts.append(
                ToolCallResponse(
                    id=getattr(block, "id", None),
                    response=_tool_result_response(
                        getattr(block, "output", "")
                    ),
                )
            )
    return parts


def _tool_result_response(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return _blocks_to_parts(value)
    return value


def _tool_definitions(tools: list[dict[str, Any]] | None) -> list[Any]:
    if not tools:
        return []
    definitions = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        definitions.append(
            FunctionToolDefinition(
                name=function.get("name", ""),
                description=function.get("description"),
                parameters=function.get("parameters"),
            )
        )
    return definitions


def _get_provider_name(model: Any) -> str:
    class_name = model.__class__.__name__.lower() if model is not None else ""
    if "dashscope" in class_name:
        return "dashscope"
    if "openai" in class_name:
        return "openai"
    if "anthropic" in class_name:
        return "anthropic"
    if "gemini" in class_name:
        return "gcp.gen_ai"
    if "ollama" in class_name:
        return "ollama"
    return "agentscope"


def _is_first_token_event(item: Any) -> bool:
    event_type = getattr(item, "type", None)
    return event_type in _FIRST_TOKEN_EVENT_TYPES


def _middleware_arg_position() -> int | None:
    try:
        parameters = list(inspect.signature(Agent.__init__).parameters)
        return parameters.index(_MIDDLEWARE_PARAMETER) - 1
    except (TypeError, ValueError):
        return None


def _is_streaming_model(
    model: ChatModelBase, input_kwargs: dict[str, Any]
) -> bool:
    if "stream" in input_kwargs:
        return bool(input_kwargs["stream"])
    return bool(getattr(model, "stream", False))


def _loads_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _set_if_present(
    invocation: LLMInvocation,
    field_name: str,
    source: Any,
) -> None:
    value = (
        source.get(field_name)
        if isinstance(source, dict)
        else getattr(source, field_name, None)
    )
    if value is not None:
        setattr(invocation, field_name, value)
