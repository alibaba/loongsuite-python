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

logger = logging.getLogger(__name__)

_MIDDLEWARE_PARAMETER = "middlewares"
_FIRST_TOKEN_EVENT_TYPES = {
    "text_block_delta",
    "thinking_block_delta",
    "tool_call_delta",
}


@dataclass
class _AgentState:
    handler: ExtendedTelemetryHandler
    invocation: InvokeAgentInvocation
    context: Context
    first_token_seen: bool = False
    last_msg: Msg | None = None
    finalized: bool = False


@dataclass
class _ActingState:
    handler: ExtendedTelemetryHandler
    react_invocation: ReactStepInvocation
    tool_invocation: ExecuteToolInvocation
    react_context: Context
    tool_context: Context
    last_item: Any = None
    tool_finalized: bool = False
    react_finalized: bool = False


def _abandon_invocation(invocation: Any) -> None:
    token = invocation.context_token
    invocation.context_token = None
    _safe_detach(token)
    span = invocation.span
    if span is not None and span.is_recording():
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
    if state.last_msg is not None:
        invocation.output_messages = [_message_to_output(state.last_msg)]
        if state.last_msg.usage is not None:
            invocation.input_tokens = state.last_msg.usage.input_tokens
            invocation.output_tokens = state.last_msg.usage.output_tokens

    token = None
    span = invocation.span
    try:
        token = otel_context.attach(state.context)
        invocation.context_token = token
        if error is None or isinstance(error, GeneratorExit):
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


@hook_advice("agentscope", "start_acting")
def _start_acting(
    handler: ExtendedTelemetryHandler,
    agent: Agent,
    tool_call: Any,
) -> _ActingState:
    react_invocation = ReactStepInvocation(
        round=getattr(
            getattr(agent, "state", None),
            "cur_iter",
            0,
        )
        + 1
    )
    tool_invocation = ExecuteToolInvocation(
        tool_name=getattr(tool_call, "name", "unknown_tool"),
        tool_type="function",
        tool_call_id=getattr(tool_call, "id", None),
        tool_call_arguments=_loads_json(getattr(tool_call, "input", None)),
        provider="agentscope",
    )

    try:
        handler.start_react_step(react_invocation, context=get_current())
        react_context = get_current()
        handler.start_execute_tool(tool_invocation)
        tool_context = get_current()
    except Exception:
        _abandon_invocation(tool_invocation)
        _abandon_invocation(react_invocation)
        raise

    tool_token = tool_invocation.context_token
    tool_invocation.context_token = None
    _safe_detach(tool_token)
    react_token = react_invocation.context_token
    react_invocation.context_token = None
    _safe_detach(react_token)
    return _ActingState(
        handler=handler,
        react_invocation=react_invocation,
        tool_invocation=tool_invocation,
        react_context=react_context,
        tool_context=tool_context,
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
    if error is None or isinstance(error, GeneratorExit):
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
        if error is None or isinstance(error, GeneratorExit):
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
    state: _ActingState,
    error: BaseException | None = None,
) -> None:
    if state.react_finalized:
        return
    state.react_finalized = True

    invocation = state.react_invocation
    token = None
    span = invocation.span
    try:
        token = otel_context.attach(state.react_context)
        invocation.context_token = token
        if error is None or isinstance(error, GeneratorExit):
            invocation.finish_reason = "tool_calls"
            state.handler.stop_react_step(invocation)
        else:
            state.handler.fail_react_step(
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


def _finish_acting(
    state: _ActingState,
    error: BaseException | None = None,
) -> None:
    _finish_tool(state, error)
    _finish_react(state, error)


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
                _finish_agent(state, close_error)
                raise
            finally:
                _detach_stream_context(token)
            _finish_agent(state, exc)
            raise
        except BaseException as exc:
            _finish_agent(state, exc)
            raise
        finally:
            if not state.finalized:
                _finish_agent(state)

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
        span_context = get_current()
        handler.start_llm(invocation, context=span_context)
        try:
            result = await next_handler(**input_kwargs)
            if inspect.isasyncgen(result):
                return self._wrap_model_stream(
                    result,
                    invocation,
                    handler,
                )

            _finish_llm_invocation(invocation, result)
            handler.stop_llm(invocation)
            return result
        except BaseException as exc:
            handler.fail_llm(
                invocation,
                Error(message=str(exc) or type(exc).__name__, type=type(exc)),
            )
            raise

    async def _wrap_model_stream(
        self,
        result: AsyncGenerator[ChatResponse, None],
        invocation: LLMInvocation,
        handler: ExtendedTelemetryHandler,
    ) -> AsyncGenerator[ChatResponse, None]:
        first_token_seen = False
        last_chunk = None
        closed = False
        try:
            async for chunk in result:
                if not first_token_seen:
                    invocation.monotonic_first_token_s = timeit.default_timer()
                    first_token_seen = True
                last_chunk = chunk
                yield chunk
        except BaseException as exc:
            handler.fail_llm(
                invocation,
                Error(message=str(exc) or type(exc).__name__, type=type(exc)),
            )
            closed = True
            raise
        else:
            _finish_llm_invocation(invocation, last_chunk)
            handler.stop_llm(invocation)
            closed = True
        finally:
            if not closed:
                handler.stop_llm(invocation)

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
        state = _start_acting(handler, agent, tool_call)
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
            if not state.tool_finalized or not state.react_finalized:
                _finish_acting(state)


def _create_agent_invocation(
    agent: Agent,
    input_kwargs: dict[str, Any],
) -> InvokeAgentInvocation:
    model = getattr(agent, "model", None)
    request_model = getattr(model, "model", None)
    provider = _get_provider_name(model)
    inputs = input_kwargs.get("inputs")
    return InvokeAgentInvocation(
        provider=provider,
        agent_name=getattr(agent, "name", "unknown_agent"),
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


def _create_llm_invocation(
    model: ChatModelBase,
    input_kwargs: dict[str, Any],
) -> LLMInvocation:
    invocation = LLMInvocation(
        request_model=getattr(model, "model", None),
        provider=_get_provider_name(model),
        input_messages=_messages_to_inputs(input_kwargs.get("messages")),
        tool_definitions=_tool_definitions(input_kwargs.get("tools")),
    )
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


def _messages_to_inputs(value: Any) -> list[InputMessage]:
    if value is None:
        return []
    if isinstance(value, Msg):
        return [_message_to_input(value)]
    if isinstance(value, list):
        return [
            _message_to_input(item) for item in value if isinstance(item, Msg)
        ]
    return []


def _message_to_input(msg: Msg) -> InputMessage:
    return InputMessage(role=msg.role, parts=_blocks_to_parts(msg.content))


def _message_to_output(msg: Msg) -> OutputMessage:
    return OutputMessage(
        role=msg.role,
        parts=_blocks_to_parts(msg.content),
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
