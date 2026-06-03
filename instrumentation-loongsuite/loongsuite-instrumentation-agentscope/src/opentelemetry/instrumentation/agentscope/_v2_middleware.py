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
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from typing import Any

from agentscope.agent import Agent
from agentscope.message import Msg
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.tool import ToolResponse

from opentelemetry.context import Context, get_current
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
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
        self._react_round: ContextVar[int] = ContextVar(
            "loongsuite_agentscope_v2_react_round",
            default=0,
        )

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
        handler.start_invoke_agent(invocation)
        round_token = self._react_round.set(0)
        first_token_seen = False
        last_msg = None
        closed = False
        try:
            async for item in next_handler(**input_kwargs):
                if not first_token_seen and _is_first_token_event(item):
                    invocation.monotonic_first_token_s = timeit.default_timer()
                    first_token_seen = True
                if isinstance(item, Msg):
                    last_msg = item
                yield item
        except BaseException as exc:
            handler.fail_invoke_agent(
                invocation,
                Error(message=str(exc) or type(exc).__name__, type=type(exc)),
            )
            closed = True
            raise
        else:
            if last_msg is not None:
                invocation.output_messages = [_message_to_output(last_msg)]
                if last_msg.usage is not None:
                    invocation.input_tokens = last_msg.usage.input_tokens
                    invocation.output_tokens = last_msg.usage.output_tokens
            handler.stop_invoke_agent(invocation)
            closed = True
        finally:
            self._react_round.reset(round_token)
            if not closed:
                handler.stop_invoke_agent(invocation)

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
        started = False
        if not _is_streaming_model(model, input_kwargs):
            handler.start_llm(invocation, context=span_context)
            started = True
        try:
            result = await next_handler(**input_kwargs)
            if inspect.isasyncgen(result):
                return self._wrap_model_stream(
                    result,
                    invocation,
                    span_context,
                    handler,
                    span_started=started,
                )

            if not started:
                handler.start_llm(invocation, context=span_context)
                started = True
            _finish_llm_invocation(invocation, result)
            handler.stop_llm(invocation)
            return result
        except BaseException as exc:
            if not started:
                handler.start_llm(invocation, context=span_context)
            handler.fail_llm(
                invocation,
                Error(message=str(exc) or type(exc).__name__, type=type(exc)),
            )
            raise

    async def _wrap_model_stream(
        self,
        result: AsyncGenerator[ChatResponse, None],
        invocation: LLMInvocation,
        span_context: Context,
        handler: ExtendedTelemetryHandler,
        *,
        span_started: bool,
    ) -> AsyncGenerator[ChatResponse, None]:
        first_token_seen = False
        last_chunk = None
        closed = False
        if not span_started:
            handler.start_llm(invocation, context=span_context)
            span_started = True
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
            if span_started and not closed:
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
        react_invocation = ReactStepInvocation(round=self._next_react_round())
        handler.start_react_step(react_invocation, context=get_current())
        invocation = ExecuteToolInvocation(
            tool_name=getattr(tool_call, "name", "unknown_tool"),
            tool_call_id=getattr(tool_call, "id", None),
            tool_call_arguments=_loads_json(getattr(tool_call, "input", None)),
            provider="agentscope",
        )
        handler.start_execute_tool(invocation)
        last_item = None
        tool_closed = False
        react_closed = False
        try:
            async for item in next_handler(**input_kwargs):
                last_item = item
                yield item
        except BaseException as exc:
            error = Error(
                message=str(exc) or type(exc).__name__, type=type(exc)
            )
            handler.fail_execute_tool(
                invocation,
                error,
            )
            tool_closed = True
            handler.fail_react_step(react_invocation, error)
            react_closed = True
            raise
        else:
            if isinstance(last_item, ToolResponse):
                invocation.tool_call_result = _jsonable(
                    _blocks_to_parts(last_item.content)
                )
            elif last_item is not None:
                invocation.tool_call_result = str(last_item)
            handler.stop_execute_tool(invocation)
            tool_closed = True
            react_invocation.finish_reason = "tool_calls"
            handler.stop_react_step(react_invocation)
            react_closed = True
        finally:
            if not tool_closed:
                handler.stop_execute_tool(invocation)
            if not react_closed:
                handler.stop_react_step(react_invocation)

    def _next_react_round(self) -> int:
        current = self._react_round.get() + 1
        self._react_round.set(current)
        return current


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
                    response=getattr(block, "output", ""),
                )
            )
    return parts


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
