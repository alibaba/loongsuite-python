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

from __future__ import annotations

import contextvars
import inspect
import timeit
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping

from opentelemetry import context as otel_context
from opentelemetry.util.genai.extended_types import ReactStepInvocation
from opentelemetry.util.genai.types import Error

from .utils import (
    create_agent_invocation,
    create_llm_invocation,
    create_tool_invocation,
    reset_current_agno_run_identity,
    set_current_agno_run_identity,
    update_agent_invocation_from_events,
    update_agent_invocation_from_response,
    update_llm_invocation_from_response,
    update_tool_invocation_from_response,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
    )


def bind_arguments(
    method: Callable[..., Any], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    method_signature = inspect.signature(method)
    bound_arguments = method_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return OrderedDict(
        {
            key: value
            for key, value in bound_arguments.arguments.items()
            if key != "self" and value is not None
        }
    )


def _is_streaming(instance: Any, kwargs: Mapping[str, Any]) -> bool:
    stream = kwargs.get("stream")
    if stream is None:
        stream = getattr(instance, "stream", False)
    return bool(stream)


def _finish_invocation(
    finish: Callable[..., Any], invocation: Any, *args: Any
) -> Any:
    return finish(invocation, *args)


def _error(exc: BaseException) -> Error:
    return Error(message=str(exc), type=type(exc))


def _is_stream_close(exc: BaseException) -> bool:
    return isinstance(exc, (GeneratorExit, StopIteration, StopAsyncIteration))


@dataclass
class _AgnoRunState:
    handler: ExtendedTelemetryHandler
    agent_context: Any
    react_round: int = 0
    active_step: ReactStepInvocation | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cache_read_tokens: int = 0
    llm_cache_write_tokens: int = 0


_AGNO_RUN_STATE: contextvars.ContextVar[_AgnoRunState | None] = (
    contextvars.ContextVar("agno_run_state", default=None)
)


def _current_run_state(
    handler: ExtendedTelemetryHandler,
) -> _AgnoRunState | None:
    state = _AGNO_RUN_STATE.get()
    if state is None or state.handler is not handler:
        return None
    return state


def _close_active_react_step(
    handler: ExtendedTelemetryHandler,
    state: _AgnoRunState,
    finish_reason: str,
) -> None:
    step = state.active_step
    if step is None:
        return
    state.active_step = None
    step.finish_reason = step.finish_reason or finish_reason
    _finish_invocation(handler.stop_react_step, step)


def _fail_active_react_step(
    handler: ExtendedTelemetryHandler,
    state: _AgnoRunState | None,
    error: Error,
) -> None:
    if state is None or state.active_step is None:
        return
    step = state.active_step
    state.active_step = None
    _finish_invocation(handler.fail_react_step, step, error)


def _start_next_react_step(
    handler: ExtendedTelemetryHandler,
    state: _AgnoRunState | None,
) -> None:
    if state is None:
        return
    if state.active_step is not None:
        _close_active_react_step(handler, state, "tool_calls")
    state.react_round += 1
    invocation = ReactStepInvocation(round=state.react_round)
    handler.start_react_step(invocation, context=state.agent_context)
    state.active_step = invocation


def _finish_react_step_after_llm(
    handler: ExtendedTelemetryHandler,
    state: _AgnoRunState | None,
    assistant_message: Any,
) -> None:
    if state is None or state.active_step is None:
        return
    if getattr(assistant_message, "tool_calls", None):
        state.active_step.finish_reason = "tool_calls"
        return
    _close_active_react_step(handler, state, "stop")


def _record_llm_usage(
    state: _AgnoRunState | None,
    invocation: Any,
) -> None:
    if state is None:
        return
    state.llm_input_tokens += invocation.input_tokens or 0
    state.llm_output_tokens += invocation.output_tokens or 0
    state.llm_cache_read_tokens += (
        invocation.usage_cache_read_input_tokens or 0
    )
    state.llm_cache_write_tokens += (
        invocation.usage_cache_creation_input_tokens or 0
    )


def _apply_agent_token_fallback(
    invocation: Any,
    state: _AgnoRunState | None,
) -> None:
    if state is None:
        return
    if invocation.input_tokens is None and state.llm_input_tokens:
        invocation.input_tokens = state.llm_input_tokens
    if invocation.output_tokens is None and state.llm_output_tokens:
        invocation.output_tokens = state.llm_output_tokens
    if (
        invocation.usage_cache_read_input_tokens is None
        and state.llm_cache_read_tokens
    ):
        invocation.usage_cache_read_input_tokens = state.llm_cache_read_tokens
    if (
        invocation.usage_cache_creation_input_tokens is None
        and state.llm_cache_write_tokens
    ):
        invocation.usage_cache_creation_input_tokens = (
            state.llm_cache_write_tokens
        )


class AgnoAgentWrapper:
    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            return wrapped(*args, **kwargs)
        if _is_streaming(instance, arguments):
            return self._run_stream(
                wrapped,
                instance,
                args,
                kwargs,
                arguments,
                otel_context.get_current(),
            )
        return self._run(wrapped, instance, args, kwargs, arguments)

    def _run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        arguments: Mapping[str, Any],
    ) -> Any:
        invocation = create_agent_invocation(instance, arguments)
        self._handler.start_invoke_agent(
            invocation, context=otel_context.get_current()
        )
        state = _AgnoRunState(
            handler=self._handler,
            agent_context=otel_context.get_current(),
        )
        state_token = _AGNO_RUN_STATE.set(state)
        identity_token = set_current_agno_run_identity(invocation.attributes)
        try:
            response = wrapped(*args, **kwargs)
            update_agent_invocation_from_response(invocation, response)
            _apply_agent_token_fallback(invocation, state)
            _close_active_react_step(self._handler, state, "stop")
            _finish_invocation(self._handler.stop_invoke_agent, invocation)
            return response
        except Exception as exc:
            error = _error(exc)
            _fail_active_react_step(self._handler, state, error)
            _finish_invocation(
                self._handler.fail_invoke_agent,
                invocation,
                error,
            )
            raise
        finally:
            reset_current_agno_run_identity(identity_token)
            _AGNO_RUN_STATE.reset(state_token)

    def _run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        arguments: Mapping[str, Any],
        parent_context: Any,
    ) -> Iterator[Any]:
        invocation = create_agent_invocation(instance, arguments)

        def generator() -> Iterator[Any]:
            events = []
            finalized = False
            error = None
            state = None
            state_token = None
            self._handler.start_invoke_agent(
                invocation, context=parent_context
            )
            identity_token = None
            state = _AgnoRunState(
                handler=self._handler,
                agent_context=otel_context.get_current(),
            )

            def enter_identity() -> None:
                nonlocal identity_token
                identity_token = set_current_agno_run_identity(
                    invocation.attributes
                )

            def exit_identity() -> None:
                nonlocal identity_token
                if identity_token is not None:
                    reset_current_agno_run_identity(identity_token)
                    identity_token = None

            def enter_run_state() -> None:
                nonlocal state_token
                state_token = _AGNO_RUN_STATE.set(state)

            def exit_run_state() -> None:
                nonlocal state_token
                if state_token is not None:
                    _AGNO_RUN_STATE.reset(state_token)
                    state_token = None

            enter_identity()
            enter_run_state()
            try:
                stream = wrapped(*args, **kwargs)
                for event in stream:
                    if invocation.monotonic_first_token_s is None:
                        invocation.monotonic_first_token_s = (
                            timeit.default_timer()
                        )
                    events.append(event)
                    # Do not expose run-local state to caller code while the
                    # stream chunk is yielded outside this wrapper.
                    exit_run_state()
                    exit_identity()
                    try:
                        yield event
                    finally:
                        enter_identity()
                        enter_run_state()
                update_agent_invocation_from_events(invocation, events)
                _apply_agent_token_fallback(invocation, state)
                _close_active_react_step(self._handler, state, "stop")
                _finish_invocation(self._handler.stop_invoke_agent, invocation)
                finalized = True
            except BaseException as exc:
                error = exc
                raise
            finally:
                try:
                    if not finalized:
                        if error is None or _is_stream_close(error):
                            update_agent_invocation_from_events(
                                invocation, events
                            )
                            _apply_agent_token_fallback(invocation, state)
                            _close_active_react_step(
                                self._handler, state, "stop"
                            )
                            _finish_invocation(
                                self._handler.stop_invoke_agent, invocation
                            )
                        else:
                            invoke_error = _error(error)
                            _fail_active_react_step(
                                self._handler, state, invoke_error
                            )
                            _finish_invocation(
                                self._handler.fail_invoke_agent,
                                invocation,
                                invoke_error,
                            )
                finally:
                    exit_run_state()
                    exit_identity()

        return generator()

    def arun(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            return wrapped(*args, **kwargs)
        if _is_streaming(instance, arguments):
            return self._arun_stream(
                wrapped,
                instance,
                args,
                kwargs,
                arguments,
                otel_context.get_current(),
            )
        return self._arun(
            wrapped,
            instance,
            args,
            kwargs,
            arguments,
            otel_context.get_current(),
        )

    async def _arun(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        arguments: Mapping[str, Any],
        parent_context: Any,
    ) -> Any:
        invocation = create_agent_invocation(instance, arguments)
        self._handler.start_invoke_agent(invocation, context=parent_context)
        state = _AgnoRunState(
            handler=self._handler,
            agent_context=otel_context.get_current(),
        )
        state_token = _AGNO_RUN_STATE.set(state)
        identity_token = set_current_agno_run_identity(invocation.attributes)
        try:
            response = wrapped(*args, **kwargs)
            if inspect.isawaitable(response):
                response = await response
            update_agent_invocation_from_response(invocation, response)
            _apply_agent_token_fallback(invocation, state)
            _close_active_react_step(self._handler, state, "stop")
            _finish_invocation(self._handler.stop_invoke_agent, invocation)
            return response
        except Exception as exc:
            error = _error(exc)
            _fail_active_react_step(self._handler, state, error)
            _finish_invocation(
                self._handler.fail_invoke_agent,
                invocation,
                error,
            )
            raise
        finally:
            reset_current_agno_run_identity(identity_token)
            _AGNO_RUN_STATE.reset(state_token)

    async def _arun_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        arguments: Mapping[str, Any],
        parent_context: Any,
    ) -> AsyncIterator[Any]:
        invocation = create_agent_invocation(instance, arguments)

        events = []
        finalized = False
        error = None
        state = None
        state_token = None
        self._handler.start_invoke_agent(invocation, context=parent_context)
        identity_token = None
        state = _AgnoRunState(
            handler=self._handler,
            agent_context=otel_context.get_current(),
        )

        def enter_identity() -> None:
            nonlocal identity_token
            identity_token = set_current_agno_run_identity(
                invocation.attributes
            )

        def exit_identity() -> None:
            nonlocal identity_token
            if identity_token is not None:
                reset_current_agno_run_identity(identity_token)
                identity_token = None

        def enter_run_state() -> None:
            nonlocal state_token
            state_token = _AGNO_RUN_STATE.set(state)

        def exit_run_state() -> None:
            nonlocal state_token
            if state_token is not None:
                _AGNO_RUN_STATE.reset(state_token)
                state_token = None

        enter_identity()
        enter_run_state()
        try:
            stream = wrapped(*args, **kwargs)
            if inspect.isawaitable(stream):
                stream = await stream
            async for event in stream:
                if invocation.monotonic_first_token_s is None:
                    invocation.monotonic_first_token_s = timeit.default_timer()
                events.append(event)
                # Do not expose run-local state to caller code while the
                # stream chunk is yielded outside this wrapper.
                exit_run_state()
                exit_identity()
                try:
                    yield event
                finally:
                    enter_identity()
                    enter_run_state()
            update_agent_invocation_from_events(invocation, events)
            _apply_agent_token_fallback(invocation, state)
            _close_active_react_step(self._handler, state, "stop")
            _finish_invocation(self._handler.stop_invoke_agent, invocation)
            finalized = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                if not finalized:
                    if error is None or _is_stream_close(error):
                        update_agent_invocation_from_events(invocation, events)
                        _apply_agent_token_fallback(invocation, state)
                        _close_active_react_step(self._handler, state, "stop")
                        _finish_invocation(
                            self._handler.stop_invoke_agent, invocation
                        )
                    else:
                        invoke_error = _error(error)
                        _fail_active_react_step(
                            self._handler, state, invoke_error
                        )
                        _finish_invocation(
                            self._handler.fail_invoke_agent,
                            invocation,
                            invoke_error,
                        )
            finally:
                exit_run_state()
                exit_identity()


class AgnoFunctionCallWrapper:
    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def execute(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if instance is None:
            return wrapped(*args, **kwargs)
        invocation = create_tool_invocation(instance)
        self._handler.start_execute_tool(
            invocation, context=otel_context.get_current()
        )
        try:
            response = wrapped(*args, **kwargs)
            update_tool_invocation_from_response(invocation, response)
            _finish_invocation(self._handler.stop_execute_tool, invocation)
            return response
        except Exception as exc:
            _finish_invocation(
                self._handler.fail_execute_tool,
                invocation,
                _error(exc),
            )
            raise

    async def aexecute(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if instance is None:
            return await wrapped(*args, **kwargs)
        invocation = create_tool_invocation(instance)
        self._handler.start_execute_tool(
            invocation, context=otel_context.get_current()
        )
        try:
            response = await wrapped(*args, **kwargs)
            update_tool_invocation_from_response(invocation, response)
            _finish_invocation(self._handler.stop_execute_tool, invocation)
            return response
        except Exception as exc:
            _finish_invocation(
                self._handler.fail_execute_tool,
                invocation,
                _error(exc),
            )
            raise


class AgnoModelWrapper:
    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def _start_llm_call(
        self,
        instance: Any,
        arguments: Mapping[str, Any],
    ) -> tuple[_AgnoRunState | None, Any]:
        state = _current_run_state(self._handler)
        _start_next_react_step(self._handler, state)
        invocation = create_llm_invocation(instance, arguments)
        self._handler.start_llm(invocation, context=otel_context.get_current())
        return state, invocation

    def _finish_llm_call(
        self,
        state: _AgnoRunState | None,
        invocation: Any,
        response: Any,
    ) -> None:
        update_llm_invocation_from_response(invocation, response)
        _record_llm_usage(state, invocation)
        _finish_invocation(self._handler.stop_llm, invocation)
        _finish_react_step_after_llm(self._handler, state, response)

    def _fail_llm_call(
        self,
        state: _AgnoRunState | None,
        invocation: Any,
        exc: BaseException,
    ) -> None:
        error = _error(exc)
        _finish_invocation(self._handler.fail_llm, invocation, error)
        _fail_active_react_step(self._handler, state, error)

    @staticmethod
    def _response_for_llm_finish(
        arguments: Mapping[str, Any],
        responses: list[Any] | None = None,
    ) -> Any:
        if responses:
            return _merge_model_responses(responses)
        assistant_message = arguments.get("assistant_message")
        if assistant_message is not None and (
            getattr(assistant_message, "metrics", None) is not None
            or getattr(assistant_message, "content", None) is not None
            or getattr(assistant_message, "reasoning_content", None)
            is not None
            or getattr(assistant_message, "tool_calls", None)
        ):
            return assistant_message
        return assistant_message

    def process_model_response(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            return wrapped(*args, **kwargs)
        state, invocation = self._start_llm_call(instance, arguments)
        try:
            response = wrapped(*args, **kwargs)
            self._finish_llm_call(
                state,
                invocation,
                self._response_for_llm_finish(arguments),
            )
            return response
        except BaseException as exc:
            self._fail_llm_call(state, invocation, exc)
            raise

    async def aprocess_model_response(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            return await wrapped(*args, **kwargs)
        state, invocation = self._start_llm_call(instance, arguments)
        try:
            response = await wrapped(*args, **kwargs)
            self._finish_llm_call(
                state,
                invocation,
                self._response_for_llm_finish(arguments),
            )
            return response
        except BaseException as exc:
            self._fail_llm_call(state, invocation, exc)
            raise

    def process_response_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Iterator[Any]:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            yield from wrapped(*args, **kwargs)
            return

        state, invocation = self._start_llm_call(instance, arguments)
        responses = []
        finalized = False
        error = None
        try:
            stream = wrapped(*args, **kwargs)
            for response in stream:
                if invocation.monotonic_first_token_s is None:
                    invocation.monotonic_first_token_s = timeit.default_timer()
                responses.append(response)
                yield response
            self._finish_llm_call(
                state,
                invocation,
                self._response_for_llm_finish(arguments, responses),
            )
            finalized = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if not finalized:
                if error is None or _is_stream_close(error):
                    self._finish_llm_call(
                        state,
                        invocation,
                        self._response_for_llm_finish(arguments, responses),
                    )
                else:
                    self._fail_llm_call(state, invocation, error)

    async def aprocess_response_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncIterator[Any]:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            async for response in wrapped(*args, **kwargs):
                yield response
            return

        state, invocation = self._start_llm_call(instance, arguments)
        responses = []
        finalized = False
        error = None
        try:
            stream = wrapped(*args, **kwargs)
            if inspect.isawaitable(stream):
                stream = await stream
            async for response in stream:
                if invocation.monotonic_first_token_s is None:
                    invocation.monotonic_first_token_s = timeit.default_timer()
                responses.append(response)
                yield response
            self._finish_llm_call(
                state,
                invocation,
                self._response_for_llm_finish(arguments, responses),
            )
            finalized = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if not finalized:
                if error is None or _is_stream_close(error):
                    self._finish_llm_call(
                        state,
                        invocation,
                        self._response_for_llm_finish(arguments, responses),
                    )
                else:
                    self._fail_llm_call(state, invocation, error)

    def run_function_calls(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Iterator[Any]:
        if instance is None:
            yield from wrapped(*args, **kwargs)
            return

        state = _current_run_state(self._handler)
        finalized = False
        error = None
        try:
            for response in wrapped(*args, **kwargs):
                yield response
            if state is not None:
                _close_active_react_step(self._handler, state, "tool_calls")
            finalized = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if not finalized and state is not None and state.active_step:
                if error is None or _is_stream_close(error):
                    _close_active_react_step(
                        self._handler, state, "tool_calls"
                    )
                else:
                    _fail_active_react_step(
                        self._handler,
                        state,
                        _error(error),
                    )

    async def arun_function_calls(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncIterator[Any]:
        if instance is None:
            async for response in wrapped(*args, **kwargs):
                yield response
            return

        state = _current_run_state(self._handler)
        finalized = False
        error = None
        try:
            async for response in wrapped(*args, **kwargs):
                yield response
            if state is not None:
                _close_active_react_step(self._handler, state, "tool_calls")
            finalized = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if not finalized and state is not None and state.active_step:
                if error is None or _is_stream_close(error):
                    _close_active_react_step(
                        self._handler, state, "tool_calls"
                    )
                else:
                    _fail_active_react_step(
                        self._handler,
                        state,
                        _error(error),
                    )

    def response(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            return wrapped(*args, **kwargs)
        invocation = create_llm_invocation(instance, arguments)
        self._handler.start_llm(invocation, context=otel_context.get_current())
        try:
            response = wrapped(*args, **kwargs)
            update_llm_invocation_from_response(invocation, response)
            _finish_invocation(self._handler.stop_llm, invocation)
            return response
        except Exception as exc:
            _finish_invocation(
                self._handler.fail_llm,
                invocation,
                _error(exc),
            )
            raise

    def response_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Iterator[Any]:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            yield from wrapped(*args, **kwargs)
            return

        invocation = create_llm_invocation(instance, arguments)
        self._handler.start_llm(invocation, context=otel_context.get_current())
        responses = []
        finalized = False
        error = None
        try:
            stream = wrapped(*args, **kwargs)
            for response in stream:
                if invocation.monotonic_first_token_s is None:
                    invocation.monotonic_first_token_s = timeit.default_timer()
                responses.append(response)
                yield response
            if responses:
                update_llm_invocation_from_response(
                    invocation, _merge_model_responses(responses)
                )
            _finish_invocation(self._handler.stop_llm, invocation)
            finalized = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if not finalized:
                if error is None or _is_stream_close(error):
                    if responses:
                        update_llm_invocation_from_response(
                            invocation, _merge_model_responses(responses)
                        )
                    _finish_invocation(self._handler.stop_llm, invocation)
                else:
                    _finish_invocation(
                        self._handler.fail_llm,
                        invocation,
                        _error(error),
                    )

    def aresponse(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            return wrapped(*args, **kwargs)
        invocation = create_llm_invocation(instance, arguments)
        parent_context = otel_context.get_current()

        async def coroutine() -> Any:
            self._handler.start_llm(invocation, context=parent_context)
            try:
                response = await wrapped(*args, **kwargs)
                update_llm_invocation_from_response(invocation, response)
                _finish_invocation(self._handler.stop_llm, invocation)
                return response
            except Exception as exc:
                _finish_invocation(
                    self._handler.fail_llm,
                    invocation,
                    _error(exc),
                )
                raise

        return coroutine()

    async def aresponse_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncIterator[Any]:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if instance is None:
            async for response in wrapped(*args, **kwargs):
                yield response
            return

        invocation = create_llm_invocation(instance, arguments)
        self._handler.start_llm(invocation, context=otel_context.get_current())
        responses = []
        finalized = False
        error = None
        try:
            stream = wrapped(*args, **kwargs)
            if inspect.isawaitable(stream):
                stream = await stream
            async for response in stream:
                if invocation.monotonic_first_token_s is None:
                    invocation.monotonic_first_token_s = timeit.default_timer()
                responses.append(response)
                yield response
            if responses:
                update_llm_invocation_from_response(
                    invocation, _merge_model_responses(responses)
                )
            _finish_invocation(self._handler.stop_llm, invocation)
            finalized = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if not finalized:
                if error is None or _is_stream_close(error):
                    if responses:
                        update_llm_invocation_from_response(
                            invocation, _merge_model_responses(responses)
                        )
                    _finish_invocation(self._handler.stop_llm, invocation)
                else:
                    _finish_invocation(
                        self._handler.fail_llm,
                        invocation,
                        _error(error),
                    )


def _merge_model_responses(responses: list[Any]) -> Any:
    if not responses:
        return None

    first = responses[0]
    if len(responses) == 1:
        return first

    content = []
    reasoning = []
    tool_calls = []
    for response in responses:
        value = getattr(response, "content", None)
        if value is not None:
            content.append(str(value))
        reasoning_value = getattr(response, "reasoning_content", None)
        if reasoning_value is not None:
            reasoning.append(str(reasoning_value))
        response_tool_calls = getattr(response, "tool_calls", None) or []
        tool_calls.extend(response_tool_calls)

    merged = SimpleNamespace(
        id=getattr(first, "id", None),
        model=getattr(first, "model", None),
        role=getattr(first, "role", None) or "assistant",
        content=getattr(first, "content", None),
        reasoning_content=getattr(first, "reasoning_content", None),
        tool_calls=tool_calls or getattr(first, "tool_calls", None),
        finish_reason=next(
            (
                getattr(response, "finish_reason", None)
                for response in reversed(responses)
                if getattr(response, "finish_reason", None) is not None
            ),
            None,
        ),
    )
    if content:
        merged.content = "".join(content)
    if reasoning:
        merged.reasoning_content = "".join(reasoning)

    usage_totals = {}
    for name, aliases in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
        "cache_read_tokens": ("cache_read_tokens",),
        "cache_write_tokens": ("cache_write_tokens",),
    }.items():
        delta_total = 0
        summary_total = None
        for response in responses:
            is_usage_summary = (
                getattr(response, "content", None) is None
                and getattr(response, "reasoning_content", None) is None
            )
            value = next(
                (
                    getattr(response, alias)
                    for alias in aliases
                    if getattr(response, alias, None) is not None
                ),
                None,
            )
            if value is None:
                usage = getattr(response, "response_usage", None)
                value = (
                    next(
                        (
                            getattr(usage, alias)
                            for alias in aliases
                            if getattr(usage, alias, None) is not None
                        ),
                        0,
                    )
                    if usage is not None
                    else 0
                )
            if is_usage_summary and value:
                summary_total = value
            else:
                delta_total += value or 0
        total = max(delta_total, summary_total or 0)
        usage_totals[name] = total
        if total:
            setattr(merged, name, total)
    merged.response_usage = SimpleNamespace(**usage_totals)
    return merged
