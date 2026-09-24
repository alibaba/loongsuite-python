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

"""Wrap QwenPaw request streams with task-safe LoongSuite Entry telemetry."""

from __future__ import annotations

import asyncio
import logging
import timeit
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.handler import _safe_detach
from opentelemetry.util.genai.types import Error, OutputMessage

from ._entry_utils import (
    build_entry_invocation,
    build_runtime_entry_invocation,
    output_message_from_runtime_item,
    output_message_from_yield_item,
    parse_query_handler_call,
    parse_runtime_call,
)

logger = logging.getLogger(__name__)


@hook_advice("qwenpaw", "record_cancellation")
def _record_cancellation(
    invocation: EntryInvocation, error: BaseException | None
) -> None:
    if (
        not isinstance(error, asyncio.CancelledError)
        or invocation.span is None
    ):
        return
    # Only low-cardinality reason codes supplied by the caller are captured.
    # Never infer a cause from partial output or export arbitrary exception text.
    reason = error.args[0] if error.args else None
    if not isinstance(reason, str) or reason not in {
        "client_stop",
        "session_closed",
        "session_reset",
        "reset",
        "matrix_orchestration_failed",
    }:
        reason = "unknown"
    invocation.span.set_attribute("qwenpaw.cancelled", True)
    invocation.span.set_attribute("qwenpaw.cancellation.reason", reason)


@dataclass
class _EntryState:
    handler: ExtendedTelemetryHandler
    invocation: EntryInvocation
    context: Context
    saw_first_token: bool = False
    last_assistant: OutputMessage | None = None
    finalized: bool = False


@hook_advice("qwenpaw", "build_entry")
def _build_entry(
    factory: Callable[..., EntryInvocation],
    *args: Any,
) -> EntryInvocation:
    return factory(*args)


def _build_query_handler_entry(
    instance: Any,
    args: Any,
    kwargs: Any,
) -> EntryInvocation:
    msgs, request = parse_query_handler_call(args, kwargs)
    return build_entry_invocation(instance, msgs, request)


def _build_runtime_entry(
    instance: Any,
    args: Any,
    kwargs: Any,
) -> EntryInvocation:
    request = parse_runtime_call(args, kwargs)
    return build_runtime_entry_invocation(instance, request)


@hook_advice("qwenpaw", "start_entry")
def _start_entry(
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
) -> _EntryState:
    """Start Entry, save its Context, then detach before stream ownership moves."""

    try:
        handler.start_entry(invocation)
        entry_context = otel_context.get_current()
    except Exception:
        token = invocation.context_token
        invocation.context_token = None
        _safe_detach(token)
        span = invocation.span
        if span is not None and span.is_recording():
            span.end()
        raise

    token = invocation.context_token
    invocation.context_token = None
    _safe_detach(token)
    return _EntryState(
        handler=handler,
        invocation=invocation,
        context=entry_context,
    )


@hook_advice("qwenpaw", "attach_entry")
def _attach_entry(state: _EntryState) -> object:
    return otel_context.attach(state.context)


@hook_advice("qwenpaw", "detach_entry")
def _detach_entry(token: object | None) -> None:
    _safe_detach(token)


@hook_advice("qwenpaw", "record_entry_first_token")
def _record_entry_first_token(
    state: _EntryState,
    item: Any,
    first_token_predicate: Callable[[Any], bool],
) -> None:
    if not state.saw_first_token and first_token_predicate(item):
        state.invocation.response_time_to_first_token = int(
            (timeit.default_timer() - state.invocation.monotonic_start_s)
            * 1_000_000_000
        )
        state.saw_first_token = True


@hook_advice("qwenpaw", "record_entry_output")
def _record_entry_output(
    state: _EntryState,
    item: Any,
    output_mapper: Callable[[Any], OutputMessage | None],
) -> None:
    output = output_mapper(item)
    if output is not None:
        state.last_assistant = output


def _is_query_handler_first_token(item: Any) -> bool:
    del item
    return True


def _is_runtime_first_token(item: Any) -> bool:
    """Ignore QwenPaw protocol envelopes until user-visible output arrives."""

    if getattr(item, "delta", False) is True:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return True

        data = getattr(item, "data", None)
        if isinstance(data, dict):
            arguments = data.get("arguments")
            if isinstance(arguments, str) and arguments:
                return True

    return output_message_from_runtime_item(item) is not None


@hook_advice("qwenpaw", "finish_entry")
def _finish_entry(
    state: _EntryState,
    error: BaseException | None = None,
) -> None:
    """Finalize once, with cleanup even if handler post-processing fails."""

    if state.finalized:
        return
    state.finalized = True

    invocation = state.invocation
    if state.last_assistant is not None:
        invocation.output_messages = [state.last_assistant]

    token = None
    span = invocation.span
    try:
        token = otel_context.attach(state.context)
        invocation.context_token = token
        _record_cancellation(invocation, error)
        if error is None or isinstance(
            error, (GeneratorExit, asyncio.CancelledError)
        ):
            state.handler.stop_entry(invocation)
        else:
            state.handler.fail_entry(
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


async def _close_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


async def _entry_stream(
    wrapped: Any,
    args: Any,
    kwargs: Any,
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
    first_token_predicate: Callable[[Any], bool],
    output_mapper: Callable[[Any], OutputMessage | None],
) -> AsyncIterator[Any]:
    """Advance one business item per same-Context Entry attach/detach pair."""

    business_stream = wrapped(*args, **kwargs)
    iterator = business_stream.__aiter__()
    state = _start_entry(handler, invocation)
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
            token = _attach_entry(state)
            try:
                item = await iterator.__anext__()
            except StopAsyncIteration:
                break
            finally:
                _detach_entry(token)

            _record_entry_first_token(state, item, first_token_predicate)
            _record_entry_output(state, item, output_mapper)
            yield item
    except GeneratorExit as exc:
        token = _attach_entry(state)
        try:
            await _close_iterator(iterator)
        except BaseException as close_error:
            _finish_entry(state, close_error)
            raise
        finally:
            _detach_entry(token)
        _finish_entry(state, exc)
        raise
    except BaseException as exc:
        _finish_entry(state, exc)
        raise
    finally:
        if not state.finalized:
            _finish_entry(state)


def make_query_handler_wrapper(
    handler: ExtendedTelemetryHandler,
    module_name: str,
) -> Callable[..., Any]:
    """Wrap QwenPaw 1 / CoPaw ``AgentRunner.query_handler``."""

    def query_handler_wrapper(
        wrapped: Any,
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        invocation = _build_entry(
            _build_query_handler_entry,
            instance,
            args,
            kwargs,
        )
        if invocation is None:
            return wrapped(*args, **kwargs)
        logger.debug("Tracing %s.AgentRunner.query_handler", module_name)
        return _entry_stream(
            wrapped,
            args,
            kwargs,
            handler,
            invocation,
            _is_query_handler_first_token,
            output_message_from_yield_item,
        )

    return query_handler_wrapper


def make_runtime_wrapper(
    handler: ExtendedTelemetryHandler,
    module_name: str,
) -> Callable[..., Any]:
    """Wrap QwenPaw 2 ``Runtime.run`` as one Entry span per request."""

    def runtime_wrapper(
        wrapped: Any,
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        invocation = _build_entry(
            _build_runtime_entry,
            instance,
            args,
            kwargs,
        )
        if invocation is None:
            return wrapped(*args, **kwargs)
        logger.debug("Tracing %s.Runtime.run", module_name)
        return _entry_stream(
            wrapped,
            args,
            kwargs,
            handler,
            invocation,
            _is_runtime_first_token,
            output_message_from_runtime_item,
        )

    return runtime_wrapper
