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

"""Wrapper functions for LiteLLM completion instrumentation."""

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from opentelemetry import context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.litellm._stream_wrapper import (
    AsyncStreamWrapper,
    StreamWrapper,
)
from opentelemetry.instrumentation.litellm._utils import (
    apply_litellm_llm_response_to_invocation,
    create_llm_invocation_from_litellm,
    extract_finish_reasons_from_litellm_response,
    normalize_litellm_completion_kwargs,
)
from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.types import Error

# Environment variable to control instrumentation
ENABLE_LITELLM_INSTRUMENTOR = "ENABLE_LITELLM_INSTRUMENTOR"


@dataclass
class _CompletionAdviceState:
    invocation: Any
    is_stream: bool


def _is_instrumentation_enabled() -> bool:
    """Check if instrumentation is enabled via environment variable."""
    enabled = os.getenv(ENABLE_LITELLM_INSTRUMENTOR, "true").lower()
    return enabled != "false"


@hook_advice("litellm", "prepare")
def _prepare_advice(
    handler: Any,
    original_func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _CompletionAdviceState:
    """Prepare probe state without owning the application call."""
    invocation = None
    added_stream_options = False
    try:
        request_kwargs = normalize_litellm_completion_kwargs(
            original_func, args, kwargs
        )
        is_stream = request_kwargs.get("stream", False)

        if is_stream and "stream_options" not in request_kwargs:
            kwargs["stream_options"] = {"include_usage": True}
            request_kwargs["stream_options"] = kwargs["stream_options"]
            added_stream_options = True

        invocation = create_llm_invocation_from_litellm(**request_kwargs)
        handler.start_llm(invocation)
        return _CompletionAdviceState(invocation, is_stream)
    except Exception:
        if added_stream_options:
            kwargs.pop("stream_options", None)
        if invocation is not None:
            handler.abandon_llm(invocation)
        raise


@hook_advice("litellm", "detach_stream_context")
def _detach_stream_context_advice(
    handler: Any, state: _CompletionAdviceState
) -> bool:
    """Detach in the creation context before stream ownership is transferred."""
    handler.detach_llm_context(state.invocation)
    return True


@hook_advice("litellm", "success")
def _success_advice(
    handler: Any,
    state: _CompletionAdviceState,
    response: Any,
) -> None:
    """Map a non-streaming response and finalize its telemetry."""
    try:
        apply_litellm_llm_response_to_invocation(state.invocation, response)
        handler.stop_llm(state.invocation)
    except Exception:
        handler.abandon_llm(state.invocation)
        raise


@hook_advice("litellm", "error")
def _error_advice(
    handler: Any,
    state: _CompletionAdviceState,
    error: BaseException,
) -> None:
    """Record an application failure without replacing that failure."""
    try:
        handler.fail_llm(
            state.invocation,
            Error(message=str(error), type=type(error)),
        )
    except Exception:
        handler.abandon_llm(state.invocation)
        raise


@hook_advice("litellm", "stream_success")
def _stream_success_advice(
    handler: Any,
    state: _CompletionAdviceState,
    last_chunk: Optional[Any],
    stream_wrapper: Any,
) -> None:
    """Map accumulated stream data and finalize its telemetry."""
    try:
        output_messages = stream_wrapper.get_output_messages()
        if output_messages:
            state.invocation.output_messages = output_messages

        if last_chunk:
            apply_litellm_llm_response_to_invocation(
                state.invocation,
                last_chunk,
                include_output_messages=False,
            )

        finish_reasons = stream_wrapper.finish_reasons()
        if not finish_reasons:
            finish_reasons = extract_finish_reasons_from_litellm_response(
                last_chunk
            )
        if finish_reasons:
            state.invocation.finish_reasons = finish_reasons

        handler.stop_llm(state.invocation)
    except Exception:
        handler.abandon_llm(state.invocation)
        raise


@hook_advice("litellm", "stream_wrap")
def _wrap_stream_advice(
    handler: Any,
    state: _CompletionAdviceState,
    response: Any,
    *,
    asynchronous: bool,
) -> Any:
    """Wrap a business stream while keeping iteration outside advice."""
    wrapper_type = AsyncStreamWrapper if asynchronous else StreamWrapper
    stream_wrapper = wrapper_type(
        stream=response,
        span=state.invocation.span,
        callback=None,
        invocation=state.invocation,
    )

    def finalize(
        _span: Any,
        last_chunk: Optional[Any],
        error: Optional[BaseException],
    ) -> None:
        if error is not None:
            _error_advice(handler, state, error)
        else:
            _stream_success_advice(handler, state, last_chunk, stream_wrapper)

    stream_wrapper.callback = finalize
    return stream_wrapper


@hook_advice("litellm", "abandon")
def _abandon_advice(handler: Any, state: _CompletionAdviceState) -> None:
    """End telemetry when a business result cannot be instrumented."""
    handler.abandon_llm(state.invocation)


class CompletionWrapper:
    """Wrapper for ``litellm.completion()``."""

    def __init__(self, handler: Any, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not _is_instrumentation_enabled() or context.get_value(
            _SUPPRESS_INSTRUMENTATION_KEY
        ):
            return self.original_func(*args, **kwargs)

        state = _prepare_advice(
            self._handler, self.original_func, args, kwargs
        )

        try:
            response = self.original_func(*args, **kwargs)
        except BaseException as error:
            if state is not None:
                _error_advice(self._handler, state, error)
            raise

        if state is None:
            return response

        if not state.is_stream:
            _success_advice(self._handler, state, response)
            return response

        if not _detach_stream_context_advice(self._handler, state):
            _abandon_advice(self._handler, state)
            return response

        stream_wrapper = _wrap_stream_advice(
            self._handler,
            state,
            response,
            asynchronous=False,
        )
        if stream_wrapper is None:
            _abandon_advice(self._handler, state)
            return response

        return stream_wrapper


class AsyncCompletionWrapper:
    """Wrapper for ``litellm.acompletion()``."""

    def __init__(self, handler: Any, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not _is_instrumentation_enabled() or context.get_value(
            _SUPPRESS_INSTRUMENTATION_KEY
        ):
            return await self.original_func(*args, **kwargs)

        state = _prepare_advice(
            self._handler, self.original_func, args, kwargs
        )

        try:
            response = await self.original_func(*args, **kwargs)
        except BaseException as error:
            if state is not None:
                _error_advice(self._handler, state, error)
            raise

        if state is None:
            return response

        if not state.is_stream:
            _success_advice(self._handler, state, response)
            return response

        if not _detach_stream_context_advice(self._handler, state):
            _abandon_advice(self._handler, state)
            return response

        stream_wrapper = _wrap_stream_advice(
            self._handler,
            state,
            response,
            asynchronous=True,
        )
        if stream_wrapper is None:
            _abandon_advice(self._handler, state)
            return response

        return stream_wrapper
