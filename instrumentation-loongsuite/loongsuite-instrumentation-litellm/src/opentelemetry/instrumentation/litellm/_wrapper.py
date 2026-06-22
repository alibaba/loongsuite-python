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
Wrapper functions for LiteLLM completion instrumentation.
"""

import logging
import os
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
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)

# Environment variable to control instrumentation
ENABLE_LITELLM_INSTRUMENTOR = "ENABLE_LITELLM_INSTRUMENTOR"


def _is_instrumentation_enabled() -> bool:
    """Check if instrumentation is enabled via environment variable."""
    enabled = os.getenv(ENABLE_LITELLM_INSTRUMENTOR, "true").lower()
    return enabled != "false"


class CompletionWrapper:
    """Wrapper for litellm.completion()"""

    def __init__(self, handler, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    def __call__(self, *args, **kwargs):
        """Wrap litellm.completion()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return self.original_func(*args, **kwargs)

        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_func(*args, **kwargs)

        request_kwargs = normalize_litellm_completion_kwargs(
            self.original_func, args, kwargs
        )
        is_stream = request_kwargs.get("stream", False)

        # For streaming, enable usage tracking if not explicitly disabled
        # This ensures we get token usage information in the final chunk
        if is_stream and "stream_options" not in request_kwargs:
            kwargs["stream_options"] = {"include_usage": True}
            request_kwargs["stream_options"] = kwargs["stream_options"]

        # For streaming, we need special handling
        if is_stream:
            # Create invocation object
            invocation = create_llm_invocation_from_litellm(**request_kwargs)

            # Start LLM invocation
            self._handler.start_llm(invocation)

            try:
                # Call original function
                response = self.original_func(*args, **kwargs)

                # Wrap the streaming response
                # We pass invocation and handler so the callback can fill data and call stop_llm
                stream_wrapper = StreamWrapper(
                    stream=response,
                    span=invocation.span,  # For TTFT tracking
                    callback=None,
                    invocation=invocation,
                )
                stream_wrapper.callback = lambda span, last_chunk, error: (
                    self._handle_stream_end_with_handler(
                        invocation, last_chunk, error, stream_wrapper
                    )
                )
                response = stream_wrapper

                return response
            except Exception as e:
                # Fail LLM invocation
                self._handler.fail_llm(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

        else:
            # Create invocation object
            invocation = create_llm_invocation_from_litellm(**request_kwargs)

            # Start LLM invocation (handler creates and manages span)
            self._handler.start_llm(invocation)

            try:
                # Call original function
                response = self.original_func(*args, **kwargs)

                apply_litellm_llm_response_to_invocation(invocation, response)

                # End LLM invocation successfully (handler ends span and records metrics)
                self._handler.stop_llm(invocation)

                return response

            except Exception as e:
                # Fail LLM invocation (handler marks span as error)
                self._handler.fail_llm(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

    def _handle_stream_end_with_handler(
        self,
        invocation,
        last_chunk: Optional[Any],
        error: Optional[Exception],
        stream_wrapper: Optional[Any] = None,
    ):
        """Handle the end of a streaming response using Handler pattern."""

        try:
            if error:
                # Fail LLM invocation
                self._handler.fail_llm(
                    invocation, Error(message=str(error), type=type(error))
                )
                return

            if stream_wrapper and hasattr(
                stream_wrapper, "get_output_messages"
            ):
                output_messages = stream_wrapper.get_output_messages()
                if output_messages:
                    invocation.output_messages = output_messages

            if last_chunk:
                apply_litellm_llm_response_to_invocation(
                    invocation,
                    last_chunk,
                    include_output_messages=False,
                )

            if stream_wrapper and hasattr(stream_wrapper, "finish_reasons"):
                finish_reasons = stream_wrapper.finish_reasons()
            else:
                finish_reasons = extract_finish_reasons_from_litellm_response(
                    last_chunk
                )
            if finish_reasons:
                invocation.finish_reasons = finish_reasons

            # End LLM invocation successfully
            self._handler.stop_llm(invocation)

        except Exception as e:
            logger.debug(f"Error handling stream end with handler: {e}")
            # Try to fail gracefully
            try:
                self._handler.fail_llm(
                    invocation, Error(message=str(e), type=type(e))
                )
            except Exception as handler_error:
                # Swallow exceptions from telemetry failure reporting, but log them for diagnostics.
                logger.debug(
                    "Error while reporting LLM failure in _handle_stream_end_with_handler: %s",
                    handler_error,
                )


class AsyncCompletionWrapper:
    """Wrapper for litellm.acompletion()"""

    def __init__(self, handler, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    async def __call__(self, *args, **kwargs):
        """Wrap litellm.acompletion()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return await self.original_func(*args, **kwargs)

        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await self.original_func(*args, **kwargs)

        request_kwargs = normalize_litellm_completion_kwargs(
            self.original_func, args, kwargs
        )
        is_stream = request_kwargs.get("stream", False)

        # For streaming, enable usage tracking if not explicitly disabled
        if is_stream and "stream_options" not in request_kwargs:
            kwargs["stream_options"] = {"include_usage": True}
            request_kwargs["stream_options"] = kwargs["stream_options"]

        # For streaming, we need special handling
        if is_stream:
            # Create invocation object
            invocation = create_llm_invocation_from_litellm(**request_kwargs)

            # Start LLM invocation
            self._handler.start_llm(invocation)

            try:
                # Call original function
                response = await self.original_func(*args, **kwargs)

                # Wrap the async streaming response
                stream_wrapper = AsyncStreamWrapper(
                    stream=response,
                    span=invocation.span,  # For TTFT tracking
                    callback=None,
                    invocation=invocation,
                )
                stream_wrapper.callback = lambda span, last_chunk, error: (
                    self._handle_stream_end_with_handler(
                        invocation, last_chunk, error, stream_wrapper
                    )
                )
                response = stream_wrapper

                return response
            except Exception as e:
                # Fail LLM invocation
                self._handler.fail_llm(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

        else:
            # Non-streaming: use Handler pattern
            # Create invocation object
            invocation = create_llm_invocation_from_litellm(**request_kwargs)

            # Start LLM invocation
            self._handler.start_llm(invocation)

            try:
                # Call original function
                response = await self.original_func(*args, **kwargs)

                apply_litellm_llm_response_to_invocation(invocation, response)

                # End LLM invocation successfully
                self._handler.stop_llm(invocation)

                return response

            except Exception as e:
                # Fail LLM invocation
                self._handler.fail_llm(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

    def _handle_stream_end_with_handler(
        self,
        invocation,
        last_chunk: Optional[Any],
        error: Optional[Exception],
        stream_wrapper: Optional[Any] = None,
    ):
        """Handle the end of an async streaming response using Handler pattern."""
        # Reuse sync logic
        completion_wrapper = CompletionWrapper(self._handler, None)
        completion_wrapper._handle_stream_end_with_handler(
            invocation, last_chunk, error, stream_wrapper
        )
