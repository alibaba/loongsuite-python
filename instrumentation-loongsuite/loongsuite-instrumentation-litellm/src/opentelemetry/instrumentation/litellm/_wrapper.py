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

import json
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
    create_llm_invocation_from_litellm,
    extract_output_from_litellm_response,
)
from opentelemetry.util.genai.types import (
    Error,
    OutputMessage,
    Text,
    ToolCall,
)

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

        # Extract request parameters
        is_stream = kwargs.get("stream", False)

        # For streaming, enable usage tracking if not explicitly disabled
        # This ensures we get token usage information in the final chunk
        if is_stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        # For streaming, we need special handling
        if is_stream:
            # Create invocation object
            invocation = create_llm_invocation_from_litellm(**kwargs)

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
                )
                stream_wrapper.callback = (
                    lambda span,
                    last_chunk,
                    error: self._handle_stream_end_with_handler(
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
            invocation = create_llm_invocation_from_litellm(**kwargs)

            # Start LLM invocation (handler creates and manages span)
            self._handler.start_llm(invocation)

            try:
                # Call original function
                response = self.original_func(*args, **kwargs)

                # Fill response data into invocation
                invocation.output_messages = (
                    extract_output_from_litellm_response(response)
                )

                # Extract token usage
                if hasattr(response, "usage") and response.usage:
                    invocation.input_tokens = getattr(
                        response.usage, "prompt_tokens", None
                    )
                    invocation.output_tokens = getattr(
                        response.usage, "completion_tokens", None
                    )

                # Extract response metadata
                if hasattr(response, "id"):
                    invocation.response_id = response.id
                if hasattr(response, "model"):
                    invocation.response_model_name = response.model

                # Extract finish reasons
                if hasattr(response, "choices") and response.choices:
                    finish_reasons = []
                    for choice in response.choices:
                        if (
                            hasattr(choice, "finish_reason")
                            and choice.finish_reason
                        ):
                            finish_reasons.append(choice.finish_reason)
                    if finish_reasons:
                        invocation.finish_reasons = finish_reasons

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

            # Construct output message from accumulated content
            parts = []
            if stream_wrapper and hasattr(
                stream_wrapper, "accumulated_content"
            ):
                full_content = "".join(stream_wrapper.accumulated_content)
                if full_content:
                    parts.append(Text(content=full_content))

                # Handle accumulated tool calls if any
                if (
                    hasattr(stream_wrapper, "accumulated_tool_calls")
                    and stream_wrapper.accumulated_tool_calls
                ):
                    for tc in stream_wrapper.accumulated_tool_calls:
                        if hasattr(tc, "function"):
                            # Parse arguments if it's a JSON string
                            arguments = getattr(tc.function, "arguments", "")
                            if isinstance(arguments, str) and arguments:
                                try:
                                    arguments = json.loads(arguments)
                                except Exception:
                                    # If arguments are not valid JSON, keep the original string
                                    pass

                            parts.append(
                                ToolCall(
                                    id=getattr(tc, "id", None),
                                    name=getattr(tc.function, "name", ""),
                                    arguments=arguments,
                                )
                            )

            # If we have parts, create output message
            if parts:
                invocation.output_messages = [
                    OutputMessage(
                        role="assistant", parts=parts, finish_reason="stop"
                    )
                ]

            # Extract token usage from last chunk
            if (
                last_chunk
                and hasattr(last_chunk, "usage")
                and last_chunk.usage
            ):
                invocation.input_tokens = getattr(
                    last_chunk.usage, "prompt_tokens", None
                )
                invocation.output_tokens = getattr(
                    last_chunk.usage, "completion_tokens", None
                )

            # Extract response metadata
            if last_chunk:
                if hasattr(last_chunk, "id"):
                    invocation.response_id = last_chunk.id
                if hasattr(last_chunk, "model"):
                    invocation.response_model_name = last_chunk.model

                # Extract finish_reason from last chunk's choice
                if hasattr(last_chunk, "choices") and last_chunk.choices:
                    finish_reasons = []
                    for choice in last_chunk.choices:
                        if (
                            hasattr(choice, "finish_reason")
                            and choice.finish_reason
                        ):
                            finish_reasons.append(choice.finish_reason)
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

        # Extract request parameters
        is_stream = kwargs.get("stream", False)

        # For streaming, enable usage tracking if not explicitly disabled
        if is_stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        # For streaming, we need special handling
        if is_stream:
            # Create invocation object
            invocation = create_llm_invocation_from_litellm(**kwargs)

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
                )
                stream_wrapper.callback = (
                    lambda span,
                    last_chunk,
                    error: self._handle_stream_end_with_handler(
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
            invocation = create_llm_invocation_from_litellm(**kwargs)

            # Start LLM invocation
            self._handler.start_llm(invocation)

            try:
                # Call original function
                response = await self.original_func(*args, **kwargs)

                # Fill response data into invocation
                invocation.output_messages = (
                    extract_output_from_litellm_response(response)
                )

                # Extract token usage
                if hasattr(response, "usage") and response.usage:
                    invocation.input_tokens = getattr(
                        response.usage, "prompt_tokens", None
                    )
                    invocation.output_tokens = getattr(
                        response.usage, "completion_tokens", None
                    )

                # Extract response metadata
                if hasattr(response, "id"):
                    invocation.response_id = response.id
                if hasattr(response, "model"):
                    invocation.response_model_name = response.model

                # Extract finish reasons
                if hasattr(response, "choices") and response.choices:
                    finish_reasons = []
                    for choice in response.choices:
                        if (
                            hasattr(choice, "finish_reason")
                            and choice.finish_reason
                        ):
                            finish_reasons.append(choice.finish_reason)
                    if finish_reasons:
                        invocation.finish_reasons = finish_reasons

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
