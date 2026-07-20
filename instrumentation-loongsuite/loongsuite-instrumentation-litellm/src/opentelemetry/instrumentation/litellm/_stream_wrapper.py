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
Stream wrapper for LiteLLM streaming responses.
"""

import logging
import timeit
from typing import Any, Iterator, Optional

from opentelemetry.instrumentation.litellm._utils import (
    extract_litellm_text_parts,
    get_litellm_value,
    parse_tool_call_arguments,
)
from opentelemetry.instrumentation.loongsuite import call_advice
from opentelemetry.util.genai.types import (
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
)

logger = logging.getLogger(__name__)


class _StreamAccumulator:
    """Accumulate LiteLLM streaming deltas into GenAI output messages."""

    def __init__(self, invocation: Any = None):
        self.invocation = invocation
        self._choice_states: dict[int, dict[str, Any]] = {}

    def record_chunk(self, chunk: Any) -> None:
        choices = get_litellm_value(chunk, "choices") or []
        if not choices:
            return

        saw_token = False
        for default_index, choice in enumerate(choices):
            index = get_litellm_value(choice, "index", default_index)
            if not isinstance(index, int):
                index = default_index

            state = self._choice_states.setdefault(
                index,
                {
                    "role": "assistant",
                    "reasoning": [],
                    "content": [],
                    "finish_reason": None,
                    "tool_calls": {},
                },
            )

            finish_reason = get_litellm_value(choice, "finish_reason")
            if finish_reason:
                state["finish_reason"] = finish_reason

            delta = get_litellm_value(choice, "delta")
            if delta is None:
                continue

            role = get_litellm_value(delta, "role")
            if role:
                state["role"] = role

            content = get_litellm_value(delta, "content")
            content_parts = extract_litellm_text_parts(content)
            if content_parts:
                state["content"].extend(content_parts)
                saw_token = True

            reasoning_content = get_litellm_value(delta, "reasoning_content")
            if reasoning_content is None:
                reasoning_content = get_litellm_value(delta, "reasoning")
            reasoning_parts = extract_litellm_text_parts(reasoning_content)
            if reasoning_parts:
                state["reasoning"].extend(reasoning_parts)
                saw_token = True

            tool_calls = get_litellm_value(delta, "tool_calls")
            if tool_calls:
                saw_token = True
                self._record_tool_calls(state, tool_calls)

        if saw_token and self.invocation is not None:
            first_token_time = getattr(
                self.invocation, "monotonic_first_token_s", None
            )
            if first_token_time is None:
                self.invocation.monotonic_first_token_s = (
                    timeit.default_timer()
                )

    def get_output_messages(self) -> list[OutputMessage]:
        output_messages = []
        for index in sorted(self._choice_states):
            state = self._choice_states[index]
            parts = []
            reasoning = "".join(state["reasoning"])
            if reasoning:
                parts.append(Reasoning(content=reasoning))

            content = "".join(state["content"])
            if content:
                parts.append(Text(content=content))

            for tool_index in sorted(state["tool_calls"]):
                tool_call = state["tool_calls"][tool_index]
                arguments = parse_tool_call_arguments(
                    tool_call.get("arguments", "")
                )
                if (
                    tool_call.get("id")
                    or tool_call.get("name")
                    or arguments not in (None, "")
                ):
                    parts.append(
                        ToolCall(
                            id=tool_call.get("id"),
                            name=tool_call.get("name", ""),
                            arguments=arguments,
                        )
                    )

            if not parts:
                parts.append(Text(content=""))

            output_messages.append(
                OutputMessage(
                    role=state["role"] or "assistant",
                    parts=parts,
                    finish_reason=state["finish_reason"] or "stop",
                )
            )
        return output_messages

    def finish_reasons(self) -> list[str]:
        finish_reasons = []
        for index in sorted(self._choice_states):
            state = self._choice_states[index]
            if state["finish_reason"]:
                finish_reasons.append(state["finish_reason"])
        return finish_reasons

    @staticmethod
    def _record_tool_calls(
        state: dict[str, Any], tool_calls: list[Any]
    ) -> None:
        for fallback_index, tool_call in enumerate(tool_calls):
            tool_index = get_litellm_value(tool_call, "index", fallback_index)
            if not isinstance(tool_index, int):
                tool_index = fallback_index

            stored = state["tool_calls"].setdefault(
                tool_index,
                {"id": None, "name": "", "arguments": ""},
            )

            tool_id = get_litellm_value(tool_call, "id")
            if tool_id:
                stored["id"] = tool_id

            function = get_litellm_value(tool_call, "function")
            function_name = get_litellm_value(function, "name")
            if function_name:
                stored["name"] = function_name

            arguments = get_litellm_value(function, "arguments")
            if isinstance(arguments, str):
                stored["arguments"] += arguments
            elif arguments:
                logger.debug(
                    "Skipping non-string LiteLLM streamed tool-call arguments"
                )


class StreamWrapper:
    """
    Wrapper for synchronous streaming responses.
    Note: To avoid memory leaks, we only keep the last chunk instead of all chunks.
    This is sufficient for extracting usage information which is typically in the last chunk.

    Supports context manager protocol for reliable cleanup.
    """

    _warned_unclosed_stream = False

    def __init__(
        self,
        stream: Iterator,
        span: Any,
        callback: callable,
        invocation: Any = None,
    ):
        self.stream = stream
        self.span = span
        self.callback = callback
        self._accumulator = _StreamAccumulator(invocation)
        self.last_chunk = None  # Only keep last chunk to avoid memory leak
        self.chunk_count = 0
        self._finalized = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)

            call_advice(
                self._accumulator.record_chunk,
                chunk,
                instrumentation_name="litellm",
                advice_method="stream_chunk",
            )

            # Only keep the last chunk (contains usage info)
            self.last_chunk = chunk
            self.chunk_count += 1

            return chunk
        except StopIteration:
            # Stream ended normally, finalize span
            self._finalize()
            raise
        except Exception as e:
            # Error during streaming
            logger.debug(f"Error during streaming: {e}")
            self._finalize(error=e)
            raise

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure finalization on context exit."""
        if exc_type is not None:
            # Exception occurred during iteration
            self._finalize(error=exc_val)
        else:
            # Normal exit (may have completed or early terminated)
            self._finalize()
        return False

    def close(self):
        """Explicitly close and finalize the stream."""
        self._finalize()

    def __del__(self):
        if getattr(self, "_finalized", True):
            return

        if not StreamWrapper._warned_unclosed_stream:
            StreamWrapper._warned_unclosed_stream = True
            logger.warning(
                "LiteLLM stream wrapper was garbage-collected before close; "
                "finalizing the span. Use a context manager or call close() "
                "when terminating streams early."
            )

        try:
            self._finalize()
        except Exception as exc:
            logger.debug("Error finalizing unclosed LiteLLM stream: %s", exc)

    def _close_stream(self) -> None:
        close = getattr(self.stream, "close", None)
        if not callable(close):
            return

        try:
            close()
        except Exception as exc:
            logger.debug("Error closing LiteLLM stream: %s", exc)

    def _finalize(self, error: Optional[Exception] = None):
        """Finalize the span with data from last chunk."""
        if self._finalized:
            return

        self._finalized = True
        self._close_stream()
        try:
            # Call the callback with only the last chunk
            # Note: The callback is responsible for calling handler.stop_llm()
            # or handler.fail_llm().
            # which will end the span. We no longer call span.end() here.
            if self.callback:
                self.callback(self.span, self.last_chunk, error)

            # Clear reference to avoid holding memory
            self.last_chunk = None
        except Exception as e:
            logger.debug(f"Error finalizing stream: {e}")

    def get_output_messages(self) -> list[OutputMessage]:
        return self._accumulator.get_output_messages()

    def finish_reasons(self) -> list[str]:
        return self._accumulator.finish_reasons()


class AsyncStreamWrapper:
    """
    Wrapper for asynchronous streaming responses.
    Note: To avoid memory leaks, we only keep the last chunk instead of all chunks.
    This is sufficient for extracting usage information which is typically in the last chunk.

    Important: AsyncStreamWrapper must be consumed within an async context that ensures
    finalization, either by:
    1. Using as an async context manager: async with response: ...
    2. Explicitly calling close() after iteration
    3. Letting the wrapper detect stream exhaustion
    """

    def __init__(
        self,
        stream,
        span: Any,
        callback: callable,
        invocation: Any = None,
    ):
        self.stream = stream
        self.span = span
        self.callback = callback
        self._accumulator = _StreamAccumulator(invocation)
        self.last_chunk = None  # Only keep last chunk to avoid memory leak
        self.chunk_count = 0
        self._finalized = False
        self._stream_exhausted = False

    def __aiter__(self):
        # Return an async generator that wraps the stream and ensures finalization
        return self._wrapped_iteration()

    async def _wrapped_iteration(self):
        """
        Async generator that wraps the underlying stream and ensures finalization.
        This approach guarantees that _finalize() is called when:
        1. The stream is exhausted normally
        2. An exception occurs
        3. The generator is closed early (via aclose())
        """
        error = None
        try:
            async for chunk in self.stream:
                call_advice(
                    self._accumulator.record_chunk,
                    chunk,
                    instrumentation_name="litellm",
                    advice_method="stream_chunk",
                )

                # Only keep the last chunk (contains usage info)
                self.last_chunk = chunk
                self.chunk_count += 1

                yield chunk

            # Stream exhausted normally
            logger.debug(
                f"AsyncStreamWrapper: Stream completed (chunks: {self.chunk_count})"
            )
        except Exception as e:
            # Error during streaming
            logger.debug(f"AsyncStreamWrapper: Error during streaming: {e}")
            error = e
            raise
        finally:
            # Always finalize, whether completed normally, with error, or closed early
            await self._aclose_stream()
            self._finalize(error=error)

    async def __aenter__(self):
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure finalization on async context exit."""
        await self._aclose_stream()
        if exc_type is not None:
            # Exception occurred during iteration
            self._finalize(error=exc_val)
        else:
            # Normal exit (may have completed or early terminated)
            self._finalize()
        return False

    async def aclose(self):
        """Explicitly close and finalize the async stream."""
        await self._aclose_stream()
        self._finalize()

    def close(self):
        """Synchronous close method for compatibility."""
        self._close_stream()
        self._finalize()

    def _close_stream(self) -> None:
        close = getattr(self.stream, "close", None)
        if not callable(close):
            return

        try:
            close()
        except Exception as exc:
            logger.debug("Error closing LiteLLM async stream: %s", exc)

    async def _aclose_stream(self) -> None:
        aclose = getattr(self.stream, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
                return
            except Exception as exc:
                logger.debug("Error closing LiteLLM async stream: %s", exc)

        self._close_stream()

    def _finalize(self, error: Optional[Exception] = None):
        """Finalize the span with data from last chunk."""
        if self._finalized:
            return

        self._finalized = True
        try:
            # Call the callback with only the last chunk
            # Note: The callback is responsible for calling handler.stop_llm()
            # or handler.fail_llm().
            # which will end the span. We no longer call span.end() here.
            if self.callback:
                try:
                    self.callback(self.span, self.last_chunk, error)
                except Exception as callback_error:
                    logger.debug(f"Error in stream callback: {callback_error}")

            # Clear reference to avoid holding memory
            self.last_chunk = None
        except Exception as e:
            logger.debug(f"Error finalizing async stream: {e}")

    def get_output_messages(self) -> list[OutputMessage]:
        return self._accumulator.get_output_messages()

    def finish_reasons(self) -> list[str]:
        return self._accumulator.finish_reasons()
