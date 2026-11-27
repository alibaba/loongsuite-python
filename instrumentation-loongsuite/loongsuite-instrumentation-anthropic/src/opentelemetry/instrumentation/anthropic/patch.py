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

"""Patching functions for Anthropic instrumentation."""

from __future__ import annotations

import asyncio
import json
import logging
import timeit
from typing import TYPE_CHECKING, Any, Callable, Optional

from wrapt import ObjectProxy

from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    Error,
    LLMInvocation,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
)

from .utils import (
    create_anthropic_invocation,
    is_streaming,
    populate_response,
    total_input_tokens,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anthropic.resources.messages import AsyncMessages, Messages
    from anthropic.types import Message


def _fail_llm(handler, invocation, error):
    try:
        handler.fail_llm(
            invocation, Error(type=type(error), message=type(error).__name__)
        )
    except Exception:
        logger.debug("Could not record Anthropic error", exc_info=True)


def messages_create(
    handler: TelemetryHandler,
    content_capturing_mode: ContentCapturingMode,
) -> Callable[..., "Message"]:
    """Wrap the `create` method of the `Messages` class to trace it."""
    capture_content = content_capturing_mode != ContentCapturingMode.NO_CONTENT

    def traced_method(
        wrapped: Callable[..., "Message"],
        instance: "Messages",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> "Message":
        if not is_instrumentation_enabled():
            return wrapped(*args, **kwargs)
        try:
            invocation = handler.start_llm(
                create_anthropic_invocation(kwargs, instance, capture_content)
            )
        except Exception:
            logger.debug("Could not start Anthropic telemetry", exc_info=True)
            return wrapped(*args, **kwargs)

        try:
            result = wrapped(*args, **kwargs)
        except (Exception, asyncio.CancelledError) as error:
            _fail_llm(handler, invocation, error)
            raise

        if is_streaming(kwargs):
            try:
                handler.detach_llm_context(invocation)
                return AnthropicStreamWrapper(
                    result, handler, invocation, capture_content
                )
            except Exception:
                _abandon_llm(handler, invocation)
                logger.debug("Could not wrap Anthropic stream", exc_info=True)
                return result

        try:
            populate_response(invocation, result, capture_content)
            handler.stop_llm(invocation)
        except Exception:
            _abandon_llm(handler, invocation)
            logger.debug(
                "Could not finalize Anthropic telemetry", exc_info=True
            )
        return result

    return traced_method


def async_messages_create(
    handler: TelemetryHandler,
    content_capturing_mode: ContentCapturingMode,
) -> Callable[..., "Message"]:
    """Wrap the `create` method of the `AsyncMessages` class to trace it."""
    capture_content = content_capturing_mode != ContentCapturingMode.NO_CONTENT

    async def traced_method(
        wrapped: Callable[..., "Message"],
        instance: "AsyncMessages",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> "Message":
        if not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)
        try:
            invocation = handler.start_llm(
                create_anthropic_invocation(kwargs, instance, capture_content)
            )
        except Exception:
            logger.debug("Could not start Anthropic telemetry", exc_info=True)
            return await wrapped(*args, **kwargs)

        try:
            result = await wrapped(*args, **kwargs)
        except (Exception, asyncio.CancelledError) as error:
            _fail_llm(handler, invocation, error)
            raise

        if is_streaming(kwargs):
            try:
                handler.detach_llm_context(invocation)
                return AsyncAnthropicStreamWrapper(
                    result, handler, invocation, capture_content
                )
            except Exception:
                _abandon_llm(handler, invocation)
                logger.debug("Could not wrap Anthropic stream", exc_info=True)
                return result

        try:
            populate_response(invocation, result, capture_content)
            handler.stop_llm(invocation)
        except Exception:
            _abandon_llm(handler, invocation)
            logger.debug(
                "Could not finalize Anthropic telemetry", exc_info=True
            )
        return result

    return traced_method


class _ContentBlockAccumulator:
    """Keep each supported block in wire order, including partial blocks."""

    def __init__(self):
        self._blocks = []
        self._current = None

    def on_content_block_start(self, content_block):
        block = {
            "type": _value(content_block, "type", None),
            "text": [_value(content_block, "text", "")],
            "thinking": [_value(content_block, "thinking", "")],
            "name": _value(content_block, "name", ""),
            "id": _value(content_block, "id", None),
            "input": _value(content_block, "input", None),
            "json": [],
        }
        self._current = block
        self._blocks.append(block)

    def on_content_block_delta(self, delta):
        if self._current is None:
            return
        delta_type = _value(delta, "type", None)
        if delta_type == "text_delta":
            self._current["text"].append(_value(delta, "text", ""))
        elif delta_type == "thinking_delta":
            self._current["thinking"].append(_value(delta, "thinking", ""))
        elif delta_type == "input_json_delta":
            self._current["json"].append(_value(delta, "partial_json", ""))

    def on_content_block_stop(self):
        self._current = None

    def build_output_messages(self, stop_reason):
        parts: list[MessagePart] = []
        for block in self._blocks:
            if block["type"] == "text":
                parts.append(Text(content="".join(block["text"])))
            elif block["type"] == "thinking":
                parts.append(Reasoning(content="".join(block["thinking"])))
            elif block["type"] == "tool_use":
                arguments = block["input"]
                if block["json"]:
                    raw_json = "".join(block["json"])
                    try:
                        arguments = json.loads(raw_json)
                    except (json.JSONDecodeError, ValueError):
                        arguments = raw_json
                parts.append(
                    ToolCall(
                        name=block["name"], id=block["id"], arguments=arguments
                    )
                )
        if not parts:
            return []
        return [
            OutputMessage(
                role="assistant", parts=parts, finish_reason=stop_reason
            )
        ]


class AnthropicStreamWrapper:
    """Wraps an Anthropic streaming response to capture telemetry."""

    def __init__(
        self,
        stream: Any,
        handler: TelemetryHandler,
        invocation: LLMInvocation,
        capture_content: bool,
    ):
        self.stream = stream
        self.handler = handler
        self.invocation = invocation
        self.capture_content = capture_content
        self._accumulator = _ContentBlockAccumulator()
        self._started = True
        self._response_model: str | None = None
        self._response_id: str | None = None
        self._stop_reason: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._cache_creation_input_tokens: int | None = None
        self._cache_read_input_tokens: int | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self.stream.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._cleanup(exc_val)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self.stream)
            try:
                self._process_event(event)
            except Exception:
                logger.debug(
                    "Could not record Anthropic stream event", exc_info=True
                )
            return event
        except StopIteration:
            self._cleanup()
            raise
        except (Exception, asyncio.CancelledError) as error:
            self._cleanup(error)
            raise

    def close(self):
        try:
            self.stream.close()
        finally:
            self._cleanup()

    def _process_event(self, event: Any) -> None:
        """Process a single streaming event."""
        event_type = getattr(event, "type", None)

        # Record time to first token
        if self.invocation.monotonic_first_token_s is None and event_type in (
            "content_block_delta",
        ):
            self.invocation.monotonic_first_token_s = timeit.default_timer()

        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message:
                self._response_model = getattr(message, "model", None)
                self._response_id = getattr(message, "id", None)
                usage = getattr(message, "usage", None)
                if usage:
                    self._input_tokens = _value(usage, "input_tokens", None)
                    if (
                        _value(usage, "cache_creation_input_tokens", None)
                        is not None
                    ):
                        self._cache_creation_input_tokens = _value(
                            usage, "cache_creation_input_tokens", None
                        )
                    if (
                        _value(usage, "cache_read_input_tokens", None)
                        is not None
                    ):
                        self._cache_read_input_tokens = _value(
                            usage, "cache_read_input_tokens", None
                        )

        elif event_type == "content_block_start":
            content_block = getattr(event, "content_block", None)
            if content_block and self.capture_content:
                self._accumulator.on_content_block_start(content_block)

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and self.capture_content:
                self._accumulator.on_content_block_delta(delta)

        elif event_type == "content_block_stop":
            if self.capture_content:
                self._accumulator.on_content_block_stop()

        elif event_type == "message_delta":
            delta = getattr(event, "delta", None)
            if delta:
                self._stop_reason = _value(delta, "stop_reason", None)
            usage = getattr(event, "usage", None)
            if usage:
                self._output_tokens = _value(usage, "output_tokens", None)

        elif event_type == "message_stop":
            pass  # Cleanup happens in __next__ StopIteration or __exit__

    def _cleanup(self, error: Optional[BaseException] = None) -> None:
        """Finalize the invocation with accumulated data."""
        if not self._started:
            return
        self._started = False
        try:
            self._finish(error)
        except Exception:
            _abandon_llm(self.handler, self.invocation)
            logger.debug(
                "Could not finalize Anthropic stream telemetry", exc_info=True
            )

    def _finish(self, error: Optional[BaseException]) -> None:
        self.invocation.response_model_name = self._response_model
        self.invocation.response_id = self._response_id
        self.invocation.input_tokens = total_input_tokens(
            self._input_tokens,
            self._cache_creation_input_tokens,
            self._cache_read_input_tokens,
        )
        self.invocation.output_tokens = self._output_tokens

        if self._cache_creation_input_tokens is not None:
            self.invocation.usage_cache_creation_input_tokens = (
                self._cache_creation_input_tokens
            )
        if self._cache_read_input_tokens is not None:
            self.invocation.usage_cache_read_input_tokens = (
                self._cache_read_input_tokens
            )

        finish_reason = self._stop_reason
        if finish_reason is None and error is not None:
            finish_reason = "error"
        if finish_reason is not None:
            self.invocation.finish_reasons = [finish_reason]

        if self.capture_content:
            self.invocation.output_messages = (
                self._accumulator.build_output_messages(finish_reason)
            )

        if error is not None:
            self.handler.fail_llm(
                self.invocation,
                Error(type=type(error), message=type(error).__name__),
            )
        else:
            self.handler.stop_llm(self.invocation)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying stream."""
        return getattr(self.stream, name)


class AsyncAnthropicStreamWrapper:
    """Wraps an Anthropic async streaming response to capture telemetry."""

    def __init__(
        self,
        stream: Any,
        handler: TelemetryHandler,
        invocation: LLMInvocation,
        capture_content: bool,
    ):
        self.stream = stream
        self.handler = handler
        self.invocation = invocation
        self.capture_content = capture_content
        self._accumulator = _ContentBlockAccumulator()
        self._started = True
        self._response_model: str | None = None
        self._response_id: str | None = None
        self._stop_reason: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._cache_creation_input_tokens: int | None = None
        self._cache_read_input_tokens: int | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self.stream.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._cleanup(exc_val)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self.stream.__anext__()
            try:
                self._process_event(event)
            except Exception:
                logger.debug(
                    "Could not record Anthropic stream event", exc_info=True
                )
            return event
        except StopAsyncIteration:
            self._cleanup()
            raise
        except (Exception, asyncio.CancelledError) as error:
            self._cleanup(error)
            raise

    async def close(self):
        try:
            await self.stream.close()
        finally:
            self._cleanup()

    def _process_event(self, event: Any) -> None:
        """Process a single streaming event - same logic as sync."""
        event_type = getattr(event, "type", None)

        if self.invocation.monotonic_first_token_s is None and event_type in (
            "content_block_delta",
        ):
            self.invocation.monotonic_first_token_s = timeit.default_timer()

        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message:
                self._response_model = getattr(message, "model", None)
                self._response_id = getattr(message, "id", None)
                usage = getattr(message, "usage", None)
                if usage:
                    self._input_tokens = _value(usage, "input_tokens", None)
                    if (
                        _value(usage, "cache_creation_input_tokens", None)
                        is not None
                    ):
                        self._cache_creation_input_tokens = _value(
                            usage, "cache_creation_input_tokens", None
                        )
                    if (
                        _value(usage, "cache_read_input_tokens", None)
                        is not None
                    ):
                        self._cache_read_input_tokens = _value(
                            usage, "cache_read_input_tokens", None
                        )

        elif event_type == "content_block_start":
            content_block = getattr(event, "content_block", None)
            if content_block and self.capture_content:
                self._accumulator.on_content_block_start(content_block)

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and self.capture_content:
                self._accumulator.on_content_block_delta(delta)

        elif event_type == "content_block_stop":
            if self.capture_content:
                self._accumulator.on_content_block_stop()

        elif event_type == "message_delta":
            delta = getattr(event, "delta", None)
            if delta:
                self._stop_reason = _value(delta, "stop_reason", None)
            usage = getattr(event, "usage", None)
            if usage:
                self._output_tokens = _value(usage, "output_tokens", None)

    def _cleanup(self, error: Optional[BaseException] = None) -> None:
        """Finalize the invocation with accumulated data."""
        if not self._started:
            return
        self._started = False
        try:
            self._finish(error)
        except Exception:
            _abandon_llm(self.handler, self.invocation)
            logger.debug(
                "Could not finalize Anthropic stream telemetry", exc_info=True
            )

    def _finish(self, error: Optional[BaseException]) -> None:
        self.invocation.response_model_name = self._response_model
        self.invocation.response_id = self._response_id
        self.invocation.input_tokens = total_input_tokens(
            self._input_tokens,
            self._cache_creation_input_tokens,
            self._cache_read_input_tokens,
        )
        self.invocation.output_tokens = self._output_tokens

        if self._cache_creation_input_tokens is not None:
            self.invocation.usage_cache_creation_input_tokens = (
                self._cache_creation_input_tokens
            )
        if self._cache_read_input_tokens is not None:
            self.invocation.usage_cache_read_input_tokens = (
                self._cache_read_input_tokens
            )

        finish_reason = self._stop_reason
        if finish_reason is None and error is not None:
            finish_reason = "error"
        if finish_reason is not None:
            self.invocation.finish_reasons = [finish_reason]

        if self.capture_content:
            self.invocation.output_messages = (
                self._accumulator.build_output_messages(finish_reason)
            )

        if error is not None:
            self.handler.fail_llm(
                self.invocation,
                Error(type=type(error), message=type(error).__name__),
            )
        else:
            self.handler.stop_llm(self.invocation)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying stream."""
        return getattr(self.stream, name)


def _abandon_llm(handler, invocation):
    try:
        handler.abandon_llm(invocation)
    except Exception:
        logger.debug("Could not clean up Anthropic telemetry", exc_info=True)


def messages_stream(handler, content_capturing_mode, asynchronous=False):
    """Trace the SDK helper while retaining its snapshot and text-stream APIs."""
    capture_content = content_capturing_mode != ContentCapturingMode.NO_CONTENT

    def traced_method(wrapped, instance, args, kwargs):
        if not is_instrumentation_enabled():
            return wrapped(*args, **kwargs)
        try:
            invocation = create_anthropic_invocation(
                kwargs, instance, capture_content
            )
        except Exception:
            logger.debug(
                "Could not prepare Anthropic telemetry", exc_info=True
            )
            return wrapped(*args, **kwargs)
        # The SDK helper is lazy: start the span only when entering its manager.
        manager = wrapped(*args, **kwargs)
        try:
            wrapper = _AsyncStreamManager if asynchronous else _StreamManager
            return wrapper(manager, handler, invocation, capture_content)
        except Exception:
            logger.debug(
                "Could not wrap Anthropic stream manager", exc_info=True
            )
            return manager

    return traced_method


class _StreamManager:
    def __init__(self, manager, handler, invocation, capture_content):
        self.manager = manager
        self.handler = handler
        self.invocation = invocation
        self.capture_content = capture_content
        self.telemetry = None

    def _start(self):
        if not is_instrumentation_enabled():
            return False
        try:
            self.handler.start_llm(self.invocation)
            return True
        except Exception:
            _abandon_llm(self.handler, self.invocation)
            logger.debug(
                "Could not start Anthropic helper telemetry", exc_info=True
            )
            return False

    def _wrap(self, stream, asynchronous=False):
        try:
            self.handler.detach_llm_context(self.invocation)
            # Both the oldest and latest SDK helpers consume _iterator through
            # iteration, text_stream, and get_final_message. Keep the SDK helper
            # itself so its parsing and custom event handlers remain intact.
            wrapper = (
                AsyncAnthropicStreamWrapper
                if asynchronous
                else AnthropicStreamWrapper
            )
            self.telemetry = wrapper(
                stream._iterator,
                self.handler,
                self.invocation,
                self.capture_content,
            )
            proxy = (
                _AsyncHelperStream(stream, self.telemetry)
                if asynchronous
                else _HelperStream(stream, self.telemetry)
            )
            stream._iterator = self.telemetry
            return proxy
        except Exception:
            _abandon_llm(self.handler, self.invocation)
            logger.debug(
                "Could not wrap Anthropic helper iterator", exc_info=True
            )
            return stream

    def __enter__(self):
        started = self._start()
        try:
            stream = self.manager.__enter__()
        except (Exception, asyncio.CancelledError) as error:
            if started:
                _fail_llm(self.handler, self.invocation, error)
            raise
        return self._wrap(stream) if started else stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self.manager.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self.telemetry is not None:
                self.telemetry._cleanup(exc_val)

    def __getattr__(self, name):
        return getattr(self.manager, name)


class _AsyncStreamManager(_StreamManager):
    async def __aenter__(self):
        started = self._start()
        try:
            stream = await self.manager.__aenter__()
        except (Exception, asyncio.CancelledError) as error:
            if started:
                _fail_llm(self.handler, self.invocation, error)
            raise
        return self._wrap(stream, asynchronous=True) if started else stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self.manager.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self.telemetry is not None:
                self.telemetry._cleanup(exc_val)


class _HelperStream(ObjectProxy):
    def __init__(self, stream, telemetry):
        super().__init__(stream)
        self._self_telemetry = telemetry

    def __enter__(self):
        self.__wrapped__.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._self_telemetry._cleanup(exc_val)

    def close(self):
        try:
            return self.__wrapped__.close()
        finally:
            self._self_telemetry._cleanup()


class _AsyncHelperStream(_HelperStream):
    async def __aenter__(self):
        await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._self_telemetry._cleanup(exc_val)

    def __aiter__(self):
        return self.__wrapped__.__aiter__()

    async def __anext__(self):
        return await self.__wrapped__.__anext__()

    async def close(self):
        try:
            return await self.__wrapped__.close()
        finally:
            self._self_telemetry._cleanup()


def _value(obj, name, default=None):
    """Older SDKs retain newly introduced event fields as dictionaries."""
    return (
        obj.get(name, default)
        if isinstance(obj, dict)
        else getattr(obj, name, default)
    )
