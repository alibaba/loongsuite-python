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

"""Protocol-preserving wrappers for Google GenAI streaming responses."""

from __future__ import annotations

import logging
import timeit
from typing import Any

from opentelemetry.util.genai.types import Error, LLMInvocation

from ._utils import ResponseAccumulator, apply_response

_logger = logging.getLogger(__name__)


class _StreamLifecycle:
    def __init__(self, stream: Any, invocation: LLMInvocation, handler: Any):
        self._stream = stream
        self._invocation = invocation
        self._handler = handler
        self._accumulator = ResponseAccumulator()
        self._finished = False

    def _observe(self, response: Any) -> None:
        if self._invocation.monotonic_first_token_s is None:
            self._invocation.monotonic_first_token_s = timeit.default_timer()
        try:
            apply_response(self._invocation, response)
            self._accumulator.add(response)
        except Exception as exc:  # telemetry must not affect the stream
            _logger.debug(
                "Failed to process Google GenAI stream chunk: %s", exc
            )

    def _finish_success(self) -> None:
        if self._finished:
            return
        self._finished = True
        output_messages = self._accumulator.output_messages()
        if output_messages:
            self._invocation.output_messages = output_messages
        try:
            self._handler.stop_llm(self._invocation)
        except Exception as exc:
            _logger.debug("Failed to finalize Google GenAI stream: %s", exc)

    def _finish_error(self, error: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        output_messages = self._accumulator.output_messages()
        if output_messages:
            self._invocation.output_messages = output_messages
        try:
            self._handler.fail_llm(
                self._invocation,
                Error(message=str(error), type=type(error)),
            )
        except Exception as exc:
            _logger.debug(
                "Failed to report Google GenAI stream error: %s", exc
            )


class SyncStreamWrapper(_StreamLifecycle):
    def _close_underlying(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            response = next(self._stream)
        except StopIteration:
            self._finish_success()
            raise
        except BaseException as error:
            self._finish_error(error)
            raise
        self._observe(response)
        return response

    def send(self, value):
        try:
            response = self._stream.send(value)
        except StopIteration:
            self._finish_success()
            raise
        except BaseException as error:
            self._finish_error(error)
            raise
        self._observe(response)
        return response

    def throw(self, *args, **kwargs):
        try:
            response = self._stream.throw(*args, **kwargs)
        except StopIteration:
            self._finish_success()
            raise
        except BaseException as error:
            self._finish_error(error)
            raise
        self._observe(response)
        return response

    def close(self) -> None:
        try:
            self._close_underlying()
        except BaseException as error:
            if not self._finished:
                self._finish_error(error)
            raise
        self._finish_success()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            try:
                self._close_underlying()
            except BaseException as close_error:
                _logger.debug(
                    "Failed to close Google GenAI stream after error: %s",
                    close_error,
                )
            finally:
                self._finish_error(exc_value)
            return False
        self.close()
        return False


class AsyncStreamWrapper(_StreamLifecycle):
    async def _close_underlying(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if close is not None:
            await close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            response = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish_success()
            raise
        except BaseException as error:
            self._finish_error(error)
            raise
        self._observe(response)
        return response

    async def asend(self, value):
        try:
            response = await self._stream.asend(value)
        except StopAsyncIteration:
            self._finish_success()
            raise
        except BaseException as error:
            self._finish_error(error)
            raise
        self._observe(response)
        return response

    async def athrow(self, *args, **kwargs):
        try:
            response = await self._stream.athrow(*args, **kwargs)
        except StopAsyncIteration:
            self._finish_success()
            raise
        except BaseException as error:
            self._finish_error(error)
            raise
        self._observe(response)
        return response

    async def aclose(self) -> None:
        try:
            await self._close_underlying()
        except BaseException as error:
            if not self._finished:
                self._finish_error(error)
            raise
        self._finish_success()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            try:
                await self._close_underlying()
            except BaseException as close_error:
                _logger.debug(
                    "Failed to close Google GenAI stream after error: %s",
                    close_error,
                )
            finally:
                self._finish_error(exc_value)
            return False
        await self.aclose()
        return False
