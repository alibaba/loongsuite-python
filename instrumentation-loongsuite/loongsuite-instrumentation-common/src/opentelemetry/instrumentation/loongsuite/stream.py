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

"""Stream proxies that preserve application protocol and isolate callbacks."""

import inspect
import logging
from threading import Lock
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    Optional,
    TypeVar,
)

from opentelemetry.instrumentation.loongsuite.advice import (
    async_call_advice,
    call_advice,
)

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class _FinalizationState:
    def __init__(self) -> None:
        self._finalized = False
        self._finalization_lock = Lock()
        self._closed = False
        self._close_lock = Lock()

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def _claim_finalization(self) -> bool:
        with self._finalization_lock:
            if self._finalized:
                return False
            self._finalized = True
            return True

    def _claim_close(self) -> bool:
        with self._close_lock:
            if self._closed:
                return False
            self._closed = True
            return True


class IsolatedStream(_FinalizationState, Generic[_T], Iterator[_T]):
    """Proxy a synchronous application iterator without changing its protocol."""

    def __init__(
        self,
        stream: Iterator[_T],
        *,
        on_chunk: Optional[Callable[[_T], Any]] = None,
        on_finish: Optional[Callable[[], Any]] = None,
        on_error: Optional[Callable[[BaseException], Any]] = None,
        instrumentation_name: str = "unknown",
    ) -> None:
        super().__init__()
        self.stream = stream
        self._iterator = stream
        self._on_chunk = on_chunk
        self._on_finish = on_finish
        self._on_error = on_error
        self._instrumentation_name = instrumentation_name

    def __iter__(self) -> "IsolatedStream[_T]":
        return self

    def _advance(self, operation: Callable[[], _T]) -> _T:
        try:
            chunk = operation()
        except StopIteration:
            self._finalize_stream(None)
            raise
        except BaseException as exception:
            self._finalize_stream(exception)
            raise

        if self._on_chunk is not None:
            try:
                call_advice(
                    self._on_chunk,
                    chunk,
                    instrumentation_name=self._instrumentation_name,
                    advice_method="stream_chunk",
                )
            except BaseException as exception:
                self._finalize_stream(exception)
                raise
        return chunk

    def __next__(self) -> _T:
        return self._advance(self._iterator.__next__)

    def send(self, value: Any) -> _T:
        send = getattr(self._iterator, "send")
        return self._advance(lambda: send(value))

    def throw(self, *args: Any) -> _T:
        throw = getattr(self._iterator, "throw")
        return self._advance(lambda: throw(*args))

    def close(self) -> Any:
        try:
            result = self._close_underlying()
        except BaseException as exception:
            self._finalize_stream(exception)
            raise
        self._finalize_stream(None)
        return result

    def _close_underlying(self) -> Any:
        if not self._claim_close():
            return None
        close = getattr(self.stream, "close", None)
        return close() if callable(close) else None

    def __enter__(self) -> "IsolatedStream[_T]":
        enter = getattr(self.stream, "__enter__", None)
        if not callable(enter):
            return self
        try:
            entered = enter()
        except BaseException as exception:
            self._finalize_stream(exception)
            raise
        if hasattr(entered, "__next__"):
            self._iterator = entered
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        exit_method = getattr(self.stream, "__exit__", None)
        if not callable(exit_method):
            try:
                self._close_underlying()
            except BaseException as close_error:
                if exc_value is None:
                    self._finalize_stream(close_error)
                    raise
                call_advice(
                    _logger.debug,
                    "Stream close failed while preserving %r: %r",
                    exc_value,
                    close_error,
                    instrumentation_name=self._instrumentation_name,
                    advice_method="stream_close_internal",
                )
            self._finalize_stream(exc_value)
            return False
        try:
            suppressed = exit_method(exc_type, exc_value, traceback)
        except BaseException as exception:
            self._finalize_stream(exception)
            raise
        self._finalize_stream(None if suppressed else exc_value)
        return suppressed

    def _finalize_stream(self, error: Optional[BaseException]) -> None:
        if not self._claim_finalization():
            return
        if error is None:
            if self._on_finish is not None:
                call_advice(
                    self._on_finish,
                    instrumentation_name=self._instrumentation_name,
                    advice_method="stream_finish",
                )
            return
        if self._on_error is not None:
            call_advice(
                self._on_error,
                error,
                instrumentation_name=self._instrumentation_name,
                advice_method="stream_error",
            )


class IsolatedAsyncStream(_FinalizationState, Generic[_T], AsyncIterator[_T]):
    """Proxy an async application iterator without changing its protocol."""

    def __init__(
        self,
        stream: AsyncIterator[_T],
        *,
        on_chunk: Optional[Callable[[_T], Any]] = None,
        on_finish: Optional[Callable[[], Any]] = None,
        on_error: Optional[Callable[[BaseException], Any]] = None,
        instrumentation_name: str = "unknown",
    ) -> None:
        super().__init__()
        self.stream = stream
        self._iterator = stream
        self._on_chunk = on_chunk
        self._on_finish = on_finish
        self._on_error = on_error
        self._instrumentation_name = instrumentation_name

    def __aiter__(self) -> "IsolatedAsyncStream[_T]":
        return self

    async def __anext__(self) -> _T:
        return await self._advance(self._iterator.__anext__)

    async def _advance(self, operation: Callable[[], Any]) -> _T:
        try:
            chunk = await operation()
        except StopAsyncIteration:
            await self._finalize_stream(None)
            raise
        except BaseException as exception:
            await self._finalize_preserving(exception)
            raise

        if self._on_chunk is not None:
            try:
                await async_call_advice(
                    self._on_chunk,
                    chunk,
                    instrumentation_name=self._instrumentation_name,
                    advice_method="stream_chunk",
                )
            except BaseException as exception:
                await self._finalize_preserving(exception)
                raise
        return chunk

    async def asend(self, value: Any) -> _T:
        send = getattr(self._iterator, "asend")
        return await self._advance(lambda: send(value))

    async def athrow(self, *args: Any) -> _T:
        throw = getattr(self._iterator, "athrow")
        return await self._advance(lambda: throw(*args))

    async def aclose(self) -> Any:
        try:
            result = await self._close_underlying()
        except BaseException as exception:
            await self._finalize_preserving(exception)
            raise
        await self._finalize_stream(None)
        return result

    async def _close_underlying(self) -> Any:
        if not self._claim_close():
            return None
        aclose = getattr(self.stream, "aclose", None)
        close = getattr(self.stream, "close", None)
        if callable(aclose):
            result = aclose()
        elif callable(close):
            result = close()
        else:
            result = None
        if inspect.isawaitable(result):
            return await result
        return result

    async def __aenter__(self) -> "IsolatedAsyncStream[_T]":
        enter = getattr(self.stream, "__aenter__", None)
        if not callable(enter):
            return self
        try:
            entered = enter()
            if inspect.isawaitable(entered):
                entered = await entered
        except BaseException as exception:
            await self._finalize_preserving(exception)
            raise
        if hasattr(entered, "__anext__"):
            self._iterator = entered
        return self

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, traceback: Any
    ) -> Any:
        exit_method = getattr(self.stream, "__aexit__", None)
        if not callable(exit_method):
            try:
                await self._close_underlying()
            except BaseException as close_error:
                if exc_value is None:
                    await self._finalize_preserving(close_error)
                    raise
                call_advice(
                    _logger.debug,
                    "Async stream close failed while preserving %r: %r",
                    exc_value,
                    close_error,
                    instrumentation_name=self._instrumentation_name,
                    advice_method="stream_close_internal",
                )
            await self._finalize_stream(exc_value)
            return False
        try:
            suppressed = exit_method(exc_type, exc_value, traceback)
            if inspect.isawaitable(suppressed):
                suppressed = await suppressed
        except BaseException as exception:
            await self._finalize_preserving(exception)
            raise
        await self._finalize_stream(None if suppressed else exc_value)
        return suppressed

    async def _finalize_preserving(self, error: BaseException) -> None:
        try:
            await self._finalize_stream(error)
        except BaseException as advice_error:  # preserve the application error
            call_advice(
                _logger.debug,
                "Stream finalization failed while preserving %r: %r",
                error,
                advice_error,
                instrumentation_name=self._instrumentation_name,
                advice_method="stream_finalize_internal",
            )

    async def _finalize_stream(self, error: Optional[BaseException]) -> None:
        if not self._claim_finalization():
            return
        if error is None:
            if self._on_finish is not None:
                await async_call_advice(
                    self._on_finish,
                    instrumentation_name=self._instrumentation_name,
                    advice_method="stream_finish",
                )
            return
        if self._on_error is not None:
            await async_call_advice(
                self._on_error,
                error,
                instrumentation_name=self._instrumentation_name,
                advice_method="stream_error",
            )
