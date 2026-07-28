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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from inspect import isawaitable
from threading import Lock
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    AsyncIterable,
    Generic,
    Iterable,
    Literal,
    Protocol,
    TypeVar,
)

from opentelemetry.util.genai import hook_advice

if TYPE_CHECKING:

    class _ObjectProxy:
        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy


ChunkT = TypeVar("ChunkT")
_ChunkT_co = TypeVar("_ChunkT_co", covariant=True)
_logger = logging.getLogger(__name__)


@hook_advice("google-genai", "stream_chunk")
def _process_stream_chunk_advice(wrapper: object, chunk: object) -> None:
    wrapper._process_chunk(chunk)  # type: ignore[attr-defined]


@hook_advice("google-genai", "stream_success")
def _finish_stream_success_advice(wrapper: object) -> None:
    wrapper._on_stream_end()  # type: ignore[attr-defined]


@hook_advice("google-genai", "stream_error")
def _finish_stream_error_advice(wrapper: object, error: BaseException) -> None:
    wrapper._on_stream_error(error)  # type: ignore[attr-defined]


class _StreamWrapperMeta(ABCMeta, type(_ObjectProxy)):
    """Metaclass compatible with wrapt's proxy type and ABC hooks."""


class _SyncStream(Iterable[_ChunkT_co], Protocol[_ChunkT_co]):
    """Structural type for streams accepted by ``SyncStreamWrapper``."""

    def close(self) -> None: ...


class _AsyncStream(AsyncIterable[_ChunkT_co], Protocol[_ChunkT_co]):
    """Structural type for streams accepted by ``AsyncStreamWrapper``."""


class SyncStreamWrapper(
    _ObjectProxy,
    Generic[ChunkT],
    metaclass=_StreamWrapperMeta,
):
    """Base class for synchronous instrumented stream wrappers.

    Subclass this when wrapping a provider SDK stream that is consumed with
    normal iteration. The subclass should pass the SDK stream to
    ``super().__init__(stream)`` and implement the three telemetry hooks:
    ``_process_chunk`` for per-chunk state, ``_on_stream_end`` for successful
    finalization, and ``_on_stream_error`` for failure finalization.

    Users should consume subclasses as normal streams, for example with
    ``for chunk in wrapper`` or ``with wrapper``. The hook methods are called
    internally by the wrapper lifecycle and are not part of the public API.
    """

    def __init__(self, stream: _SyncStream[ChunkT]):
        super().__init__(stream)
        self._self_stream = stream
        self._self_iterator = iter(stream)
        self._self_finalized = False
        self._self_finalize_lock = Lock()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_val is not None:
            self._finalize_failure(exc_val)
            try:
                self._self_stream.close()
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "GenAI stream close error after user exception",
                    exc_info=True,
                )
            return False

        self.close()
        return False

    def close(self) -> None:
        try:
            self._self_stream.close()
        except Exception:
            _logger.debug(
                "Google GenAI stream close failed",
                exc_info=True,
            )
        self._finalize_success()

    def __iter__(self):
        # Override ``ObjectProxy.__iter__`` so iteration drives ``__next__``
        # below and runs ``_process_chunk`` per chunk; otherwise iteration
        # would be forwarded to the wrapped stream and bypass instrumentation.
        return self

    def __next__(self) -> ChunkT:
        try:
            chunk = next(self._self_iterator)
        except StopIteration:
            self._finalize_success()
            raise
        except GeneratorExit:
            self._finalize_success()
            raise
        except BaseException as error:
            self._finalize_failure(error)
            raise
        _process_stream_chunk_advice(self, chunk)
        return chunk

    def _finalize_success(self) -> None:
        with self._self_finalize_lock:
            if self._self_finalized:
                return
            self._self_finalized = True
        _finish_stream_success_advice(self)

    def _finalize_failure(self, error: BaseException) -> None:
        with self._self_finalize_lock:
            if self._self_finalized:
                return
            self._self_finalized = True
        _finish_stream_error_advice(self, error)

    def __del__(self) -> None:
        if getattr(self, "_self_finalized", True):
            return
        try:
            self._finalize_success()
        except Exception:
            pass

    @abstractmethod
    def _process_chunk(self, chunk: ChunkT) -> None:
        """Process one stream chunk for telemetry."""

    @abstractmethod
    def _on_stream_end(self) -> None:
        """Finalize the stream successfully."""

    @abstractmethod
    def _on_stream_error(self, error: BaseException) -> None:
        """Finalize the stream with failure."""


class AsyncStreamWrapper(
    _ObjectProxy,
    Generic[ChunkT],
    metaclass=_StreamWrapperMeta,
):
    """Base class for asynchronous instrumented stream wrappers.

    Subclass this when wrapping a provider SDK stream that is consumed with
    async iteration. The subclass should pass the SDK stream to
    ``super().__init__(stream)`` and implement the three telemetry hooks:
    ``_process_chunk`` for per-chunk state, ``_on_stream_end`` for successful
    finalization, and ``_on_stream_error`` for failure finalization.

    Users should consume subclasses as normal async streams, for example with
    ``async for chunk in wrapper`` or ``async with wrapper``. The hook methods
    remain synchronous telemetry hooks; async stream reads and close handling
    are owned by this base class.
    """

    def __init__(self, stream: _AsyncStream[ChunkT]):
        super().__init__(stream)
        self._self_stream = stream
        # LoongSuite supports Python 3.9, where aiter/anext are unavailable.
        self._self_aiter = stream.__aiter__()
        self._self_finalized = False
        self._self_finalize_lock = Lock()

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_val is not None:
            self._finalize_failure(exc_val)
            try:
                await self._close_wrapped_stream()
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "GenAI stream close error after user exception",
                    exc_info=True,
                )
            return False

        await self.close()
        return False

    async def close(self) -> None:
        try:
            await self._close_wrapped_stream()
        except Exception:
            _logger.debug(
                "Google GenAI async stream close failed",
                exc_info=True,
            )
        self._finalize_success()

    async def aclose(self) -> None:
        """Close SDKs such as Google GenAI that expose ``aclose``."""

        await self.close()

    async def _close_wrapped_stream(self) -> None:
        close = getattr(self._self_stream, "aclose", None)
        if close is None:
            close = getattr(self._self_stream, "close", None)
        if close is None:
            return
        result = close()
        if isawaitable(result):
            await result

    def __aiter__(self):
        # Override ``ObjectProxy.__aiter__`` so iteration drives ``__anext__``
        # below and runs ``_process_chunk`` per chunk; otherwise iteration
        # would be forwarded to the wrapped stream and bypass instrumentation.
        return self

    async def __anext__(self) -> ChunkT:
        try:
            chunk = await self._self_aiter.__anext__()
        except StopAsyncIteration:
            self._finalize_success()
            raise
        except GeneratorExit:
            self._finalize_success()
            raise
        except BaseException as error:
            self._finalize_failure(error)
            raise

        _process_stream_chunk_advice(self, chunk)
        return chunk

    def _finalize_success(self) -> None:
        with self._self_finalize_lock:
            if self._self_finalized:
                return
            self._self_finalized = True
        _finish_stream_success_advice(self)

    def _finalize_failure(self, error: BaseException) -> None:
        with self._self_finalize_lock:
            if self._self_finalized:
                return
            self._self_finalized = True
        _finish_stream_error_advice(self, error)

    def __del__(self) -> None:
        if getattr(self, "_self_finalized", True):
            return
        try:
            self._finalize_success()
        except Exception:
            pass

    @abstractmethod
    def _process_chunk(self, chunk: ChunkT) -> None:
        """Process one stream chunk for telemetry."""

    @abstractmethod
    def _on_stream_end(self) -> None:
        """Finalize the stream successfully."""

    @abstractmethod
    def _on_stream_error(self, error: BaseException) -> None:
        """Finalize the stream with failure."""


__all__ = [
    "AsyncStreamWrapper",
    "SyncStreamWrapper",
]
