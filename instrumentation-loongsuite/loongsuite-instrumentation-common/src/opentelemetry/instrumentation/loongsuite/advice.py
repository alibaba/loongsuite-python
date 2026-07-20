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

"""Failure-isolated helpers for instrumentation-only callbacks."""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

_logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])


def _log_advice_failure(
    instrumentation_name: str,
    advice_method: str,
    exception: Exception,
) -> None:
    """Report instrumentation failure without creating another failure path."""
    try:
        _logger.debug(
            "LoongSuite instrumentation %s advice %s failed: %s",
            instrumentation_name,
            advice_method,
            exception,
            exc_info=True,
        )
    except Exception:  # pragma: no cover - logging implementations vary
        pass


def call_advice(
    callback: Callable[..., Any],
    *args: Any,
    instrumentation_name: str = "unknown",
    advice_method: str = "unknown",
    **kwargs: Any,
) -> Any:
    """Call a synchronous instrumentation callback using fail-open semantics."""
    try:
        return callback(*args, **kwargs)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        _log_advice_failure(
            instrumentation_name,
            advice_method,
            exception,
        )
        return None


async def async_call_advice(
    callback: Callable[..., Any],
    *args: Any,
    instrumentation_name: str = "unknown",
    advice_method: str = "unknown",
    **kwargs: Any,
) -> Any:
    """Call a sync or async instrumentation callback using fail-open semantics."""
    try:
        result = callback(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as exception:  # pylint: disable=broad-exception-caught
        _log_advice_failure(
            instrumentation_name,
            advice_method,
            exception,
        )
        return None


def hook_advice(
    instrumentation_name: str = "unknown",
    advice_method: str = "unknown",
    throw_exception: bool = False,
) -> Callable[[_F], _F]:
    """Decorate a synchronous instrumentation-only callback.

    Generator functions are rejected because calling them only creates a
    generator. Their body and failures occur later, outside this decorator's
    exception boundary. Use :class:`IsolatedStream` for that lifecycle.
    """

    def decorator(func: _F) -> _F:
        if inspect.isgeneratorfunction(func):
            raise TypeError(
                "hook_advice cannot decorate generator functions; "
                "use IsolatedStream"
            )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exception:  # pylint: disable=broad-exception-caught
                _log_advice_failure(
                    instrumentation_name,
                    advice_method,
                    exception,
                )
                if throw_exception:
                    raise
                return None

        return cast(_F, wrapper)

    return decorator


def async_hook_advice(
    instrumentation_name: str = "unknown",
    advice_method: str = "unknown",
    throw_exception: bool = False,
) -> Callable[[_F], _F]:
    """Decorate an asynchronous instrumentation-only callback.

    Async generators are rejected because they are asynchronous iterators, not
    awaitables. Use :class:`IsolatedAsyncStream` for that lifecycle.
    """

    def decorator(func: _F) -> _F:
        if inspect.isasyncgenfunction(func):
            raise TypeError(
                "async_hook_advice cannot decorate async generator functions; "
                "use IsolatedAsyncStream"
            )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as exception:  # pylint: disable=broad-exception-caught
                _log_advice_failure(
                    instrumentation_name,
                    advice_method,
                    exception,
                )
                if throw_exception:
                    raise
                return None

        return cast(_F, wrapper)

    return decorator
