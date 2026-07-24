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

"""Failure isolation for instrumentation-only advice functions.

These decorators must only wrap probe-owned work. Application calls and stream
iteration belong outside the decorated function so application exceptions are
never mistaken for instrumentation failures.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

_logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])


def hook_advice(
    instrumentation_name: str = "unknown",
    advice_method: str = "unknown",
    throw_exception: bool = False,
) -> Callable[[_F], _F]:
    """Decorate synchronous instrumentation-only work with fail-open semantics.

    Suppressed failures return ``None``, so callers must not depend on advice
    return values unless ``throw_exception`` is enabled.

    Generator functions are rejected because their bodies execute during later
    iteration, after this decorator has returned the generator object. Stream
    iteration must instead be isolated by the owning instrumentation wrapper.
    """

    def decorator(func: _F) -> _F:
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(
            func
        ):
            raise TypeError(
                "hook_advice cannot decorate generator functions; "
                "isolate iteration in the instrumentation stream wrapper"
            )
        if inspect.iscoroutinefunction(func):
            raise TypeError(
                "hook_advice cannot decorate coroutine functions; "
                "use async_hook_advice"
            )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "LoongSuite instrumentation %s advice %s failed: %s",
                    instrumentation_name,
                    advice_method,
                    exception,
                    exc_info=True,
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
    """Decorate asynchronous instrumentation-only work with fail-open semantics.

    Suppressed failures return ``None``, so callers must not depend on advice
    return values unless ``throw_exception`` is enabled.

    Async generators are rejected because their bodies execute during later
    iteration. The owning instrumentation wrapper must isolate that lifecycle.
    """

    def decorator(func: _F) -> _F:
        if inspect.isasyncgenfunction(func):
            raise TypeError(
                "async_hook_advice cannot decorate async generator functions; "
                "isolate iteration in the instrumentation stream wrapper"
            )
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                "async_hook_advice requires a coroutine function; "
                "use hook_advice for synchronous advice"
            )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "LoongSuite instrumentation %s advice %s failed: %s",
                    instrumentation_name,
                    advice_method,
                    exception,
                    exc_info=True,
                )
                if throw_exception:
                    raise
                return None

        return cast(_F, wrapper)

    return decorator
