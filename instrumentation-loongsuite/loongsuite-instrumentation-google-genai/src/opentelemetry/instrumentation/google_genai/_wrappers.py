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

"""Google GenAI SDK wrappers backed by ``ExtendedTelemetryHandler``."""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import Any, Callable

from opentelemetry import context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.util.genai.types import Error

from ._context import GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
from ._stream import AsyncStreamWrapper, SyncStreamWrapper
from ._utils import (
    apply_embedding_response,
    apply_response,
    create_embedding_invocation,
    create_llm_invocation,
)

_logger = logging.getLogger(__name__)


def _is_suppressed() -> bool:
    return bool(context.get_value(_SUPPRESS_INSTRUMENTATION_KEY))


def _extra_attributes() -> Mapping[str, Any] | None:
    value = context.get_value(GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY)
    return value if isinstance(value, Mapping) else None


def _fail(handler: Any, invocation: Any, error: BaseException) -> None:
    try:
        handler.fail_llm(
            invocation, Error(message=str(error), type=type(error))
        )
    except Exception as exc:
        _logger.debug("Failed to report Google GenAI error: %s", exc)


def _finish(handler: Any, invocation: Any, response: Any) -> None:
    try:
        apply_response(invocation, response)
    except Exception as exc:
        _logger.debug("Failed to process Google GenAI response: %s", exc)
    try:
        handler.stop_llm(invocation)
    except Exception as exc:
        _logger.debug("Failed to finalize Google GenAI invocation: %s", exc)


def create_sync_generate_wrapper(
    original: Callable[..., Any], handler: Any, *, streaming: bool
):
    function_name = (
        "google.genai.Models.generate_content_stream"
        if streaming
        else "google.genai.Models.generate_content"
    )

    @functools.wraps(original)
    def wrapper(self, *, model, contents, config=None, **kwargs):
        if _is_suppressed():
            return original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            invocation = create_llm_invocation(
                self,
                model=model,
                contents=contents,
                config=config,
                function_name=function_name,
                extra_attributes=_extra_attributes(),
            )
            handler.start_llm(invocation)
        except Exception as exc:
            _logger.debug(
                "Failed to initialize Google GenAI instrumentation: %s", exc
            )
            return original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            response = original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except BaseException as error:
            _fail(handler, invocation, error)
            raise
        if streaming:
            return SyncStreamWrapper(response, invocation, handler)
        _finish(handler, invocation, response)
        return response

    return wrapper


def create_async_generate_wrapper(
    original: Callable[..., Any], handler: Any, *, streaming: bool
):
    function_name = (
        "google.genai.AsyncModels.generate_content_stream"
        if streaming
        else "google.genai.AsyncModels.generate_content"
    )

    @functools.wraps(original)
    async def wrapper(self, *, model, contents, config=None, **kwargs):
        if _is_suppressed():
            return await original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            invocation = create_llm_invocation(
                self,
                model=model,
                contents=contents,
                config=config,
                function_name=function_name,
                extra_attributes=_extra_attributes(),
            )
            handler.start_llm(invocation)
        except Exception as exc:
            _logger.debug(
                "Failed to initialize Google GenAI instrumentation: %s", exc
            )
            return await original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            response = await original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except BaseException as error:
            _fail(handler, invocation, error)
            raise
        if streaming:
            return AsyncStreamWrapper(response, invocation, handler)
        _finish(handler, invocation, response)
        return response

    return wrapper


def create_sync_embedding_wrapper(original: Callable[..., Any], handler: Any):
    @functools.wraps(original)
    def wrapper(self, *, model, contents, config=None, **kwargs):
        if _is_suppressed():
            return original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            invocation = create_embedding_invocation(
                self,
                model=model,
                function_name="google.genai.Models.embed_content",
            )
            handler.start_embedding(invocation)
        except Exception as exc:
            _logger.debug(
                "Failed to initialize Google GenAI embedding telemetry: %s",
                exc,
            )
            return original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            response = original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except BaseException as error:
            try:
                handler.fail_embedding(
                    invocation, Error(message=str(error), type=type(error))
                )
            except Exception as exc:
                _logger.debug(
                    "Failed to report Google GenAI embedding error: %s", exc
                )
            raise
        try:
            apply_embedding_response(invocation, response)
            handler.stop_embedding(invocation)
        except Exception as exc:
            _logger.debug(
                "Failed to finalize Google GenAI embedding telemetry: %s", exc
            )
        return response

    return wrapper


def create_async_embedding_wrapper(original: Callable[..., Any], handler: Any):
    @functools.wraps(original)
    async def wrapper(self, *, model, contents, config=None, **kwargs):
        if _is_suppressed():
            return await original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            invocation = create_embedding_invocation(
                self,
                model=model,
                function_name="google.genai.AsyncModels.embed_content",
            )
            handler.start_embedding(invocation)
        except Exception as exc:
            _logger.debug(
                "Failed to initialize Google GenAI embedding telemetry: %s",
                exc,
            )
            return await original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        try:
            response = await original(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except BaseException as error:
            try:
                handler.fail_embedding(
                    invocation, Error(message=str(error), type=type(error))
                )
            except Exception as exc:
                _logger.debug(
                    "Failed to report Google GenAI embedding error: %s", exc
                )
            raise
        try:
            apply_embedding_response(invocation, response)
            handler.stop_embedding(invocation)
        except Exception as exc:
            _logger.debug(
                "Failed to finalize Google GenAI embedding telemetry: %s", exc
            )
        return response

    return wrapper
