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

"""Patch functions for DashScope Generation API."""

from __future__ import annotations

import logging
import timeit

from opentelemetry.util.genai.types import Error

from ..utils import (
    _create_accumulated_response,
    _create_invocation_from_generation,
    _get_parameter,
    _update_invocation_from_response,
)
from .common import _is_streaming_response

logger = logging.getLogger(__name__)


def _generation_chunk_has_token(chunk) -> bool:
    output = getattr(chunk, "output", None)
    if output is None:
        return False
    if getattr(output, "text", None) or getattr(output, "content", None):
        return True
    for choice in getattr(output, "choices", None) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        if getattr(message, "content", None):
            return True
        if getattr(message, "tool_calls", None):
            return True
    return False


def wrap_generation_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for Generation.call (sync).

    Uses TelemetryHandler from opentelemetry-util-genai to manage span lifecycle.

    Args:
        wrapped: The original function being wrapped
        instance: The instance the method is bound to (if any)
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    # Extract model from kwargs
    model = kwargs.get("model")
    if not model:
        logger.warning("Model not found in kwargs, skipping instrumentation")
        return wrapped(*args, **kwargs)

    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return wrapped(*args, **kwargs)

    try:
        # Create invocation object
        invocation = _create_invocation_from_generation(kwargs, model)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Handle streaming response
            if _is_streaming_response(result):
                # Check incremental_output parameter (default is False, meaning full output)
                incremental_output = _get_parameter(
                    kwargs, "incremental_output"
                )
                return _wrap_sync_generator(
                    result,
                    handler,
                    invocation,
                    incremental_output=incremental_output,
                )

            # Handle non-streaming response
            _update_invocation_from_response(invocation, result)
            handler.stop_llm(invocation)
            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)


async def wrap_aio_generation_call(
    wrapped, instance, args, kwargs, handler=None
):
    """Wrapper for AioGeneration.call (async).

    Uses TelemetryHandler from opentelemetry-util-genai to manage span lifecycle.

    Args:
        wrapped: The original function being wrapped
        instance: The instance the method is bound to (if any)
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    # Extract model from kwargs
    model = kwargs.get("model")
    if not model:
        logger.warning("Model not found in kwargs, skipping instrumentation")
        return await wrapped(*args, **kwargs)

    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return await wrapped(*args, **kwargs)

    try:
        # Create invocation object
        invocation = _create_invocation_from_generation(kwargs, model)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call
            result = await wrapped(*args, **kwargs)

            # Handle streaming response
            if _is_streaming_response(result):
                # Check incremental_output parameter (default is False, meaning full output)
                incremental_output = _get_parameter(
                    kwargs, "incremental_output"
                )
                return _wrap_async_generator(
                    result,
                    handler,
                    invocation,
                    incremental_output=incremental_output,
                )

            # Handle non-streaming response
            _update_invocation_from_response(invocation, result)
            handler.stop_llm(invocation)
            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in async instrumentation wrapper: %s", e)
        return await wrapped(*args, **kwargs)


def _wrap_sync_generator(
    generator, handler, invocation, incremental_output=None
):
    """Wrap a synchronous generator to collect data and set attributes.

    Args:
        generator: The generator to wrap
        handler: TelemetryHandler instance
        invocation: LLMInvocation object
        incremental_output: If True, chunks contain only incremental text (need to accumulate).
                          If False or None (default), chunks contain full accumulated text.
    """
    last_response = None
    accumulated_text = ""
    try:
        for chunk in generator:
            now = timeit.default_timer()
            invocation.stream = True
            if invocation.monotonic_first_chunk_s is None:
                invocation.monotonic_first_chunk_s = now
            if (
                invocation.monotonic_first_token_s is None
                and _generation_chunk_has_token(chunk)
            ):
                invocation.monotonic_first_token_s = now

            last_response = chunk

            # If incremental_output is True, accumulate text from each chunk
            if incremental_output:
                try:
                    # TODO check choice
                    output = getattr(chunk, "output", None)
                    if output:
                        chunk_text = getattr(output, "text", None) or getattr(
                            output, "content", None
                        )
                        if chunk_text:
                            accumulated_text += chunk_text
                except (KeyError, AttributeError) as e:
                    logger.debug(
                        "Failed to extract chunk text from generation response: %s",
                        e,
                    )

            yield chunk

        # After generator completes, update invocation and set attributes
        if last_response:
            # If incremental_output is True, create a modified response with accumulated text
            if incremental_output and accumulated_text:
                last_response = _create_accumulated_response(
                    last_response, accumulated_text
                )

            _update_invocation_from_response(invocation, last_response)
        handler.stop_llm(invocation)

    except Exception as e:
        error = Error(message=str(e), type=type(e))
        handler.fail_llm(invocation, error)
        raise


async def _wrap_async_generator(
    generator, handler, invocation, incremental_output=None
):
    """Wrap an asynchronous generator to collect data and set attributes.

    Args:
        generator: The async generator to wrap
        handler: TelemetryHandler instance
        invocation: LLMInvocation object
        incremental_output: If True, chunks contain only incremental text (need to accumulate).
                          If False or None (default), chunks contain full accumulated text.
    """
    last_response = None
    accumulated_text = ""
    try:
        async for chunk in generator:
            now = timeit.default_timer()
            invocation.stream = True
            if invocation.monotonic_first_chunk_s is None:
                invocation.monotonic_first_chunk_s = now
            if (
                invocation.monotonic_first_token_s is None
                and _generation_chunk_has_token(chunk)
            ):
                invocation.monotonic_first_token_s = now

            last_response = chunk

            # If incremental_output is True, accumulate text from each chunk
            if incremental_output:
                try:
                    output = getattr(chunk, "output", None)
                    if output:
                        chunk_text = getattr(output, "text", None) or getattr(
                            output, "content", None
                        )
                        if chunk_text:
                            accumulated_text += chunk_text
                except (KeyError, AttributeError) as e:
                    logger.debug(
                        "Failed to extract chunk text from generation response: %s",
                        e,
                    )

            yield chunk

        # After generator completes, update invocation and set attributes
        if last_response:
            # If incremental_output is True, create a modified response with accumulated text
            if incremental_output and accumulated_text:
                last_response = _create_accumulated_response(
                    last_response, accumulated_text
                )

            _update_invocation_from_response(invocation, last_response)
        handler.stop_llm(invocation)

    except Exception as e:
        error = Error(message=str(e), type=type(e))
        handler.fail_llm(invocation, error)
        raise
