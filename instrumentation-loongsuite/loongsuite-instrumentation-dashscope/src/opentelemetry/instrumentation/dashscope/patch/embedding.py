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

"""Patch functions for DashScope TextEmbedding API."""

from __future__ import annotations

import logging

from opentelemetry.util.genai.extended_types import EmbeddingInvocation
from opentelemetry.util.genai.types import Error

from ..utils import _get_parameter

logger = logging.getLogger(__name__)


def wrap_text_embedding_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for TextEmbedding.call.

    Uses ExtendedTelemetryHandler which supports embedding operations.

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
        # Create embedding invocation object
        invocation = EmbeddingInvocation(request_model=model)
        invocation.provider = "dashscope"
        invocation.server_address = "dashscope.aliyuncs.com"
        invocation.server_port = 443

        # Extract parameters from kwargs or kwargs["parameters"] dict
        parameters = kwargs.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        # Extract dimension count if specified
        dimension = _get_parameter(
            kwargs, "dimension", parameters
        ) or _get_parameter(kwargs, "dimensions", parameters)
        if dimension is not None:
            invocation.dimension_count = dimension

        # Start embedding invocation (creates span)
        handler.start_embedding(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Extract usage information and other attributes
            # DashScope embedding response uses total_tokens (not input_tokens)
            if result:
                try:
                    usage = getattr(result, "usage", None)
                    if usage is not None and isinstance(usage, dict):
                        # For embedding, DashScope uses total_tokens instead of input_tokens
                        total_tokens = usage.get("total_tokens")
                        if total_tokens is not None:
                            invocation.input_tokens = total_tokens
                except (KeyError, AttributeError) as e:
                    # If usage extraction fails, continue without setting input_tokens
                    logger.debug(
                        "Failed to extract usage information from embedding response: %s",
                        e,
                    )

            # Successfully complete (sets attributes and ends span)
            handler.stop_embedding(invocation)
            return result

        except Exception as e:
            # Failure handling (sets error attributes and ends span)
            error = Error(message=str(e), type=type(e))
            handler.fail_embedding(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in embedding instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)
