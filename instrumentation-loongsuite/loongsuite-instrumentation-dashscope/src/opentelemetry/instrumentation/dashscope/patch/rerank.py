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

"""Patch functions for DashScope TextReRank API."""

from __future__ import annotations

import logging

from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GenAiExtendedProviderNameValues as GenAI,
)
from opentelemetry.util.genai.extended_types import RerankInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


def wrap_text_rerank_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for TextReRank.call.

    Uses ExtendedTelemetryHandler which supports rerank operations.

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
        # Create rerank invocation object
        invocation = RerankInvocation(
            provider=GenAI.DASHSCOPE.value, request_model=model
        )
        invocation.provider = "dashscope"

        # Start rerank invocation (creates span)
        handler.start_rerank(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Successfully complete (sets attributes and ends span)
            handler.stop_rerank(invocation)
            return result

        except Exception as e:
            # Failure handling (sets error attributes and ends span)
            error = Error(message=str(e), type=type(e))
            handler.fail_rerank(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in rerank instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)
