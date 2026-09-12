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

"""
Embedding wrapper for LiteLLM instrumentation.
"""

import os
from typing import Callable

from opentelemetry import context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.litellm._utils import (
    apply_litellm_embedding_response_to_invocation,
    create_embedding_invocation_from_litellm,
    normalize_litellm_embedding_kwargs,
)
from opentelemetry.util.genai.types import Error


def _is_instrumentation_enabled() -> bool:
    """Check if instrumentation is enabled via environment variable."""
    enabled = os.getenv("ENABLE_LITELLM_INSTRUMENTOR", "true").lower()
    return enabled != "false"


class EmbeddingWrapper:
    """Wrapper for litellm.embedding()"""

    def __init__(self, handler, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    def __call__(self, *args, **kwargs):
        """Wrap litellm.embedding()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return self.original_func(*args, **kwargs)

        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_func(*args, **kwargs)

        request_kwargs = normalize_litellm_embedding_kwargs(
            self.original_func, args, kwargs
        )
        invocation = create_embedding_invocation_from_litellm(**request_kwargs)

        # Start Embedding invocation
        self._handler.start_embedding(invocation)

        try:
            # Call original function
            response = self.original_func(*args, **kwargs)

            apply_litellm_embedding_response_to_invocation(
                invocation, response
            )

            # End Embedding invocation successfully
            self._handler.stop_embedding(invocation)

            return response

        except Exception as e:
            # Fail Embedding invocation
            self._handler.fail_embedding(
                invocation, Error(message=str(e), type=type(e))
            )
            raise


class AsyncEmbeddingWrapper:
    """Wrapper for litellm.aembedding()"""

    def __init__(self, handler, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    async def __call__(self, *args, **kwargs):
        """Wrap litellm.aembedding()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return await self.original_func(*args, **kwargs)

        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await self.original_func(*args, **kwargs)

        request_kwargs = normalize_litellm_embedding_kwargs(
            self.original_func, args, kwargs
        )
        invocation = create_embedding_invocation_from_litellm(**request_kwargs)

        # Start Embedding invocation
        self._handler.start_embedding(invocation)

        try:
            # Call original function
            response = await self.original_func(*args, **kwargs)

            apply_litellm_embedding_response_to_invocation(
                invocation, response
            )

            # End Embedding invocation successfully
            self._handler.stop_embedding(invocation)

            return response

        except Exception as e:
            # Fail Embedding invocation
            self._handler.fail_embedding(
                invocation, Error(message=str(e), type=type(e))
            )
            raise
