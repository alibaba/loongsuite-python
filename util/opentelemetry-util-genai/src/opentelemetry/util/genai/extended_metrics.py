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
Extended metrics recorder for GenAI invocations.

This module provides LoongSuite GenAI metrics recording.
It supports multiple GenAI invocation types: chat, generate_content, embedding, execute_tool, invoke_agent,
create_agent, retrieval, and rerank.

This is just an empty implementation for now, which is a placeholder for enterprise implementation.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from opentelemetry.trace import Span
from opentelemetry.util.genai.extended_memory.memory_types import (
    MemoryInvocation,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
    RerankInvocation,
    RetrievalInvocation,
)
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import LLMInvocation

_logger = logging.getLogger(__name__)


class ExtendedInvocationMetricsRecorder(InvocationMetricsRecorder):
    """
    Extended metrics recorder that supports multiple GenAI invocation types.

    This class provides LoongSuite GenAI metrics recording.
    It supports:
    - Chat/Generate content operations
    - Embedding operations
    - Execute tool operations
    - Invoke agent operations
    - Create agent operations
    - Retrieval operations
    - Rerank documents operations
    - Entry operations
    - ReAct Step operations
    """

    def record_extended(
        self,
        span: Optional[Span],
        invocation: Union[
            LLMInvocation,
            EmbeddingInvocation,
            ExecuteToolInvocation,
            InvokeAgentInvocation,
            CreateAgentInvocation,
            RetrievalInvocation,
            RerankInvocation,
            MemoryInvocation,
            EntryInvocation,
            ReactStepInvocation,
        ],
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record duration and token metrics for any GenAI invocation type.

        This method automatically routes to the appropriate handler based on
        the invocation type.

        Args:
            span: The span associated with this invocation
            invocation: The invocation object (any supported type)
            error_type: Optional error type if the invocation failed
        """
        if isinstance(invocation, LLMInvocation):
            self.record(span, invocation, error_type=error_type)
            return
        # TODO: Implement extended metrics recorder


__all__ = ["ExtendedInvocationMetricsRecorder"]
