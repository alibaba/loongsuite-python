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
Memory operation utilities for GenAI operations.
This module provides types and utility functions for memory operations.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import Span
from opentelemetry.util.genai.extended_memory.memory_types import (
    MemoryInvocation,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_memory_attributes import (
    GEN_AI_MEMORY_QUERY_TEXT,
    GEN_AI_MEMORY_RECORD_COUNT,
    GEN_AI_MEMORY_RECORD_ID,
    GEN_AI_MEMORY_RECORDS,
    GEN_AI_MEMORY_STORE_ID,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
)


def _get_memory_common_attributes(
    invocation: MemoryInvocation,
) -> dict[str, Any]:
    """Get common attributes defined by the upstream Memory span convention.

    Returns a dictionary of attributes.
    """
    attributes: dict[str, Any] = {}

    attributes[GenAI.GEN_AI_OPERATION_NAME] = invocation.operation
    if invocation.provider is not None:
        attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider
    if invocation.store_id is not None:
        attributes[GEN_AI_MEMORY_STORE_ID] = invocation.store_id
    if invocation.record_id is not None:
        attributes[GEN_AI_MEMORY_RECORD_ID] = invocation.record_id
    if invocation.record_count is not None:
        attributes[GEN_AI_MEMORY_RECORD_COUNT] = invocation.record_count

    return attributes


def _get_memory_content_attributes(
    invocation: MemoryInvocation,
) -> dict[str, Any]:
    """
    Get memory content attributes (input/output messages).
    This is a controlled operation that only records content when:
    - Experimental mode is enabled
    - Content capturing mode is SPAN_ONLY or SPAN_AND_EVENT

    Returns empty dict if not in experimental mode or content capturing is disabled.
    """
    attributes: dict[str, Any] = {}

    if not is_experimental_mode() or get_content_capturing_mode() not in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return attributes

    if invocation.query_text is not None:
        attributes[GEN_AI_MEMORY_QUERY_TEXT] = invocation.query_text
    if invocation.records is not None:
        attributes[GEN_AI_MEMORY_RECORDS] = (
            invocation.records
            if isinstance(invocation.records, str)
            else gen_ai_json_dumps(invocation.records)
        )

    return attributes


def _apply_memory_finish_attributes(
    span: Span, invocation: MemoryInvocation
) -> None:
    """Apply attributes for memory operations."""
    span.update_name(invocation.operation)

    span.set_attribute(GEN_AI_SPAN_KIND, GenAiSpanKindValues.MEMORY.value)

    # Build all attributes
    attributes: dict[str, Any] = {}
    attributes.update(_get_memory_common_attributes(invocation))

    # Recommended attributes
    if invocation.server_address is not None:
        attributes[ServerAttributes.SERVER_ADDRESS] = invocation.server_address
    if invocation.server_port is not None:
        attributes[ServerAttributes.SERVER_PORT] = invocation.server_port

    attributes.update(_get_memory_content_attributes(invocation))

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


__all__ = [
    "MemoryInvocation",
    "_apply_memory_finish_attributes",
]
