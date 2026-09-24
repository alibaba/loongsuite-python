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

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.context import get_current
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import Span
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.util.genai.extended_memory.memory_types import (
    MemoryInvocation,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_memory_attributes import (
    GEN_AI_MEMORY_AGENT_ID,
    GEN_AI_MEMORY_APP_ID,
    GEN_AI_MEMORY_ID,
    GEN_AI_MEMORY_INPUT_MESSAGES,
    GEN_AI_MEMORY_LIMIT,
    GEN_AI_MEMORY_MEMORY_TYPE,
    GEN_AI_MEMORY_OPERATION,
    GEN_AI_MEMORY_OUTPUT_MESSAGES,
    GEN_AI_MEMORY_PAGE,
    GEN_AI_MEMORY_PAGE_SIZE,
    GEN_AI_MEMORY_RERANK,
    GEN_AI_MEMORY_RUN_ID,
    GEN_AI_MEMORY_THRESHOLD,
    GEN_AI_MEMORY_TOP_K,
    GEN_AI_MEMORY_USER_ID,
)
from opentelemetry.util.genai.types import Error
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
    should_emit_event,
)


def _get_memory_common_attributes(
    invocation: MemoryInvocation,
) -> dict[str, Any]:
    """Get common memory attributes (operation_name, operation type, identifiers).

    Returns a dictionary of attributes.
    """
    attributes: dict[str, Any] = {}

    # Operation name
    attributes[GenAI.GEN_AI_OPERATION_NAME] = "memory_operation"

    # Required: memory operation type
    attributes[GEN_AI_MEMORY_OPERATION] = invocation.operation

    # Conditionally required identifiers
    if invocation.user_id is not None:
        attributes[GEN_AI_MEMORY_USER_ID] = invocation.user_id
    if invocation.agent_id is not None:
        attributes[GEN_AI_MEMORY_AGENT_ID] = invocation.agent_id
    if invocation.run_id is not None:
        attributes[GEN_AI_MEMORY_RUN_ID] = invocation.run_id
    if invocation.app_id is not None:
        attributes[GEN_AI_MEMORY_APP_ID] = invocation.app_id

    return attributes


def _get_memory_parameter_attributes(
    invocation: MemoryInvocation,
) -> dict[str, Any]:
    """Get memory operation parameter attributes.

    Returns a dictionary of attributes.
    """
    attributes: dict[str, Any] = {}

    # Optional parameters
    if invocation.memory_id is not None:
        attributes[GEN_AI_MEMORY_ID] = invocation.memory_id
    if invocation.limit is not None:
        attributes[GEN_AI_MEMORY_LIMIT] = invocation.limit
    if invocation.page is not None:
        attributes[GEN_AI_MEMORY_PAGE] = invocation.page
    if invocation.page_size is not None:
        attributes[GEN_AI_MEMORY_PAGE_SIZE] = invocation.page_size
    if invocation.top_k is not None:
        attributes[GEN_AI_MEMORY_TOP_K] = invocation.top_k
    if invocation.memory_type is not None:
        attributes[GEN_AI_MEMORY_MEMORY_TYPE] = invocation.memory_type
    if invocation.threshold is not None:
        attributes[GEN_AI_MEMORY_THRESHOLD] = invocation.threshold
    if invocation.rerank is not None:
        attributes[GEN_AI_MEMORY_RERANK] = invocation.rerank

    return attributes


def _get_memory_content_attributes(
    invocation: MemoryInvocation,
    for_span: bool = True,
) -> dict[str, Any]:
    """
    Get memory content attributes (input/output messages).
    This is a controlled operation that only records content when:
    - Experimental mode is enabled
    - For spans: content capturing mode is SPAN_ONLY or SPAN_AND_EVENT
    - For events: content capturing mode is EVENT_ONLY or SPAN_AND_EVENT

    Args:
        invocation: The memory invocation
        for_span: If True, check for span content capturing mode; if False, check for event mode

    Returns empty dict if not in experimental mode or content capturing is disabled.
    """
    attributes: dict[str, Any] = {}

    # Check experimental mode and content capturing mode based on usage
    if not is_experimental_mode():
        return attributes

    if for_span:
        # For spans: only capture if SPAN_ONLY or SPAN_AND_EVENT
        if get_content_capturing_mode() not in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            return attributes
    else:
        # For events: only capture if EVENT_ONLY or SPAN_AND_EVENT
        if get_content_capturing_mode() not in (
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            return attributes

    if invocation.input_messages is not None:
        if isinstance(invocation.input_messages, str):
            attributes[GEN_AI_MEMORY_INPUT_MESSAGES] = (
                invocation.input_messages
            )
        else:
            attributes[GEN_AI_MEMORY_INPUT_MESSAGES] = gen_ai_json_dumps(
                invocation.input_messages
            )

    if invocation.output_messages is not None:
        if isinstance(invocation.output_messages, str):
            attributes[GEN_AI_MEMORY_OUTPUT_MESSAGES] = (
                invocation.output_messages
            )
        else:
            attributes[GEN_AI_MEMORY_OUTPUT_MESSAGES] = gen_ai_json_dumps(
                invocation.output_messages
            )

    return attributes


def _apply_memory_finish_attributes(
    span: Span, invocation: MemoryInvocation
) -> None:
    """Apply attributes for memory operations."""
    # Update span name with actual operation
    if invocation.operation:
        span.update_name(f"memory_operation {invocation.operation}")
    else:
        span.update_name("memory_operation")

    span.set_attribute(GEN_AI_SPAN_KIND, GenAiSpanKindValues.MEMORY.value)

    # Build all attributes
    attributes: dict[str, Any] = {}
    attributes.update(_get_memory_common_attributes(invocation))
    attributes.update(_get_memory_parameter_attributes(invocation))

    # Recommended attributes
    if invocation.server_address is not None:
        attributes[ServerAttributes.SERVER_ADDRESS] = invocation.server_address
    if invocation.server_port is not None:
        attributes[ServerAttributes.SERVER_PORT] = invocation.server_port

    # Content attributes (controlled by content capturing mode)
    # For spans, only capture if SPAN_ONLY or SPAN_AND_EVENT
    attributes.update(
        _get_memory_content_attributes(invocation, for_span=True)
    )

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


def _maybe_emit_memory_event(
    logger: Logger | None,
    span: Span,
    invocation: MemoryInvocation,
    error: Error | None = None,
) -> None:
    """Emit a gen_ai.memory.operation.details event to the logger.

    This function creates a LogRecord event for memory operations following
    the semantic convention for gen_ai.memory.operation.details.
    """
    if not is_experimental_mode() or not should_emit_event() or logger is None:
        return

    # Build event attributes
    attributes: dict[str, Any] = {}
    attributes.update(_get_memory_common_attributes(invocation))
    attributes.update(_get_memory_parameter_attributes(invocation))

    # Content attributes for events (controlled by content capturing mode)
    attributes.update(
        _get_memory_content_attributes(invocation, for_span=False)
    )

    # Add error.type if operation ended in error
    if error is not None:
        attributes[ErrorAttributes.ERROR_TYPE] = error.type.__qualname__

    # Create and emit the event with span context
    context = set_span_in_context(span, get_current())
    event = LogRecord(
        event_name="gen_ai.memory.operation.details",
        attributes=attributes,
        context=context,
    )
    logger.emit(event)


__all__ = [
    "MemoryInvocation",
    "_apply_memory_finish_attributes",
    "_maybe_emit_memory_event",
]
