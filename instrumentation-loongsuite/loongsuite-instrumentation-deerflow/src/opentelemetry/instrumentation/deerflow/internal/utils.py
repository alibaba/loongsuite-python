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

"""Identity and content helpers for DeerFlow ENTRY spans."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from opentelemetry import baggage, trace
from opentelemetry.util.genai.extended_handler import is_entry_context_active
from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    InputMessage,
    OutputMessage,
    Text,
)
from opentelemetry.util.genai.utils import (
    get_content_capturing_mode,
    is_experimental_mode,
)

from .constants import (
    DEERFLOW_ASSISTANT_ID,
    DEERFLOW_RUN_ID,
    DEERFLOW_TRACE_ID,
    DEERFLOW_TRACE_METADATA_KEY,
    GEN_AI_AGENT_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)

logger = logging.getLogger(__name__)


def non_empty_string(value: Any) -> str | None:
    """Return a stripped string value, or ``None`` for empty inputs."""
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001
        return None
    return text or None


def should_capture_content() -> bool:
    """Whether sensitive message content may be written to spans."""
    if not is_experimental_mode():
        return False
    try:
        return get_content_capturing_mode() in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    except ValueError:
        return False


def baggage_identity(key: str) -> str | None:
    """Read a non-empty GenAI identity value from current baggage."""
    return non_empty_string(baggage.get_baggage(key))


def resolve_session_id(thread_id: Any) -> str | None:
    """Apply the ENTRY session precedence contract."""
    return baggage_identity(GEN_AI_SESSION_ID) or non_empty_string(thread_id)


def resolve_user_id(deerflow_user_id: Any) -> str | None:
    """Apply the ENTRY user precedence contract."""
    return baggage_identity(GEN_AI_USER_ID) or non_empty_string(
        deerflow_user_id
    )


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("role") or message.get("type")
    else:
        role = getattr(message, "role", None) or getattr(message, "type", None)
    role_text = non_empty_string(role) or "user"
    return {
        "ai": "assistant",
        "human": "user",
    }.get(role_text, role_text)


def _message_content(message: Any) -> str | None:
    if isinstance(message, str):
        return non_empty_string(message)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if isinstance(content, str):
        return non_empty_string(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text = non_empty_string(part)
            elif isinstance(part, dict):
                text = non_empty_string(part.get("text"))
            else:
                text = non_empty_string(getattr(part, "text", None))
            if text:
                parts.append(text)
        return "".join(parts) or None
    return non_empty_string(content)


def to_input_messages(value: Any) -> list[InputMessage]:
    """Convert common DeerFlow/LangChain input shapes to GenAI messages."""
    if not should_capture_content():
        return []

    messages: Iterable[Any]
    if isinstance(value, dict):
        raw_messages = value.get("messages")
        if isinstance(raw_messages, (list, tuple)):
            messages = raw_messages
        else:
            direct = value.get("input") or value.get("query")
            messages = [direct] if direct is not None else []
    elif isinstance(value, (list, tuple)):
        messages = value
    else:
        messages = [value]

    converted: list[InputMessage] = []
    for message in messages:
        content = _message_content(message)
        if content:
            converted.append(
                InputMessage(
                    role=_message_role(message),
                    parts=[Text(content=content)],
                )
            )
    return converted


def to_output_messages(value: Any) -> list[OutputMessage]:
    """Convert a final DeerFlow response to a GenAI output message."""
    if not should_capture_content():
        return []
    content = _message_content(value)
    if not content:
        return []
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=content)],
            finish_reason="stop",
        )
    ]


def trace_id_from_sources(*sources: Any) -> str | None:
    """Resolve DeerFlow's correlation id without treating it as an OTel id."""

    def normalize(value: Any) -> str | None:
        try:
            from deerflow.trace_context import (  # noqa: PLC0415
                normalize_trace_id,
            )

            return normalize_trace_id(value)
        except (ImportError, ModuleNotFoundError):
            return non_empty_string(value)

    for source in sources:
        if isinstance(source, dict):
            value = source.get(DEERFLOW_TRACE_METADATA_KEY) or source.get(
                DEERFLOW_TRACE_ID
            )
            text = normalize(value)
            if text:
                return text
    try:
        from deerflow.trace_context import (  # noqa: PLC0415
            get_current_trace_id,
        )

        return non_empty_string(get_current_trace_id())
    except Exception:  # noqa: BLE001
        return None


def create_entry_invocation(
    *,
    thread_id: Any,
    user_id: Any,
    agent_name: Any,
    assistant_id: Any = None,
    run_id: Any = None,
    deerflow_trace_id: Any = None,
    input_value: Any = None,
) -> EntryInvocation:
    """Build a DeerFlow ENTRY invocation with stable identity attributes."""
    resolved_agent_name = non_empty_string(agent_name) or "lead-agent"
    resolved_assistant_id = non_empty_string(assistant_id)
    resolved_run_id = non_empty_string(run_id)
    resolved_trace_id = non_empty_string(deerflow_trace_id)

    attributes: dict[str, Any] = {
        GEN_AI_FRAMEWORK: "deerflow",
        GEN_AI_AGENT_NAME: resolved_agent_name,
    }
    if resolved_assistant_id:
        attributes[DEERFLOW_ASSISTANT_ID] = resolved_assistant_id
    if resolved_run_id:
        attributes[DEERFLOW_RUN_ID] = resolved_run_id
    if resolved_trace_id:
        attributes[DEERFLOW_TRACE_ID] = resolved_trace_id

    return EntryInvocation(
        session_id=resolve_session_id(thread_id),
        user_id=resolve_user_id(user_id),
        input_messages=to_input_messages(input_value),
        attributes=attributes,
    )


def has_active_host_entry() -> bool:
    """Best-effort detection of an ENTRY created by another instrumentor."""
    if is_entry_context_active():
        return True

    span = trace.get_current_span()
    if getattr(span, "name", None) == "enter_ai_application_system":
        return True

    attributes = getattr(span, "attributes", None)
    if attributes is None:
        attributes = getattr(span, "_attributes", None)
    if isinstance(attributes, dict):
        return attributes.get("gen_ai.span.kind") == "ENTRY"
    try:
        return attributes.get("gen_ai.span.kind") == "ENTRY"
    except (AttributeError, TypeError):
        return False
