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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from google.genai import types as genai_types

from opentelemetry.util.genai.types import (
    Blob,
    FinishReason,
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCallResponse,
    Uri,
)
from opentelemetry.util.genai.types import (
    ToolCall as ToolCallRequest,
)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


_logger = logging.getLogger(__name__)


@dataclass
class _StreamCandidateState:
    role: str = ""
    finish_reason: Union[FinishReason, str] = ""
    parts: List[MessagePart] = field(default_factory=list)
    position_slots: Dict[int, Tuple[int, str]] = field(default_factory=dict)


class StreamOutputMessageAccumulator:
    """Merge Google GenAI streaming candidates into logical output messages."""

    def __init__(self) -> None:
        self._candidate_states: Dict[int, _StreamCandidateState] = {}

    def add_candidates(
        self,
        candidates: List[genai_types.Candidate],
    ) -> None:
        for fallback_index, candidate in enumerate(candidates):
            candidate_index = candidate.index
            if not isinstance(candidate_index, int):
                candidate_index = fallback_index
            state = self._candidate_states.setdefault(
                candidate_index,
                _StreamCandidateState(),
            )
            self._add_candidate(state, candidate)

    def to_output_messages(self) -> List[OutputMessage]:
        messages = []
        for candidate_index in sorted(self._candidate_states):
            state = self._candidate_states[candidate_index]
            if not state.parts:
                continue
            messages.append(
                OutputMessage(
                    role=state.role,
                    parts=state.parts,
                    finish_reason=state.finish_reason,
                )
            )
        return messages

    @staticmethod
    def _add_candidate(
        state: _StreamCandidateState,
        candidate: genai_types.Candidate,
    ) -> None:
        finish_reason = _to_finish_reason(candidate.finish_reason)
        if finish_reason:
            state.finish_reason = finish_reason

        content = candidate.content
        if content is None:
            return
        role = _to_role(content.role)
        if role:
            state.role = role

        for position, provider_part in enumerate(content.parts or []):
            part = _to_part(provider_part, position)
            if part is None:
                continue
            part_kind = getattr(part, "type", type(part).__name__)
            previous = state.position_slots.get(position)
            if previous is not None:
                slot, previous_kind = previous
                if previous_kind == part_kind and _merge_stream_parts(
                    state.parts[slot],
                    part,
                ):
                    continue

            slot = len(state.parts)
            state.parts.append(part)
            state.position_slots[position] = (slot, part_kind)


def _merge_stream_parts(existing: MessagePart, incoming: MessagePart) -> bool:
    if isinstance(existing, Text) and isinstance(incoming, Text):
        existing.content += incoming.content
        return True
    if isinstance(existing, Reasoning) and isinstance(incoming, Reasoning):
        existing.content += incoming.content
        return True
    if isinstance(existing, ToolCallRequest) and isinstance(
        incoming, ToolCallRequest
    ):
        if (existing.id and incoming.id and existing.id != incoming.id) or (
            existing.name and incoming.name and existing.name != incoming.name
        ):
            return False
        existing.id = incoming.id or existing.id
        existing.name = incoming.name or existing.name
        existing.arguments = _merge_stream_values(
            existing.arguments,
            incoming.arguments,
        )
        return True
    if isinstance(existing, ToolCallResponse) and isinstance(
        incoming, ToolCallResponse
    ):
        if existing.id and incoming.id and existing.id != incoming.id:
            return False
        existing.id = incoming.id or existing.id
        existing.response = _merge_stream_values(
            existing.response,
            incoming.response,
        )
        return True
    if isinstance(existing, Blob) and isinstance(incoming, Blob):
        return existing == incoming
    if isinstance(existing, Uri) and isinstance(incoming, Uri):
        return existing == incoming
    return False


def _merge_stream_values(existing: Any, incoming: Any) -> Any:
    if incoming in (None, "", {}, []):
        return existing
    if existing in (None, "", {}, []):
        return incoming
    if isinstance(existing, dict) and isinstance(incoming, dict):
        return {**existing, **incoming}
    if isinstance(existing, str) and isinstance(incoming, str):
        return existing + incoming
    return incoming


def to_input_messages(
    *,
    contents: List[genai_types.Content],
) -> List[InputMessage]:
    return [_to_input_message(content) for content in contents]


def to_output_messages(
    *,
    candidates: List[genai_types.Candidate],
) -> List[OutputMessage]:
    def content_to_output_message(
        candidate: genai_types.Candidate,
    ) -> Optional[OutputMessage]:
        if not candidate.content:
            return None

        message = _to_input_message(candidate.content)
        return OutputMessage(
            finish_reason=_to_finish_reason(candidate.finish_reason),
            role=message.role,
            parts=message.parts,
        )

    messages = (
        content_to_output_message(candidate) for candidate in candidates
    )
    return [message for message in messages if message is not None]


def to_system_instructions(
    *,
    content: genai_types.Content,
) -> List[MessagePart]:
    parts = (
        _to_part(part, idx) for idx, part in enumerate(content.parts or [])
    )
    return [part for part in parts if part is not None]


def _to_input_message(
    content: genai_types.Content,
) -> InputMessage:
    parts = (
        _to_part(part, idx) for idx, part in enumerate(content.parts or [])
    )
    return InputMessage(
        role=_to_role(content.role),
        # filter Nones
        parts=[part for part in parts if part is not None],
    )


def _to_part(part: genai_types.Part, idx: int) -> Optional[MessagePart]:
    def tool_call_id(name: Optional[str]) -> str:
        if name:
            return f"{name}_{idx}"
        return f"{idx}"

    if (text := part.text) is not None:
        if getattr(part, "thought", False):
            return Reasoning(content=text)
        return Text(content=text)

    if inline_data := part.inline_data:
        mime_type = inline_data.mime_type or ""
        modality = mime_type.split("/")[0] if mime_type else ""
        return Blob(
            mime_type=mime_type,
            modality=modality,
            content=inline_data.data or b"",
        )

    if file_data := part.file_data:
        mime_type = file_data.mime_type or ""
        modality = mime_type.split("/")[0] if mime_type else ""
        return Uri(
            mime_type=mime_type,
            modality=modality,
            uri=file_data.file_uri or "",
        )

    if call := part.function_call:
        return ToolCallRequest(
            id=call.id or tool_call_id(call.name),
            name=call.name or "",
            arguments=call.args,
        )

    if response := part.function_response:
        return ToolCallResponse(
            id=response.id or tool_call_id(response.name),
            response=response.response,
        )

    _logger.info("Unknown part dropped from telemetry %s", part)
    return None


def _to_role(role: Optional[str]) -> str:
    if role == "user":
        return Role.USER.value
    if role == "model":
        return Role.ASSISTANT.value
    return ""


def _to_finish_reason(
    finish_reason: Optional[genai_types.FinishReason],
) -> Union[FinishReason, str]:
    if finish_reason is None:
        return ""
    if (
        finish_reason is genai_types.FinishReason.FINISH_REASON_UNSPECIFIED
        or finish_reason is genai_types.FinishReason.OTHER
    ):
        return "error"
    if finish_reason is genai_types.FinishReason.STOP:
        return "stop"
    if finish_reason is genai_types.FinishReason.MAX_TOKENS:
        return "length"

    # If there is no 1:1 mapping to an OTel preferred enum value, use the exact vertex reason
    return finish_reason.name
