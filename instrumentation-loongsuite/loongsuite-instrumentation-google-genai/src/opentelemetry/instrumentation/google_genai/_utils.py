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

"""Google GenAI request and response conversion for the shared GenAI util."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from google.genai import types as google_types
from google.genai.models import t as transformers

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiProviderNameValues,
)
from opentelemetry.util.genai.extended_types import EmbeddingInvocation
from opentelemetry.util.genai.types import (
    Base64Blob,
    Blob,
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    LLMInvocation,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    Uri,
)

_logger = logging.getLogger(__name__)


def _read(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _modality(mime_type: str | None) -> str:
    if not mime_type:
        return "unknown"
    return mime_type.split("/", maxsplit=1)[0]


def _role(role: str | None) -> str:
    if role == "model":
        return "assistant"
    return role or ""


def _finish_reason(reason: Any) -> str:
    if reason is None:
        return ""
    name = getattr(reason, "name", None) or str(reason)
    normalized = name.lower().removeprefix("finish_reason_")
    if normalized == "max_tokens":
        return "length"
    if normalized in {
        "safety",
        "blocklist",
        "prohibited_content",
        "spii",
        "image_safety",
    }:
        return "content_filter"
    if normalized in {
        "malformed_function_call",
        "unexpected_tool_call",
        "other",
        "unspecified",
    }:
        return "error"
    return normalized


def _part_to_message_part(part: Any, index: int) -> MessagePart | None:
    text = _read(part, "text")
    if text is not None:
        if _read(part, "thought", False):
            return Reasoning(content=str(text))
        return Text(content=str(text))

    inline_data = _read(part, "inline_data")
    if inline_data is not None:
        mime_type = _read(inline_data, "mime_type")
        data = _read(inline_data, "data", b"")
        if isinstance(data, str):
            return Base64Blob(
                mime_type=mime_type,
                modality=_modality(mime_type),
                content=data,
            )
        return Blob(
            mime_type=mime_type,
            modality=_modality(mime_type),
            content=data or b"",
        )

    file_data = _read(part, "file_data")
    if file_data is not None:
        mime_type = _read(file_data, "mime_type")
        uri = _read(file_data, "file_uri")
        if uri:
            return Uri(
                mime_type=mime_type,
                modality=_modality(mime_type),
                uri=uri,
            )

    call = _read(part, "function_call") or _read(part, "tool_call")
    if call is not None:
        name = _read(call, "name", "")
        return ToolCall(
            id=_read(call, "id") or f"{name or 'tool'}_{index}",
            name=name,
            arguments=_read(call, "args", _read(call, "arguments", {})),
        )

    response = _read(part, "function_response") or _read(part, "tool_response")
    if response is not None:
        name = _read(response, "name", "")
        return ToolCallResponse(
            id=_read(response, "id") or f"{name or 'tool'}_{index}",
            response=_read(response, "response", {}),
        )

    executable_code = _read(part, "executable_code")
    if executable_code is not None:
        code = _read(executable_code, "code")
        if code is not None:
            return Text(content=str(code))

    code_result = _read(part, "code_execution_result")
    if code_result is not None:
        output = _read(code_result, "output")
        if output is not None:
            return Text(content=str(output))

    _logger.debug("Dropping unsupported Google GenAI message part: %r", part)
    return None


def _content_to_input_message(content: Any) -> InputMessage:
    parts = [
        converted
        for index, part in enumerate(_read(content, "parts", []) or [])
        if (converted := _part_to_message_part(part, index)) is not None
    ]
    return InputMessage(role=_role(_read(content, "role")), parts=parts)


def to_input_messages(contents: Any) -> list[InputMessage]:
    return [
        _content_to_input_message(content)
        for content in transformers.t_contents(contents)
    ]


def to_system_instruction(config: Any) -> list[MessagePart]:
    system_instruction = _read(config, "system_instruction")
    if not system_instruction:
        return []
    contents = transformers.t_contents(system_instruction)
    if not contents:
        return []
    return _content_to_input_message(contents[0]).parts


def candidate_to_output_message(candidate: Any) -> OutputMessage | None:
    content = _read(candidate, "content")
    if content is None:
        return None
    message = _content_to_input_message(content)
    return OutputMessage(
        role=message.role,
        parts=message.parts,
        finish_reason=_finish_reason(_read(candidate, "finish_reason")),
    )


def to_output_messages(response: Any) -> list[OutputMessage]:
    messages = (
        candidate_to_output_message(candidate)
        for candidate in (_read(response, "candidates", []) or [])
    )
    return [message for message in messages if message is not None]


def _normalize_config(config: Any) -> Any:
    if config is None or isinstance(
        config, google_types.GenerateContentConfig
    ):
        return config
    return google_types.GenerateContentConfig.model_validate(config)


def _tool_definitions(config: Any) -> list[ToolDefinition]:
    config = _normalize_config(config)
    tools = _read(config, "tools", []) or []
    definitions: list[ToolDefinition] = []
    for tool in tools:
        if callable(tool):
            definitions.append(
                FunctionToolDefinition(
                    name=getattr(tool, "__name__", type(tool).__name__),
                    description=(getattr(tool, "__doc__", None) or "").strip()
                    or None,
                    parameters=None,
                )
            )
            continue
        if isinstance(tool, Mapping):
            try:
                tool = google_types.Tool.model_validate(tool)
            except (TypeError, ValueError):
                definitions.append(
                    GenericToolDefinition(name="unknown", type="unknown")
                )
                continue
        function_declarations = _read(tool, "function_declarations", []) or []
        for declaration in function_declarations:
            definitions.append(
                FunctionToolDefinition(
                    name=_read(declaration, "name", ""),
                    description=_read(declaration, "description"),
                    parameters=_read(declaration, "parameters_json_schema")
                    or _read(declaration, "parameters"),
                )
            )
        try:
            tool_fields = tool.model_dump(exclude_none=True)
        except (AttributeError, TypeError):
            tool_fields = {}
        for tool_type in tool_fields:
            if tool_type != "function_declarations":
                definitions.append(
                    GenericToolDefinition(name=tool_type, type=tool_type)
                )
    return definitions


def _provider(models_object: Any) -> str:
    vertexai = _read(models_object, "vertexai")
    if vertexai is None:
        vertexai = _read(_read(models_object, "_api_client"), "vertexai")
    if vertexai:
        return GenAiProviderNameValues.GCP_VERTEX_AI.value
    return GenAiProviderNameValues.GCP_GEN_AI.value


def create_llm_invocation(
    models_object: Any,
    *,
    model: str,
    contents: Any,
    config: Any,
    function_name: str,
    extra_attributes: Mapping[str, Any] | None = None,
) -> LLMInvocation:
    config_obj = _normalize_config(config)
    invocation = LLMInvocation(
        request_model=model,
        operation_name=GenAiOperationNameValues.GENERATE_CONTENT.value,
        provider=_provider(models_object),
        input_messages=to_input_messages(contents),
        system_instruction=to_system_instruction(config_obj),
        tool_definitions=_tool_definitions(config_obj),
    )
    invocation.temperature = _read(config_obj, "temperature")
    invocation.top_p = _read(config_obj, "top_p")
    invocation.top_k = _read(config_obj, "top_k")
    invocation.max_tokens = _read(config_obj, "max_output_tokens")
    invocation.stop_sequences = _read(config_obj, "stop_sequences")
    invocation.choice_count = _read(config_obj, "candidate_count")
    invocation.seed = _read(config_obj, "seed")
    invocation.frequency_penalty = _read(config_obj, "frequency_penalty")
    invocation.presence_penalty = _read(config_obj, "presence_penalty")
    response_mime_type = _read(config_obj, "response_mime_type")
    if response_mime_type == "text/plain":
        invocation.output_type = "text"
    elif response_mime_type == "application/json":
        invocation.output_type = "json"
    elif response_mime_type:
        invocation.output_type = response_mime_type
    invocation.attributes["code.function.name"] = function_name
    if extra_attributes:
        invocation.attributes.update(extra_attributes)
    return invocation


def apply_response(invocation: LLMInvocation, response: Any) -> None:
    response_model = _read(response, "model_version")
    if response_model:
        invocation.response_model_name = response_model
    response_id = _read(response, "response_id")
    if response_id:
        invocation.response_id = response_id

    usage = _read(response, "usage_metadata")
    input_tokens = _read(usage, "prompt_token_count")
    output_tokens = _read(usage, "candidates_token_count")
    cache_read_tokens = _read(usage, "cached_content_token_count")
    if input_tokens is not None:
        invocation.input_tokens = input_tokens
    if output_tokens is not None:
        invocation.output_tokens = output_tokens
    if cache_read_tokens is not None:
        invocation.usage_cache_read_input_tokens = cache_read_tokens

    finish_reasons = [
        reason
        for candidate in (_read(response, "candidates", []) or [])
        if (reason := _finish_reason(_read(candidate, "finish_reason")))
    ]
    if finish_reasons:
        invocation.finish_reasons = finish_reasons
    output_messages = to_output_messages(response)
    if output_messages:
        invocation.output_messages = output_messages


def create_embedding_invocation(
    models_object: Any,
    *,
    model: str,
    function_name: str,
) -> EmbeddingInvocation:
    return EmbeddingInvocation(
        request_model=model,
        provider=_provider(models_object),
        attributes={"code.function.name": function_name},
    )


def apply_embedding_response(
    invocation: EmbeddingInvocation, response: Any
) -> None:
    embeddings = _read(response, "embeddings", []) or []
    if embeddings:
        values = _read(embeddings[0], "values")
        if isinstance(values, list):
            invocation.dimension_count = len(values)


class ResponseAccumulator:
    """Merge delta-style streaming candidates into final output messages."""

    def __init__(self) -> None:
        self._messages: dict[int, OutputMessage] = {}

    @staticmethod
    def _append_part(parts: list[MessagePart], part: MessagePart) -> None:
        if (
            parts
            and isinstance(part, (Text, Reasoning))
            and isinstance(parts[-1], type(part))
        ):
            parts[-1].content += part.content
        else:
            parts.append(part)

    def add(self, response: Any) -> None:
        for position, candidate in enumerate(
            _read(response, "candidates", []) or []
        ):
            index = _read(candidate, "index")
            if index is None:
                index = position
            converted = candidate_to_output_message(candidate)
            if converted is None:
                continue
            current = self._messages.get(index)
            if current is None:
                current = OutputMessage(
                    role=converted.role,
                    parts=[],
                    finish_reason=converted.finish_reason,
                )
                self._messages[index] = current
            if converted.role:
                current.role = converted.role
            if converted.finish_reason:
                current.finish_reason = converted.finish_reason
            for part in converted.parts:
                self._append_part(current.parts, part)

    def output_messages(self) -> list[OutputMessage]:
        return [self._messages[index] for index in sorted(self._messages)]
