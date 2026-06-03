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

from __future__ import annotations

import json
from os import environ
from typing import Any, Iterable, List, Mapping
from urllib.parse import urlparse

from httpx import URL
from openai import NotGiven
from openai._types import Omit

from opentelemetry._logs import LogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.genai.types import (
    File,
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
    Uri,
)

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)

_RESPONSE_TOOL_CALL_ITEM_TYPES = {
    "code_interpreter_call",
    "computer_call",
    "custom_tool_call",
    "file_search_call",
    "function_call",
    "image_generation_call",
    "local_shell_call",
    "mcp_call",
    "tool_call",
    "web_search_call",
}


def is_content_enabled() -> bool:
    capture_content = environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
    )

    return capture_content.lower() == "true"


def extract_tool_calls(item, capture_content):
    tool_calls = get_property_value(item, "tool_calls")
    if tool_calls is None:
        return None

    calls = []
    for tool_call in tool_calls:
        tool_call_dict = {}
        call_id = get_property_value(tool_call, "id")
        if call_id:
            tool_call_dict["id"] = call_id

        tool_type = get_property_value(tool_call, "type")
        if tool_type:
            tool_call_dict["type"] = tool_type

        func = get_property_value(tool_call, "function")
        if func:
            tool_call_dict["function"] = {}

            name = get_property_value(func, "name")
            if name:
                tool_call_dict["function"]["name"] = name

            arguments = get_property_value(func, "arguments")
            if capture_content and arguments:
                if isinstance(arguments, str):
                    arguments = arguments.replace("\n", "")
                tool_call_dict["function"]["arguments"] = arguments

        calls.append(tool_call_dict)
    return calls


def get_server_address_and_port(
    client_instance,
) -> tuple[str | None, int | None]:
    base_client = getattr(client_instance, "_client", None)
    base_url = getattr(base_client, "base_url", None)
    if not base_url:
        return None, None
    address = None
    port = None
    if isinstance(base_url, URL):
        address = base_url.host
        port = base_url.port
    elif isinstance(base_url, str):
        url = urlparse(base_url)
        address = url.hostname
        port = url.port

    if port == 443:
        port = None

    return address, port


def get_property_value(obj, property_name):
    if isinstance(obj, dict):
        return obj.get(property_name, None)

    return getattr(obj, property_name, None)


def message_to_event(message, capture_content):
    attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OPENAI.value
    }
    role = get_property_value(message, "role")
    content = get_property_value(message, "content")

    body = {}
    if capture_content and content:
        body["content"] = content
    if role == "assistant":
        tool_calls = extract_tool_calls(message, capture_content)
        if tool_calls:
            body = {"tool_calls": tool_calls}
    elif role == "tool":
        tool_call_id = get_property_value(message, "tool_call_id")
        if tool_call_id:
            body["id"] = tool_call_id

    return LogRecord(
        event_name=f"gen_ai.{role}.message",
        attributes=attributes,
        body=body if body else None,
    )


def choice_to_event(choice, capture_content):
    attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OPENAI.value
    }

    body = {
        "index": choice.index,
        "finish_reason": choice.finish_reason or "error",
    }

    if choice.message:
        message = {
            "role": (
                choice.message.role
                if choice.message and choice.message.role
                else None
            )
        }
        tool_calls = extract_tool_calls(choice.message, capture_content)
        if tool_calls:
            message["tool_calls"] = tool_calls
        content = get_property_value(choice.message, "content")
        if capture_content and content:
            message["content"] = content
        body["message"] = message

    return LogRecord(
        event_name="gen_ai.choice",
        attributes=attributes,
        body=body,
    )


def set_span_attributes(span, attributes: dict):
    for field, value in attributes.model_dump(by_alias=True).items():
        set_span_attribute(span, field, value)


def set_span_attribute(span, name, value):
    if non_numerical_value_is_set(value) is False:
        return

    span.set_attribute(name, value)


def is_streaming(kwargs):
    return non_numerical_value_is_set(kwargs.get("stream"))


def non_numerical_value_is_set(value: bool | str | NotGiven | None):
    return bool(value) and value_is_set(value)


def value_is_set(value):
    return value is not None and not isinstance(value, (NotGiven, Omit))


def get_llm_request_attributes(
    kwargs,
    client_instance,
    latest_experimental_enabled,
    operation_name=GenAIAttributes.GenAiOperationNameValues.CHAT.value,
):
    # pylint: disable=too-many-branches

    attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation_name,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: kwargs.get("model"),
    }

    if latest_experimental_enabled:
        attributes.update(
            {
                GenAIAttributes.GEN_AI_PROVIDER_NAME: (
                    GenAIAttributes.GenAiProviderNameValues.OPENAI.value
                ),
            }
        )
    else:
        attributes.update(
            {
                GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
            }
        )

    # Add chat-specific attributes only for chat operations
    if operation_name == GenAIAttributes.GenAiOperationNameValues.CHAT.value:
        attributes.update(
            {
                GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE: kwargs.get(
                    "temperature"
                ),
                GenAIAttributes.GEN_AI_REQUEST_TOP_P: kwargs.get("p")
                or kwargs.get("top_p"),
                GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS: kwargs.get(
                    "max_tokens"
                ),
                GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY: kwargs.get(
                    "presence_penalty"
                ),
                GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY: kwargs.get(
                    "frequency_penalty"
                ),
                GenAIAttributes.GEN_AI_REQUEST_SEED: kwargs.get("seed"),
            }
        )

        if (choice_count := kwargs.get("n")) is not None:
            # Only add non default, meaningful values
            if isinstance(choice_count, int) and choice_count != 1:
                attributes[GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT] = (
                    choice_count
                )

        if (stop_sequences := kwargs.get("stop")) is not None:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] = (
                stop_sequences
            )

        request_response_format_attr_key = (
            GenAIAttributes.GEN_AI_OUTPUT_TYPE
            if latest_experimental_enabled
            else GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT
        )
        if (response_format := kwargs.get("response_format")) is not None:
            # response_format may be string or object with a string in the `type` key
            if isinstance(response_format, Mapping):
                if (
                    response_format_type := response_format.get("type")
                ) is not None:
                    attributes[request_response_format_attr_key] = (
                        response_format_type
                    )
            else:
                attributes[request_response_format_attr_key] = response_format

        # service_tier can be passed directly or in extra_body (in SDK 1.26.0 it's via extra_body)
        service_tier = kwargs.get("service_tier")
        if service_tier is None:
            extra_body = kwargs.get("extra_body")
            if isinstance(extra_body, Mapping):
                service_tier = extra_body.get("service_tier")

        request_service_tier_attr_key = (
            OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER
            if latest_experimental_enabled
            else GenAIAttributes.GEN_AI_OPENAI_REQUEST_SERVICE_TIER
        )
        attributes[request_service_tier_attr_key] = (
            service_tier if service_tier != "auto" else None
        )

    # Add embeddings-specific attributes
    elif (
        operation_name
        == GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
    ):
        # Add embedding dimensions if specified
        if (dimensions := kwargs.get("dimensions")) is not None:
            # TODO: move to GEN_AI_EMBEDDINGS_DIMENSION_COUNT when 1.39.0 is baseline
            attributes["gen_ai.embeddings.dimension.count"] = dimensions

        # Add encoding format if specified
        if "encoding_format" in kwargs:
            attributes[GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS] = [
                kwargs["encoding_format"]
            ]

    address, port = get_server_address_and_port(client_instance)
    if address:
        attributes[ServerAttributes.SERVER_ADDRESS] = address
    if port:
        attributes[ServerAttributes.SERVER_PORT] = port

    # filter out values not set
    return {k: v for k, v in attributes.items() if value_is_set(v)}


def create_chat_invocation(
    kwargs,
    client_instance,
    capture_content: bool,
) -> LLMInvocation:
    # pylint: disable=too-many-branches

    llm_invocation = LLMInvocation(request_model=kwargs.get("model", ""))
    llm_invocation.provider = (
        GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    )
    llm_invocation.temperature = get_value(kwargs.get("temperature"))
    llm_invocation.top_p = get_value(kwargs.get("p") or kwargs.get("top_p"))
    llm_invocation.max_tokens = get_value(kwargs.get("max_tokens"))
    llm_invocation.presence_penalty = get_value(kwargs.get("presence_penalty"))
    llm_invocation.frequency_penalty = get_value(
        kwargs.get("frequency_penalty")
    )
    llm_invocation.seed = get_value(kwargs.get("seed"))
    if (stop_sequences := get_value(kwargs.get("stop"))) is not None:
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        llm_invocation.stop_sequences = stop_sequences

    address, port = get_server_address_and_port(client_instance)
    if address:
        llm_invocation.server_address = address
    if port:
        llm_invocation.server_port = port

    attributes = {}
    if (choice_count := get_value(kwargs.get("n"))) is not None:
        # Only add non default, meaningful values
        if isinstance(choice_count, int) and choice_count != 1:
            attributes[GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT] = (
                choice_count
            )

    if (
        response_format := get_value(kwargs.get("response_format"))
    ) is not None:
        # response_format may be string or object with a string in the `type` key
        if isinstance(response_format, Mapping):
            if (
                response_format_type := get_value(response_format.get("type"))
            ) is not None:
                attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
                    response_format_type
                )
        else:
            attributes[
                GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT
            ] = response_format

    # service_tier can be passed directly or in extra_body (in SDK 1.26.0 it's via extra_body)
    service_tier = get_value(kwargs.get("service_tier"))
    if service_tier is None:
        extra_body = get_value(kwargs.get("extra_body"))
        if isinstance(extra_body, Mapping):
            service_tier = get_value(extra_body.get("service_tier"))
    if service_tier is not None:
        attributes[OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER] = service_tier

    if len(attributes) > 0:
        llm_invocation.attributes = attributes

    if capture_content:  # optimization
        llm_invocation.input_messages = _prepare_input_messages(
            kwargs.get("messages", [])
        )
    return llm_invocation


def create_response_invocation(
    kwargs,
    client_instance,
    capture_content: bool,
) -> LLMInvocation:
    llm_invocation = LLMInvocation(request_model=kwargs.get("model", ""))
    llm_invocation.provider = (
        GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    )
    llm_invocation.temperature = get_value(kwargs.get("temperature"))
    llm_invocation.top_p = get_value(kwargs.get("top_p"))
    llm_invocation.max_tokens = get_value(kwargs.get("max_output_tokens"))

    text_config = get_value(kwargs.get("text"))
    output_type = _get_response_output_type(text_config)
    if output_type:
        llm_invocation.output_type = output_type

    address, port = get_server_address_and_port(client_instance)
    if address:
        llm_invocation.server_address = address
    if port:
        llm_invocation.server_port = port

    conversation_id = _get_response_conversation_id(kwargs.get("conversation"))
    if conversation_id:
        llm_invocation.conversation_id = conversation_id

    reasoning = get_value(kwargs.get("reasoning"))
    attributes = {}
    metric_attributes = {}

    service_tier = get_value(kwargs.get("service_tier"))
    if service_tier and service_tier != "auto":
        attributes[OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER] = service_tier
        metric_attributes[OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER] = (
            service_tier
        )

    _set_optional_attribute(
        attributes,
        "gen_ai.openai.request.previous_response_id",
        get_value(kwargs.get("previous_response_id")),
    )
    _set_optional_attribute(
        attributes,
        "gen_ai.openai.request.background",
        get_value(kwargs.get("background")),
    )
    _set_optional_attribute(
        attributes,
        "gen_ai.openai.request.store",
        get_value(kwargs.get("store")),
    )
    _set_optional_attribute(
        attributes,
        "gen_ai.openai.request.parallel_tool_calls",
        get_value(kwargs.get("parallel_tool_calls")),
    )
    _set_optional_attribute(
        attributes,
        "gen_ai.openai.request.reasoning.effort",
        _get_mapping_or_attr(reasoning, "effort"),
    )
    _set_optional_attribute(
        attributes,
        "gen_ai.openai.request.reasoning.summary",
        _get_mapping_or_attr(reasoning, "summary"),
    )

    llm_invocation.attributes = attributes
    llm_invocation.metric_attributes = metric_attributes

    if capture_content:
        llm_invocation.input_messages = _prepare_response_input_messages(
            kwargs.get("input")
        )
        if (instructions := get_value(kwargs.get("instructions"))) is not None:
            llm_invocation.system_instruction = [
                Text(content=str(instructions))
            ]

    llm_invocation.tool_definitions = _prepare_response_tool_definitions(
        kwargs.get("tools")
    )

    return llm_invocation


def _set_optional_attribute(attributes, key, value):
    if value is not None:
        attributes[key] = value


def _get_mapping_or_attr(obj, key):
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return get_value(obj.get(key))
    return get_value(getattr(obj, key, None))


def _get_response_output_type(text_config):
    response_format = _get_mapping_or_attr(text_config, "format")
    if response_format is None:
        return None
    response_format_type = _get_mapping_or_attr(response_format, "type")
    if response_format_type in {"json_schema", "json_object"}:
        return GenAIAttributes.GenAiOutputTypeValues.JSON.value
    if response_format_type:
        return response_format_type
    return None


def _get_response_conversation_id(conversation):
    if conversation is None:
        return None
    if isinstance(conversation, str):
        return conversation
    conversation_id = _get_mapping_or_attr(conversation, "id")
    if conversation_id:
        return conversation_id
    return None


def _prepare_response_tool_definitions(tools) -> list:
    if not value_is_set(tools):
        return []
    definitions = []
    for tool in tools or []:
        tool_type = get_property_value(tool, "type") or "function"
        if tool_type == "function":
            function = get_property_value(tool, "function") or tool
            definitions.append(
                FunctionToolDefinition(
                    name=get_property_value(function, "name") or "",
                    description=get_property_value(function, "description"),
                    parameters=get_property_value(function, "parameters"),
                )
            )
        else:
            definitions.append(
                GenericToolDefinition(name=str(tool_type), type=str(tool_type))
            )
    return definitions


def _prepare_response_input_messages(input_value) -> list[InputMessage]:
    if not value_is_set(input_value):
        return []
    if isinstance(input_value, str):
        return [InputMessage(role="user", parts=[Text(content=input_value)])]

    messages = []
    if not isinstance(input_value, Iterable) or isinstance(
        input_value, (str, bytes)
    ):
        return messages

    for item in input_value:
        item_type = get_property_value(item, "type")
        if item_type and item_type not in {"message", "input_message"}:
            continue
        role = get_property_value(item, "role") or "user"
        content = get_property_value(item, "content")
        parts = _prepare_response_message_parts(content)
        if parts:
            messages.append(InputMessage(role=str(role), parts=parts))
    return messages


def _prepare_response_message_parts(content) -> list:
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(content=content)]
    if not isinstance(content, Iterable) or isinstance(content, (str, bytes)):
        return [Text(content=str(content))]

    parts = []
    for part in content:
        part_type = get_property_value(part, "type")
        text = get_property_value(part, "text")
        if part_type in {"input_text", "output_text", "text"} and text:
            parts.append(Text(content=str(text)))
            continue

        image_url = get_property_value(part, "image_url")
        file_id = get_property_value(part, "file_id")
        if image_url:
            parts.append(
                Uri(mime_type=None, modality="image", uri=str(image_url))
            )
        elif file_id:
            parts.append(
                File(mime_type=None, modality="file", file_id=str(file_id))
            )
    return parts


def set_response_invocation_properties(
    invocation: LLMInvocation,
    result,
    capture_content: bool,
) -> LLMInvocation:
    if not invocation.request_model and getattr(result, "model", None):
        invocation.request_model = result.model
    if getattr(result, "model", None):
        invocation.response_model_name = result.model
    if getattr(result, "id", None):
        invocation.response_id = result.id

    status = getattr(result, "status", None)
    finish_reason = _response_finish_reason(result)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]
    if status:
        invocation.attributes["gen_ai.openai.response.status"] = status

    if getattr(result, "service_tier", None):
        invocation.attributes[
            OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER
        ] = result.service_tier
        invocation.metric_attributes[
            OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER
        ] = result.service_tier

    usage = getattr(result, "usage", None)
    if usage:
        invocation.input_tokens = _get_mapping_or_attr(usage, "input_tokens")
        invocation.output_tokens = _get_mapping_or_attr(usage, "output_tokens")
        input_details = _get_mapping_or_attr(usage, "input_tokens_details")
        cached_tokens = _get_mapping_or_attr(input_details, "cached_tokens")
        if cached_tokens is not None:
            invocation.usage_cache_read_input_tokens = cached_tokens

        output_details = _get_mapping_or_attr(usage, "output_tokens_details")
        reasoning_tokens = _get_mapping_or_attr(
            output_details, "reasoning_tokens"
        )
        if reasoning_tokens is not None:
            invocation.attributes[
                "gen_ai.usage.output_tokens_details.reasoning_tokens"
            ] = reasoning_tokens

    if capture_content:
        invocation.output_messages = _prepare_response_output_messages(result)
    return invocation


def _response_finish_reason(result) -> str | None:
    status = getattr(result, "status", None)
    if status == "completed":
        for item in getattr(result, "output", []) or []:
            item_type = get_property_value(item, "type")
            if _is_response_tool_call_item(item_type):
                return "tool_calls"
        return "stop"
    if status == "incomplete":
        details = getattr(result, "incomplete_details", None)
        reason = _get_mapping_or_attr(details, "reason")
        if reason == "content_filter":
            return "content_filter"
        return "length"
    if status in {"failed", "cancelled"}:
        return "error"
    return None


def _prepare_response_output_messages(result) -> list[OutputMessage]:
    output_messages = []
    finish_reason = _response_finish_reason(result) or "stop"
    for item in getattr(result, "output", []) or []:
        item_type = get_property_value(item, "type")
        if item_type == "message":
            parts = _prepare_response_message_parts(
                get_property_value(item, "content")
            )
            if parts:
                output_messages.append(
                    OutputMessage(
                        role=get_property_value(item, "role") or "assistant",
                        parts=parts,
                        finish_reason=finish_reason,
                    )
                )
            continue

        tool_call = _prepare_response_tool_call(item)
        if tool_call:
            output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=[tool_call],
                    finish_reason=finish_reason,
                )
            )
    return output_messages


def _prepare_response_tool_call(item):
    item_type = get_property_value(item, "type")
    if not _is_response_tool_call_item(item_type):
        return None

    call_id = (
        get_property_value(item, "call_id")
        or get_property_value(item, "id")
        or get_property_value(item, "item_id")
    )
    name = (
        get_property_value(item, "name")
        or get_property_value(item, "tool_name")
        or item_type
    )
    arguments = get_property_value(item, "arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return ToolCall(id=call_id, name=str(name), arguments=arguments)


def _is_response_tool_call_item(item_type) -> bool:
    return item_type in _RESPONSE_TOOL_CALL_ITEM_TYPES


def get_value(v: Any):
    if value_is_set(v):
        return v
    return None


def handle_span_exception(span, error: BaseException):
    span.set_status(Status(StatusCode.ERROR, str(error)))
    if span.is_recording():
        span.set_attribute(
            ErrorAttributes.ERROR_TYPE, type(error).__qualname__
        )
    span.end()


def _is_text_part(content: Any) -> bool:
    return isinstance(content, str) or (
        isinstance(content, Iterable)
        and all(isinstance(part, str) for part in content)
    )


def _prepare_input_messages(messages) -> List[InputMessage]:
    chat_messages = []
    for message in messages:
        role = get_property_value(message, "role")
        chat_message = InputMessage(role=str(role), parts=[])
        chat_messages.append(chat_message)

        content = get_property_value(message, "content")

        if role == "assistant":
            tool_calls = get_property_value(message, "tool_calls")
            if tool_calls:
                chat_message.parts += extract_tool_calls_new(tool_calls)
            if _is_text_part(content):
                chat_message.parts.append(Text(content=str(content)))

        elif role == "tool":
            tool_call_id = get_property_value(message, "tool_call_id")
            chat_message.parts.append(
                ToolCallResponse(id=tool_call_id, response=content)
            )

        else:
            # system, developer, user, fallback
            if _is_text_part(content):
                chat_message.parts.append(Text(content=str(content)))
    return chat_messages


def extract_tool_calls_new(tool_calls) -> list[ToolCall]:
    parts = []
    for tool_call in tool_calls:
        call_id = get_property_value(tool_call, "id")

        func_name = ""
        arguments = None
        func = get_property_value(tool_call, "function")
        if func:
            func_name = get_property_value(func, "name") or ""
            arguments_str = get_property_value(func, "arguments")
            if arguments_str:
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = arguments_str

        # TODO: support custom
        parts.append(ToolCall(id=call_id, name=func_name, arguments=arguments))
    return parts


def _prepare_output_messages(choices) -> List[OutputMessage]:
    output_messages = []
    for choice in choices:
        if choice.message:
            parts = []
            tool_calls = get_property_value(choice.message, "tool_calls")
            if tool_calls:
                parts += extract_tool_calls_new(tool_calls)
            content = get_property_value(choice.message, "content")
            if _is_text_part(content):
                parts.append(Text(content=str(content)))

            message = OutputMessage(
                finish_reason=choice.finish_reason or "error",
                role=(
                    choice.message.role
                    if choice.message and choice.message.role
                    else ""
                ),
                parts=parts,
            )
            output_messages.append(message)

    return output_messages
