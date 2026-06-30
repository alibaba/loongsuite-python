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

from typing import Any, Iterable, Mapping, Sequence

from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
)

from .semantic_conventions import AUTOGEN_PROVIDER_NAME, GEN_AI_AGENT_NAME


def field_value(value: Any, *names: str) -> Any:
    if value is None:
        return None
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        try:
            attr_value = getattr(value, name)
        except Exception:
            attr_value = None
        if attr_value is not None:
            return attr_value
        get_method = getattr(value, "get", None)
        if callable(get_method):
            try:
                got_value = get_method(name)
            except Exception:
                got_value = None
            if got_value is not None:
                return got_value
    return None


def _type_name(value: Any) -> str:
    return type(value).__name__


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return _text(value)


def _content_parts(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(content=content)]
    if isinstance(content, list):
        parts: list[Any] = []
        for item in content:
            if isinstance(item, str):
                parts.append(Text(content=item))
                continue
            if all(
                hasattr(item, name) for name in ("id", "arguments", "name")
            ):
                parts.append(
                    ToolCall(
                        arguments=field_value(item, "arguments"),
                        name=_text(field_value(item, "name")),
                        id=field_value(item, "id"),
                    )
                )
                continue
            parts.append(Text(content=_text(item)))
        return parts
    return [Text(content=_text(content))]


def to_input_messages(messages: Sequence[Any] | None) -> list[InputMessage]:
    result: list[InputMessage] = []
    for message in messages or []:
        type_name = _type_name(message)
        if type_name == "TextMessage":
            source = _text(field_value(message, "source"))
            role = "user" if source == "user" else "assistant"
            result.append(
                InputMessage(
                    role=role,
                    parts=_content_parts(field_value(message, "content")),
                )
            )
        elif type_name == "SystemMessage":
            result.append(
                InputMessage(
                    role="system",
                    parts=[
                        Text(content=_text(field_value(message, "content")))
                    ],
                )
            )
        elif type_name == "UserMessage":
            result.append(
                InputMessage(
                    role="user",
                    parts=_content_parts(field_value(message, "content")),
                )
            )
        elif type_name == "AssistantMessage":
            parts = _content_parts(field_value(message, "content"))
            thought = field_value(message, "thought")
            if thought:
                parts.append(Reasoning(content=_text(thought)))
            result.append(InputMessage(role="assistant", parts=parts))
        elif type_name == "FunctionExecutionResultMessage":
            parts = []
            for item in field_value(message, "content") or []:
                parts.append(
                    ToolCallResponse(
                        response=field_value(item, "content"),
                        id=field_value(item, "call_id"),
                    )
                )
            result.append(InputMessage(role="tool", parts=parts))
        else:
            result.append(
                InputMessage(role="user", parts=[Text(content=_text(message))])
            )
    return result


def tool_definitions(tools: Iterable[Any] | None) -> list[Any]:
    definitions: list[Any] = []
    for tool in tools or []:
        schema = field_value(tool, "schema")
        if callable(schema):
            schema = schema()
        if isinstance(schema, Mapping):
            definitions.append(
                FunctionToolDefinition(
                    name=_text(schema.get("name")),
                    description=schema.get("description"),
                    parameters=schema.get("parameters"),
                )
            )
            continue
        name = field_value(tool, "name")
        if name is not None:
            definitions.append(
                GenericToolDefinition(name=_text(name), type="function")
            )
    return definitions


def model_name(model_client: Any) -> str:
    create_args = field_value(model_client, "_create_args", "create_args")
    if isinstance(create_args, Mapping):
        value = create_args.get("model")
        if value:
            return _text(value)
    for name in ("model", "_model", "model_name", "_model_name"):
        value = field_value(model_client, name)
        if value:
            return _text(value)
    raw_config = field_value(model_client, "_raw_config", "raw_config")
    if isinstance(raw_config, Mapping):
        value = raw_config.get("model")
        if value:
            return _text(value)
    model_info = field_value(model_client, "model_info", "_model_info")
    if isinstance(model_info, Mapping):
        value = model_info.get("model") or model_info.get("family")
        if value and _text(value).lower() != "unknown":
            return _text(value)
    return "unknown"


def response_model_name(
    model_client: Any, fallback: str | None = None
) -> str | None:
    resolved = field_value(model_client, "_resolved_model", "resolved_model")
    if resolved:
        return _text(resolved)
    return fallback


def provider_name(model_client: Any) -> str:
    cls_name = type(model_client).__name__.lower()
    module = type(model_client).__module__.lower()
    raw = f"{module}.{cls_name}"
    base_url = _text(
        field_value(model_client, "base_url", "_base_url")
        or field_value(
            field_value(model_client, "_client"), "base_url", "_base_url"
        )
    ).lower()
    if "dashscope" in base_url:
        return "dashscope"
    if "azure" in raw and "openai" in raw:
        return "azure_ai_openai"
    if "openai" in raw:
        return "openai"
    if "ollama" in raw:
        return "ollama"
    if "anthropic" in raw:
        return "anthropic"
    if "gemini" in raw or "google" in raw:
        return "gemini"
    return AUTOGEN_PROVIDER_NAME


def _finish_reason(value: Any) -> str:
    reason = _text(value or "unknown")
    if reason == "function_calls":
        return "tool_calls"
    return reason


def apply_create_result(
    invocation: LLMInvocation | InvokeAgentInvocation, result: Any
) -> None:
    raw_finish_reason = field_value(result, "finish_reason")
    finish_reason = _finish_reason(raw_finish_reason)
    if raw_finish_reason is not None:
        invocation.finish_reasons = [finish_reason]

    usage = field_value(result, "usage")
    prompt_tokens = field_value(usage, "prompt_tokens")
    completion_tokens = field_value(usage, "completion_tokens")
    try:
        if prompt_tokens is not None:
            invocation.input_tokens = int(prompt_tokens)
    except (TypeError, ValueError):
        pass
    try:
        if completion_tokens is not None:
            invocation.output_tokens = int(completion_tokens)
    except (TypeError, ValueError):
        pass

    content = field_value(result, "content")
    parts = _content_parts(content)
    thought = field_value(result, "thought")
    if thought:
        parts.append(Reasoning(content=_text(thought)))
    if parts:
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=parts,
                finish_reason=finish_reason,
            )
        ]


def make_llm_invocation(
    model_client: Any,
    messages: Sequence[Any] | None,
    tools: Iterable[Any] | None,
    *,
    agent_name: str | None = None,
    output_type: str | None = None,
) -> LLMInvocation:
    request_model = model_name(model_client)
    invocation = LLMInvocation(
        request_model=request_model,
        response_model_name=response_model_name(model_client, request_model),
        provider=provider_name(model_client),
        input_messages=to_input_messages(messages),
        tool_definitions=tool_definitions(tools),
        output_type=output_type,
    )
    if agent_name:
        invocation.attributes[GEN_AI_AGENT_NAME] = agent_name
    return invocation


def make_agent_invocation(instance: Any) -> InvokeAgentInvocation:
    name = _text(
        field_value(instance, "name", "_name") or type(instance).__name__
    )
    description = field_value(instance, "description", "_description")
    model_client = field_value(instance, "_model_client")
    return InvokeAgentInvocation(
        provider=AUTOGEN_PROVIDER_NAME,
        agent_name=name,
        agent_description=_text(description)
        if description is not None
        else None,
        request_model=model_name(model_client)
        if model_client is not None
        else None,
        response_model_name=response_model_name(
            model_client, model_name(model_client)
        )
        if model_client is not None
        else None,
        input_messages=to_input_messages(
            field_value(instance, "_system_messages") or []
        ),
        tool_definitions=tool_definitions(field_value(instance, "_tools")),
    )


def apply_agent_input(
    invocation: InvokeAgentInvocation, messages: Sequence[Any] | None
) -> None:
    invocation.input_messages = [
        *invocation.input_messages,
        *to_input_messages(messages),
    ]


def apply_agent_stream_item(
    invocation: InvokeAgentInvocation, item: Any, first_token_s: float | None
) -> float | None:
    if (
        type(item).__name__ == "ModelClientStreamingChunkEvent"
        and first_token_s is not None
        and invocation.monotonic_first_token_s is None
    ):
        invocation.monotonic_first_token_s = first_token_s

    message = field_value(item, "chat_message")
    if message is None:
        message = item

    message_type = type(message).__name__
    if message_type == "MemoryQueryEvent":
        memories = field_value(message, "content") or []
        invocation.attributes["autogen.memory.result_count"] = len(memories)
    elif message_type == "HandoffMessage":
        target = field_value(message, "target")
        source = field_value(message, "source")
        context = field_value(message, "context") or []
        if source:
            invocation.attributes["autogen.handoff.source"] = _text(source)
        if target:
            invocation.attributes["autogen.handoff.target"] = _text(target)
        invocation.attributes["autogen.handoff.context_count"] = len(context)
    elif message_type == "CodeExecutionEvent":
        result = field_value(message, "result")
        exit_code = field_value(result, "exit_code")
        try:
            if exit_code is not None:
                invocation.attributes["autogen.code.exit_code"] = int(
                    exit_code
                )
        except (TypeError, ValueError):
            pass
        invocation.attributes["autogen.code.retry_attempt"] = int(
            field_value(message, "retry_attempt") or 0
        )
    elif message_type == "CodeGenerationEvent":
        code_blocks = field_value(message, "code_blocks") or []
        invocation.attributes["autogen.code.block_count"] = len(code_blocks)
        invocation.attributes["autogen.code.retry_attempt"] = int(
            field_value(message, "retry_attempt") or 0
        )
    elif message_type == "UserInputRequestedEvent":
        request_id = field_value(message, "request_id")
        if request_id:
            invocation.attributes["autogen.user_input.request_id"] = _text(
                request_id
            )
    elif message_type == "TaskResult":
        messages = field_value(message, "messages") or []
        stop_reason = field_value(message, "stop_reason")
        invocation.attributes["autogen.team.message_count"] = len(messages)
        if stop_reason:
            invocation.attributes["autogen.team.stop_reason"] = _text(
                stop_reason
            )
        if messages:
            last_message = messages[-1]
            parts = _content_parts(field_value(last_message, "content"))
            if parts:
                invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=parts,
                        finish_reason="stop",
                    )
                ]
                invocation.finish_reasons = ["stop"]

    usage = field_value(message, "models_usage")
    prompt_tokens = field_value(usage, "prompt_tokens")
    completion_tokens = field_value(usage, "completion_tokens")
    try:
        if prompt_tokens is not None:
            invocation.input_tokens = int(prompt_tokens)
    except (TypeError, ValueError):
        pass
    try:
        if completion_tokens is not None:
            invocation.output_tokens = int(completion_tokens)
    except (TypeError, ValueError):
        pass

    content = field_value(message, "content")
    parts = _content_parts(content)
    if parts and message_type != "ToolCallRequestEvent":
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=parts,
                finish_reason="stop",
            )
        ]
        invocation.finish_reasons = ["stop"]
    return first_token_s


def make_tool_invocation(tool_call: Any) -> ExecuteToolInvocation:
    name = _text(field_value(tool_call, "name"))
    return ExecuteToolInvocation(
        tool_name=name,
        provider=AUTOGEN_PROVIDER_NAME,
        tool_call_id=field_value(tool_call, "id"),
        tool_type="function",
        tool_call_arguments=_to_jsonable(field_value(tool_call, "arguments")),
    )


def apply_tool_result(
    invocation: ExecuteToolInvocation, result: Any | None
) -> None:
    if result is None:
        return
    invocation.tool_call_result = field_value(result, "content")
    if invocation.tool_call_result is None:
        invocation.tool_call_result = _to_jsonable(result)
