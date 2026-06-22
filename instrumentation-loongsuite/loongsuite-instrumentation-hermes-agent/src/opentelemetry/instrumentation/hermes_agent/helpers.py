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

"""Helper utilities for Hermes telemetry wrappers."""

from __future__ import annotations

import contextvars
import importlib
import json
from types import SimpleNamespace
from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GenAiExtendedProviderNameValues,
)
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
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

# Per-invocation state stored as a stack via ContextVar so that:
#   - Sibling concurrent calls (different asyncio tasks) get isolated state.
#   - Nested calls (parent agent → child agent) each get their own frame;
#     the child's frame is pushed on entry and the parent's frame is restored
#     on exit via ContextVar.reset(token).
_HERMES_STATE: contextvars.ContextVar["dict[str, Any] | None"] = (
    contextvars.ContextVar(
        "opentelemetry_hermes_state",
        default=None,
    )
)

_HERMES_AGENT_SYSTEM = "hermes"


def obj_get(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _normalize_platform(value: Any) -> str:
    platform = getattr(value, "value", value)
    return str(platform or "").strip().lower()


def _entry_platform(instance: Any) -> str:
    platform = _normalize_platform(getattr(instance, "platform", None))
    return platform


def to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def provider_name(instance: Any) -> str:
    provider = str(getattr(instance, "provider", "") or "").strip().lower()
    if provider in {"dashscope", "alibaba", "aliyun", "qwen"}:
        return GenAiExtendedProviderNameValues.DASHSCOPE.value
    if provider in {"openai", "openai-codex", "codex"}:
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    if provider == "deepseek":
        return GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
    if provider:
        return provider

    base_url = str(getattr(instance, "base_url", "") or "").lower()
    if "dashscope" in base_url or "aliyuncs.com" in base_url:
        return GenAiExtendedProviderNameValues.DASHSCOPE.value
    if "chatgpt.com/backend-api/codex" in base_url:
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    if "openai" in base_url:
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    if "deepseek" in base_url:
        return GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
    return "custom"


def _flatten_content(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = obj_get(item, "text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            content = obj_get(item, "content")
            if isinstance(content, str) and content:
                parts.append(content)
        flattened = "\n".join(part for part in parts if part).strip()
        return flattened or str(value)
    return str(value)


def _text_part(content: Any) -> Text | None:
    flattened = _flatten_content(content)
    if flattened is None:
        return None
    if isinstance(flattened, str) and not flattened.strip():
        return None
    return Text(content=str(flattened))


def _tool_call_part(
    *,
    call_id: Any = None,
    name: Any = None,
    arguments: Any = None,
) -> ToolCall | None:
    if not name:
        return None
    return ToolCall(
        id=str(call_id) if call_id is not None else None,
        name=str(name),
        arguments=arguments,
    )


def _tool_call_response_part(
    *,
    call_id: Any = None,
    response: Any = None,
) -> ToolCallResponse:
    return ToolCallResponse(
        id=str(call_id) if call_id is not None else None,
        response=response,
    )


def _reasoning_part(content: Any) -> Reasoning | None:
    flattened = _flatten_content(content)
    if not isinstance(flattened, str) or not flattened:
        return None
    return Reasoning(content=flattened)


def _message_finish_reason_for_output_schema(
    finish_reason: str | None,
) -> str:
    if finish_reason == "tool_calls":
        return "tool_calls"
    return finish_reason or "stop"


def _normalize_chat_message(message: Any) -> InputMessage | None:
    role = obj_get(message, "role", "")
    if not role:
        return None

    parts: list[Any] = []
    text_part = _text_part(obj_get(message, "content"))
    if text_part is not None:
        parts.append(text_part)

    for tool_call in obj_get(message, "tool_calls", []) or []:
        function = obj_get(tool_call, "function", {})
        part = _tool_call_part(
            call_id=obj_get(tool_call, "id"),
            name=obj_get(function, "name"),
            arguments=obj_get(function, "arguments"),
        )
        if part is not None:
            parts.append(part)

    tool_call_id = obj_get(message, "tool_call_id")
    if role in {"tool", "function"} and (tool_call_id or parts):
        response = _flatten_content(obj_get(message, "content"))
        parts = [
            _tool_call_response_part(
                call_id=tool_call_id,
                response=response,
            )
        ]

    if not parts:
        return None
    normalized_role = "tool" if role == "function" else str(role)
    return InputMessage(role=normalized_role, parts=parts)


def _normalize_codex_input_item(item: Any) -> InputMessage | None:
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    role = item.get("role")

    if role in {"system", "user", "assistant", "tool"}:
        if role == "tool" and item.get("tool_call_id") is not None:
            return InputMessage(
                role="tool",
                parts=[
                    _tool_call_response_part(
                        call_id=item.get("tool_call_id"),
                        response=_flatten_content(item.get("content")),
                    )
                ],
            )
        text_part = _text_part(item.get("content"))
        if text_part is None:
            return None
        return InputMessage(role=role, parts=[text_part])

    if item_type == "function_call":
        part = _tool_call_part(
            call_id=item.get("call_id") or item.get("id"),
            name=item.get("name") or item.get("function_name"),
            arguments=item.get("arguments", item.get("input", {})),
        )
        if part is None:
            return None
        return InputMessage(role="assistant", parts=[part])

    if item_type == "function_call_output":
        return InputMessage(
            role="tool",
            parts=[
                _tool_call_response_part(
                    call_id=item.get("call_id"),
                    response=_flatten_content(item.get("output")),
                )
            ],
        )

    if item_type == "message":
        text_part = _text_part(item.get("content"))
        if text_part is None:
            return None
        return InputMessage(
            role=item.get("role", "assistant"), parts=[text_part]
        )

    return None


def request_input_messages(api_kwargs: Any) -> list[InputMessage]:
    if not isinstance(api_kwargs, dict):
        return []

    messages: list[InputMessage] = []

    instructions = api_kwargs.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append(
            InputMessage(
                role="system", parts=[Text(content=instructions.strip())]
            )
        )

    raw_messages = api_kwargs.get("messages")
    if isinstance(raw_messages, list):
        for message in raw_messages:
            normalized = _normalize_chat_message(message)
            if normalized is not None:
                messages.append(normalized)

    input_items = api_kwargs.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            normalized = _normalize_codex_input_item(item)
            if normalized is not None:
                messages.append(normalized)
    elif isinstance(input_items, str) and input_items.strip():
        messages.append(
            InputMessage(
                role="user", parts=[Text(content=input_items.strip())]
            )
        )

    return messages


def tool_definitions(tools: Any) -> list[Any]:
    if not isinstance(tools, list):
        return []

    definitions: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if "function" in tool and isinstance(tool["function"], dict):
            function = tool["function"]
            name = function.get("name")
            if not name:
                continue
            definitions.append(
                FunctionToolDefinition(
                    name=name,
                    description=function.get("description"),
                    parameters=function.get("parameters"),
                )
            )
            continue

        name = tool.get("name")
        tool_type = tool.get("type")
        if name and tool_type:
            definitions.append(
                GenericToolDefinition(name=str(name), type=str(tool_type))
            )

    return definitions


def structured_response_message(instance: Any, response: Any) -> OutputMessage:
    choice = None
    try:
        choice = response.choices[0]
    except Exception:
        choice = None

    if choice is not None:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        parts: list[Any] = []
        text_part = _text_part(obj_get(message, "content"))
        if text_part is not None:
            parts.append(text_part)
        for tool_call in obj_get(message, "tool_calls", []) or []:
            function = obj_get(tool_call, "function", {})
            part = _tool_call_part(
                call_id=obj_get(tool_call, "id"),
                name=obj_get(function, "name"),
                arguments=obj_get(function, "arguments"),
            )
            if part is not None:
                parts.append(part)
        return OutputMessage(
            role="assistant",
            parts=parts,
            finish_reason=_message_finish_reason_for_output_schema(
                response_finish_reason(instance, response)
            ),
        )

    parts: list[Any] = []
    for item in obj_get(response, "output", []) or []:
        item_type = obj_get(item, "type", "")
        if item_type == "function_call":
            part = _tool_call_part(
                call_id=obj_get(item, "call_id") or obj_get(item, "id"),
                name=obj_get(item, "name") or obj_get(item, "function_name"),
                arguments=obj_get(
                    item, "arguments", obj_get(item, "input", {})
                ),
            )
            if part is not None:
                parts.append(part)
            continue

        if item_type == "reasoning":
            part = _reasoning_part(
                obj_get(item, "summary") or obj_get(item, "content")
            )
            if part is not None:
                parts.append(part)
            continue

        text_part = _text_part(obj_get(item, "content"))
        if text_part is not None:
            parts.append(text_part)
            continue

        raw_text = obj_get(item, "text") or obj_get(item, "output_text")
        if isinstance(raw_text, str) and raw_text:
            parts.append(Text(content=raw_text))

    if not parts:
        output_text = obj_get(response, "output_text")
        if isinstance(output_text, str) and output_text:
            parts.append(Text(content=output_text))

    return OutputMessage(
        role="assistant",
        parts=parts,
        finish_reason=_message_finish_reason_for_output_schema(
            response_finish_reason(instance, response)
        ),
    )


def response_output_messages(
    instance: Any, response: Any
) -> list[OutputMessage]:
    return [structured_response_message(instance, response)]


def agent_output_messages(result: Any) -> list[OutputMessage]:
    final_response = None
    if isinstance(result, dict):
        final_response = result.get("final_response")
    if not isinstance(final_response, str) or not final_response:
        return []
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=final_response)],
            finish_reason="stop",
        )
    ]


def response_message(response: Any) -> dict[str, Any]:
    structured = structured_response_message(
        SimpleNamespace(api_mode=""), response
    )
    content_parts: list[str] = []
    tool_calls: list[Any] = []
    for part in structured.parts:
        if isinstance(part, Text) and isinstance(part.content, str):
            content_parts.append(part.content)
        if isinstance(part, ToolCall):
            tool_calls.append(
                SimpleNamespace(
                    id=part.id,
                    type="function",
                    function=SimpleNamespace(
                        name=part.name,
                        arguments=part.arguments,
                    ),
                )
            )
    return {
        "role": structured.role,
        "content": "\n".join(content_parts).strip() or None,
        "tool_calls": tool_calls,
    }


def tool_call_list(response: Any) -> list[Any]:
    try:
        choice_tool_calls = response.choices[0].message.tool_calls
        if isinstance(choice_tool_calls, list):
            return choice_tool_calls
    except Exception:
        pass

    codex_tool_calls: list[Any] = []
    for item in obj_get(response, "output", []) or []:
        if obj_get(item, "type") != "function_call":
            continue
        arguments = obj_get(item, "arguments", obj_get(item, "input", {}))
        codex_tool_calls.append(
            SimpleNamespace(
                id=obj_get(item, "call_id") or obj_get(item, "id"),
                type="function",
                function=SimpleNamespace(
                    name=obj_get(item, "name")
                    or obj_get(item, "function_name"),
                    arguments=arguments,
                ),
            )
        )
    return codex_tool_calls


def canonical_usage(instance: Any, response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0

    try:
        normalize_usage = importlib.import_module(
            "agent.usage_pricing"
        ).normalize_usage
        canonical = normalize_usage(
            usage,
            provider=provider_name(instance),
            api_mode=str(getattr(instance, "api_mode", "") or "")
            .strip()
            .lower(),
        )
        input_tokens = to_int(getattr(canonical, "prompt_tokens", 0))
        output_tokens = to_int(getattr(canonical, "output_tokens", 0))
        total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens
    except Exception:
        input_tokens = to_int(obj_get(usage, "input_tokens"))
        output_tokens = to_int(obj_get(usage, "output_tokens"))
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = to_int(obj_get(usage, "prompt_tokens"))
            output_tokens = to_int(obj_get(usage, "completion_tokens"))
        total_tokens = to_int(obj_get(usage, "total_tokens")) or (
            input_tokens + output_tokens
        )
        return input_tokens, output_tokens, total_tokens


def response_finish_reason(instance: Any, response: Any) -> str | None:
    if tool_call_list(response):
        return "tool_calls"

    api_mode = str(getattr(instance, "api_mode", "") or "").strip().lower()
    if api_mode == "codex_responses":
        status = obj_get(response, "status")
        incomplete_details = obj_get(response, "incomplete_details")
        incomplete_reason = obj_get(incomplete_details, "reason")
        if status == "incomplete" and incomplete_reason in {
            "max_output_tokens",
            "length",
        }:
            return "length"
        if status == "completed" and response_message(response).get("content"):
            return "stop"
        return None

    try:
        finish_reason = response.choices[0].finish_reason
    except Exception:
        finish_reason = None
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason
    return None


def step_finish_reason(instance: Any, response: Any) -> str:
    if tool_call_list(response):
        return "tool_calls"

    finish_reason = response_finish_reason(instance, response)
    if finish_reason in {"stop", "interrupt", "length"}:
        return finish_reason

    content = response_message(response).get("content")
    if content:
        return "stop"
    return "invalid_response"


def create_agent_invocation(
    instance: Any, user_message: str
) -> InvokeAgentInvocation:
    invocation = InvokeAgentInvocation(
        provider=_HERMES_AGENT_SYSTEM,
        agent_name="Hermes",
        agent_id="Hermes",
        conversation_id=getattr(instance, "session_id", None),
        request_model=getattr(instance, "model", None),
        input_messages=[
            InputMessage(
                role="user", parts=[Text(content=str(user_message or ""))]
            )
        ],
        tool_definitions=tool_definitions(getattr(instance, "tools", None)),
    )
    user_id = getattr(instance, "_user_id", None)
    if user_id:
        invocation.attributes["gen_ai.user.id"] = user_id
    invocation.attributes["gen_ai.agent.system"] = _HERMES_AGENT_SYSTEM
    return invocation


def create_entry_invocation(
    instance: Any, user_message: str
) -> EntryInvocation:
    user_id = getattr(instance, "_user_id", None) or getattr(
        instance, "user_id", None
    )
    invocation = EntryInvocation(
        session_id=getattr(instance, "session_id", None),
        user_id=str(user_id) if user_id else None,
        input_messages=[
            InputMessage(
                role="user", parts=[Text(content=str(user_message or ""))]
            )
        ],
    )
    return invocation


def should_create_entry_for_agent(instance: Any) -> bool:
    return bool(_entry_platform(instance))


def _extract_server_info(instance: Any) -> tuple[str | None, int | None]:
    """Extract server address and port from instance base_url."""
    base_url = getattr(instance, "base_url", None)
    if not base_url:
        return None, None
    base_url_str = str(base_url).rstrip("/")
    # Handle schemeless URLs by prepending "//" for urlparse.
    if "://" not in base_url_str:
        base_url_str = "//" + base_url_str
    try:
        from urllib.parse import urlparse  # noqa: PLC0415

        parsed = urlparse(base_url_str)
        address = parsed.hostname
        if not address:
            return None, None
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return address, port
    except Exception:
        return None, None


def create_llm_invocation(instance: Any, api_kwargs: Any) -> LLMInvocation:
    if not isinstance(api_kwargs, dict):
        api_kwargs = {}

    request_model = str(
        api_kwargs.get("model") or getattr(instance, "model", "")
    )
    max_tokens = api_kwargs.get("max_tokens")
    if max_tokens is None:
        max_tokens = api_kwargs.get("max_output_tokens")

    server_address, server_port = _extract_server_info(instance)

    invocation = LLMInvocation(
        provider=provider_name(instance),
        request_model=request_model or None,
        input_messages=request_input_messages(api_kwargs),
        tool_definitions=tool_definitions(api_kwargs.get("tools")),
        conversation_id=getattr(instance, "session_id", None),
        max_tokens=max_tokens,
        temperature=api_kwargs.get("temperature"),
        top_p=api_kwargs.get("top_p"),
        frequency_penalty=api_kwargs.get("frequency_penalty"),
        presence_penalty=api_kwargs.get("presence_penalty"),
        seed=api_kwargs.get("seed"),
        stop_sequences=api_kwargs.get("stop"),
        server_address=server_address,
        server_port=server_port,
    )
    return invocation


def update_llm_invocation_from_response(
    invocation: LLMInvocation,
    instance: Any,
    response: Any,
) -> tuple[int, int, int]:
    response_model = getattr(response, "model", None)
    if response_model:
        invocation.response_model_name = response_model
    else:
        invocation.response_model_name = invocation.request_model

    response_id = getattr(response, "id", None)
    if response_id:
        invocation.response_id = response_id

    invocation.output_messages = response_output_messages(instance, response)
    finish_reason = response_finish_reason(instance, response)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]
        invocation.attributes["gen_ai.response.finish_reason"] = finish_reason
        invocation.attributes["gen_ai.response.finish_reasons"] = json.dumps(
            [finish_reason]
        )

    input_tokens, output_tokens, total_tokens = canonical_usage(
        instance, response
    )
    if input_tokens > 0:
        invocation.input_tokens = input_tokens
    if output_tokens > 0:
        invocation.output_tokens = output_tokens

    # Extract cache token usage
    usage = getattr(response, "usage", None)
    if usage is not None:
        # OpenAI-compatible: prompt_tokens_details.cached_tokens
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached = getattr(prompt_details, "cached_tokens", None)
            if cached and cached > 0:
                invocation.usage_cache_read_input_tokens = cached
        # Direct fields (some providers)
        cache_creation = getattr(
            usage, "cache_creation_input_tokens", None
        ) or getattr(usage, "cache_creation_tokens", None)
        if cache_creation and cache_creation > 0:
            invocation.usage_cache_creation_input_tokens = cache_creation
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        if (
            cache_read
            and cache_read > 0
            and not invocation.usage_cache_read_input_tokens
        ):
            invocation.usage_cache_read_input_tokens = cache_read

    return input_tokens, output_tokens, total_tokens


def create_tool_invocation(
    tool_name: str,
    *,
    provider: str | None = None,
    arguments: Any = None,
    tool_call_id: str | None = None,
) -> ExecuteToolInvocation:
    invocation = ExecuteToolInvocation(
        tool_name=tool_name,
        provider=provider or _HERMES_AGENT_SYSTEM,
    )
    invocation.tool_type = "function"
    if invocation.provider:
        invocation.attributes["gen_ai.provider.name"] = invocation.provider
    if arguments is not None:
        invocation.tool_call_arguments = arguments
    if tool_call_id:
        invocation.tool_call_id = tool_call_id
    apply_skill_attributes(invocation, tool_name, arguments=arguments)
    return invocation


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _skill_tool_name(tool_name: str) -> bool:
    return tool_name in {"skill_view", "skill_manage"}


def _skill_metadata_from_result(result: Any) -> dict[str, Any]:
    data = _json_object(result)
    if not data or data.get("success") is False:
        return {}

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    hermes_metadata = metadata.get("hermes")
    if not isinstance(hermes_metadata, dict):
        hermes_metadata = {}

    return {
        "name": data.get("name"),
        "description": data.get("description"),
        "version": data.get("version")
        or metadata.get("version")
        or hermes_metadata.get("version"),
        "id": data.get("id")
        or metadata.get("id")
        or hermes_metadata.get("id")
        or data.get("name"),
    }


def apply_skill_attributes(
    invocation: ExecuteToolInvocation,
    tool_name: str,
    *,
    arguments: Any = None,
    result: Any = None,
) -> None:
    if not _skill_tool_name(tool_name):
        return

    argument_data = _json_object(arguments)
    result_data = _skill_metadata_from_result(result)
    skill_name = result_data.get("name") or argument_data.get("name")
    skill_id = result_data.get("id") or skill_name
    skill_description = result_data.get("description")
    skill_version = result_data.get("version")

    if skill_name:
        invocation.skill_name = str(skill_name)
    if skill_id:
        invocation.skill_id = str(skill_id)
    if skill_description:
        invocation.skill_description = str(skill_description)
    if skill_version:
        invocation.skill_version = str(skill_version)


def _new_state() -> "dict[str, Any]":
    return {
        "handler": None,
        "agent_invocation": None,
        "current_step_invocation": None,
        "current_step_round": 0,
        "pending_step_finish_reason": None,
        "last_step_finish_reason": None,
        "last_response_model": None,
        "last_response_id": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "first_token_monotonic_s": None,
        "active_llm_depth": 0,
    }


def push_state(instance: Any) -> "contextvars.Token[dict[str, Any] | None]":
    """Push a fresh, isolated state frame for the current invocation.

    Returns a token that must be passed to ``reset_state`` in the
    corresponding ``finally`` block to restore the caller's frame.  This
    correctly handles nested agent calls (parent → child): each invocation
    gets its own frame, and the parent's frame is restored when the child
    returns.
    """
    return _HERMES_STATE.set(_new_state())


def reset_state(
    token: "contextvars.Token[dict[str, Any] | None]",
) -> None:
    """Restore the state to the frame that existed before ``push_state``."""
    _HERMES_STATE.reset(token)


def state(instance: Any) -> "dict[str, Any]":
    """Return the per-call telemetry state for the current execution context.

    Always call ``push_state`` at ``RunConversationWrapper`` entry before
    reading state.  If called outside that scope (e.g. in a unit test that
    has not called push_state), a fresh frame is created automatically as a
    safe fallback.
    """
    current = _HERMES_STATE.get()
    if current is None:
        current = _new_state()
        _HERMES_STATE.set(current)
    return current


def clear_state(instance: Any) -> None:
    """Drop the per-call state for the current execution context.

    Kept for backward compatibility; prefer ``reset_state(token)`` at
    ``RunConversationWrapper`` boundaries which correctly restores the parent
    frame instead of setting to None.
    """
    _HERMES_STATE.set(None)


def start_step(handler, instance: Any, finish_step) -> None:
    current_state = state(instance)
    current_state["handler"] = handler
    if current_state["current_step_invocation"] is not None:
        finish_step(
            instance,
            current_state.get("pending_step_finish_reason")
            or "invalid_response",
        )

    round_number = current_state["current_step_round"] + 1
    invocation = ReactStepInvocation(round=round_number)
    handler.start_react_step(invocation)
    current_state["current_step_invocation"] = invocation
    current_state["current_step_round"] = round_number
    current_state["pending_step_finish_reason"] = None
