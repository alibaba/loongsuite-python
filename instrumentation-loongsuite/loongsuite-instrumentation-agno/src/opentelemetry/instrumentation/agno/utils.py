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
from contextvars import ContextVar, Token
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from opentelemetry import baggage
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)
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

_PROVIDER = "agno"
_SKILL_TOOL_NAMES = {
    "get_skill_instructions",
    "get_skill_reference",
    "get_skill_script",
}
_CURRENT_AGNO_RUN_IDENTITY: ContextVar[dict[str, str] | None] = ContextVar(
    "agno_current_run_identity",
    default=None,
)


def _identity_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_identity_value(*values: Any) -> str | None:
    for value in values:
        identity = _identity_value(value)
        if identity is not None:
            return identity
    return None


def _current_baggage_value(key: str) -> str | None:
    try:
        value = baggage.get_baggage(key)
    except Exception:
        return None
    return _identity_value(value)


def entry_baggage_identity_attributes() -> dict[str, str]:
    attributes: dict[str, str] = {}
    session_id = _current_baggage_value(GEN_AI_SESSION_ID)
    user_id = _current_baggage_value(GEN_AI_USER_ID)
    if session_id:
        attributes[GEN_AI_SESSION_ID] = session_id
    if user_id:
        attributes[GEN_AI_USER_ID] = user_id
    return attributes


def _framework_identity_attributes(
    agent: Any, arguments: Mapping[str, Any]
) -> dict[str, str]:
    attributes: dict[str, str] = {}
    user_id = _first_identity_value(
        arguments.get("user_id"),
        getattr(agent, "user_id", None),
    )
    session_id = _first_identity_value(
        arguments.get("session_id"),
        getattr(agent, "session_id", None),
    )
    if user_id:
        attributes[GEN_AI_USER_ID] = user_id
    if session_id:
        attributes[GEN_AI_SESSION_ID] = session_id
    return attributes


def agno_run_identity_attributes(
    agent: Any, arguments: Mapping[str, Any]
) -> dict[str, str]:
    attributes = _framework_identity_attributes(agent, arguments)
    attributes.update(entry_baggage_identity_attributes())
    return attributes


def set_current_agno_run_identity(
    attributes: Mapping[str, Any],
) -> Token[dict[str, str] | None]:
    identity = {
        key: value
        for key in (GEN_AI_SESSION_ID, GEN_AI_USER_ID)
        if (value := _identity_value(attributes.get(key))) is not None
    }
    return _CURRENT_AGNO_RUN_IDENTITY.set(identity or None)


def reset_current_agno_run_identity(
    token: Token[dict[str, str] | None],
) -> None:
    _CURRENT_AGNO_RUN_IDENTITY.reset(token)


def _current_identity_attributes() -> dict[str, str]:
    attributes: dict[str, str] = {}
    run_identity = _CURRENT_AGNO_RUN_IDENTITY.get()
    if run_identity:
        attributes.update(run_identity)
    # Re-read baggage so ENTRY identity stays highest priority for child spans
    # even if the active OTel context changes while an Agno run is in progress.
    attributes.update(entry_baggage_identity_attributes())
    return attributes


def _apply_current_identity(invocation: Any) -> str | None:
    attributes = _current_identity_attributes()
    for key, value in attributes.items():
        invocation.attributes[key] = value
    return attributes.get(GEN_AI_SESSION_ID)


def _json_default(value: Any) -> Any:
    data = _to_dict(value)
    if data is not None:
        return data
    return f"<{type(value).__name__}>"


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    except Exception:
        return f"<{type(value).__name__}>"


def _to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            try:
                return model_dump()
            except Exception:
                return None
        except Exception:
            return None
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return None
    return None


def _mapping_from_any(value: Any) -> Dict[str, Any]:
    data = _to_dict(value)
    if data is not None:
        return data
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_skill_metadata(value: Any) -> Dict[str, Any]:
    data = _mapping_from_any(value)
    if not data:
        return {}
    frontmatter = _mapping_from_any(data.get("frontmatter"))
    metadata = _mapping_from_any(data.get("metadata")) or _mapping_from_any(
        frontmatter.get("metadata")
    )
    return {
        "name": _first_text(
            data.get("skill_name"),
            data.get("skill"),
            data.get("name"),
            frontmatter.get("name"),
        ),
        "id": _first_text(
            data.get("skill_id"),
            data.get("id"),
            metadata.get("id"),
        ),
        "description": _first_text(
            data.get("skill_description"),
            data.get("description"),
            frontmatter.get("description"),
        ),
        "version": _first_text(
            data.get("skill_version"),
            data.get("version"),
            metadata.get("version"),
        ),
    }


def _apply_skill_metadata(
    invocation: ExecuteToolInvocation, metadata: Mapping[str, Any]
) -> None:
    name = _first_text(metadata.get("name"))
    if name:
        invocation.skill_name = invocation.skill_name or name
        invocation.skill_id = (
            invocation.skill_id or _first_text(metadata.get("id")) or name
        )
    elif metadata.get("id"):
        invocation.skill_id = invocation.skill_id or _first_text(
            metadata.get("id")
        )

    description = _first_text(metadata.get("description"))
    if description:
        invocation.skill_description = (
            invocation.skill_description or description
        )

    version = _first_text(metadata.get("version"))
    if version:
        invocation.skill_version = invocation.skill_version or version


def _apply_agno_skill_tool_metadata(
    invocation: ExecuteToolInvocation, data: Any
) -> None:
    if invocation.tool_name not in _SKILL_TOOL_NAMES:
        return
    _apply_skill_metadata(invocation, _extract_skill_metadata(data))


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    data = _to_dict(value)
    if data is not None:
        return _json_dumps(data)
    if isinstance(value, (list, tuple, dict)):
        return _json_dumps(value)
    return str(value)


def _text_parts(content: Any) -> list[Text]:
    if content is None:
        return []
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _stringify(item)
            if text:
                parts.append(Text(content=text))
        return parts
    text = _stringify(content)
    return [Text(content=text)] if text else []


def _normalize_tool_arguments(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except Exception:
            return value
    data = _to_dict(value)
    if data is not None:
        return data
    return value


def _tool_call_arguments(tool_call_dict: Mapping[str, Any]) -> Any:
    function = tool_call_dict.get("function") or {}
    arguments = (
        function.get("arguments")
        if isinstance(function, Mapping) and "arguments" in function
        else tool_call_dict.get("arguments")
    )
    return _normalize_tool_arguments(arguments)


def _message_to_input(message: Any) -> InputMessage:
    role = _get_value(message, "role") or "user"
    content = _get_value(message, "content")
    tool_call_id = _get_value(message, "tool_call_id")
    parts = [] if tool_call_id else _text_parts(content)

    tool_calls = _get_value(message, "tool_calls") or []
    for tool_call in tool_calls:
        tool_call_dict = _to_dict(tool_call) or {}
        function = tool_call_dict.get("function") or {}
        parts.append(
            ToolCall(
                id=tool_call_dict.get("id"),
                name=(
                    function.get("name")
                    if isinstance(function, Mapping)
                    else None
                )
                or tool_call_dict.get("name")
                or "",
                arguments=_tool_call_arguments(tool_call_dict),
            )
        )

    if tool_call_id:
        parts.append(
            ToolCallResponse(
                id=tool_call_id,
                response=content,
            )
        )

    return InputMessage(role=role, parts=parts)


def _finish_reason(value: Any) -> str | None:
    reasons = _get_value(value, "finish_reasons")
    if (
        isinstance(reasons, Sequence)
        and not isinstance(reasons, str)
        and reasons
    ):
        reason = str(reasons[0])
        return None if reason.lower() in {"", "none", "null"} else reason
    for name in ("finish_reason", "done_reason", "stop_reason"):
        reason = _get_value(value, name)
        if reason is not None:
            reason = str(reason)
            return None if reason.lower() in {"", "none", "null"} else reason
    return None


def _system_instruction_parts(agent: Any) -> list[Text]:
    values = []
    for name in ("system_message", "instructions"):
        value = getattr(agent, name, None)
        if value:
            values.append(value)
    parts = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for item in value:
                text = _stringify(item)
                if text:
                    parts.append(Text(content=text))
            continue
        text = _stringify(value)
        if text:
            parts.append(Text(content=text))
    return parts


def _message_to_output(message: Any) -> OutputMessage:
    role = _get_value(message, "role") or "assistant"
    parts = _text_parts(_get_value(message, "content"))

    reasoning = _get_value(message, "reasoning_content") or _get_value(
        message, "redacted_reasoning_content"
    )
    if reasoning:
        parts.append(Reasoning(content=str(reasoning)))

    tool_calls = _get_value(message, "tool_calls") or []
    for tool_call in tool_calls:
        tool_call_dict = _to_dict(tool_call) or {}
        function = tool_call_dict.get("function") or {}
        parts.append(
            ToolCall(
                id=tool_call_dict.get("id"),
                name=(
                    function.get("name")
                    if isinstance(function, Mapping)
                    else None
                )
                or tool_call_dict.get("name")
                or "",
                arguments=_tool_call_arguments(tool_call_dict),
            )
        )

    if not parts:
        parts.append(Text(content=""))

    return OutputMessage(
        role=role,
        parts=parts,
        finish_reason=_finish_reason(message),
    )


def convert_agent_input(value: Any) -> list[InputMessage]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        messages = []
        for item in value:
            if hasattr(item, "role") and hasattr(item, "content"):
                messages.append(_message_to_input(item))
            elif isinstance(item, Mapping) and "role" in item:
                messages.append(_message_to_input(item))
            else:
                messages.append(
                    InputMessage(role="user", parts=_text_parts(item))
                )
        return messages
    if isinstance(value, Mapping) and "role" in value:
        return [_message_to_input(value)]
    return [InputMessage(role="user", parts=_text_parts(value))]


def convert_model_messages(messages: Any) -> list[InputMessage]:
    if not messages:
        return []
    return [
        _message_to_input(message)
        if hasattr(message, "role")
        or (isinstance(message, Mapping) and "role" in message)
        else InputMessage(role="user", parts=_text_parts(message))
        for message in messages
    ]


def convert_tool_definitions(tools: Any) -> list[Any]:
    if not tools:
        return []
    definitions = []
    for tool in tools:
        if hasattr(tool, "name"):
            definitions.append(
                FunctionToolDefinition(
                    name=getattr(tool, "name", ""),
                    description=getattr(tool, "description", None),
                    parameters=getattr(tool, "parameters", None),
                )
            )
            continue
        if isinstance(tool, Mapping):
            function = tool.get("function") or {}
            if function:
                definitions.append(
                    FunctionToolDefinition(
                        name=str(function.get("name") or ""),
                        description=function.get("description"),
                        parameters=function.get("parameters"),
                    )
                )
            else:
                definitions.append(
                    GenericToolDefinition(
                        name=str(tool.get("name") or tool.get("type") or ""),
                        type=str(tool.get("type") or "tool"),
                    )
                )
    return definitions


def create_agent_invocation(
    agent: Any, arguments: Mapping[str, Any]
) -> InvokeAgentInvocation:
    model = getattr(agent, "model", None)
    input_value = arguments.get("input")
    attributes: dict[str, Any] = agno_run_identity_attributes(agent, arguments)
    session_id = attributes.get(GEN_AI_SESSION_ID)

    invocation = InvokeAgentInvocation(
        provider=_PROVIDER,
        agent_name=getattr(agent, "name", None),
        agent_id=getattr(agent, "id", None)
        or getattr(agent, "agent_id", None),
        agent_description=getattr(agent, "description", None),
        conversation_id=session_id,
        request_model=getattr(model, "id", None),
        input_messages=convert_agent_input(input_value),
        tool_definitions=convert_tool_definitions(
            getattr(agent, "tools", None)
        ),
        system_instruction=_system_instruction_parts(agent),
        attributes=attributes,
    )

    if model is not None:
        invocation.provider = getattr(model, "provider", None) or _PROVIDER
        for name in (
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
            "seed",
        ):
            value = getattr(model, name, None)
            if value is not None:
                setattr(invocation, name, value)
        stop = getattr(model, "stop", None)
        if isinstance(stop, str):
            invocation.stop_sequences = [stop]
        elif isinstance(stop, list):
            invocation.stop_sequences = stop

    return invocation


def _usage_value(metrics: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(metrics, name, None)
        if value is not None:
            return value
    return None


def update_agent_invocation_from_response(
    invocation: InvokeAgentInvocation, response: Any
) -> None:
    if response is None:
        return

    invocation.response_id = getattr(response, "run_id", None)
    invocation.response_model_name = getattr(response, "model", None)
    invocation.provider = getattr(response, "model_provider", None) or (
        invocation.provider
    )
    invocation.output_type = getattr(response, "content_type", None)

    if (
        getattr(response, "content", None) is not None
        or getattr(response, "reasoning_content", None) is not None
        or getattr(response, "tool_calls", None)
    ):
        invocation.output_messages = [_message_to_output(response)]
        finish_reason = _finish_reason(response)
        if finish_reason:
            invocation.finish_reasons = [finish_reason]

    metrics = getattr(response, "metrics", None)
    if metrics is not None:
        invocation.input_tokens = _usage_value(metrics, "input_tokens")
        invocation.output_tokens = _usage_value(metrics, "output_tokens")
        invocation.usage_cache_read_input_tokens = _usage_value(
            metrics, "cache_read_tokens"
        )
        invocation.usage_cache_creation_input_tokens = _usage_value(
            metrics, "cache_write_tokens"
        )


def update_agent_invocation_from_events(
    invocation: InvokeAgentInvocation, events: Sequence[Any]
) -> None:
    if not events:
        return
    content = []
    reasoning = []
    completed = None
    for event in events:
        if getattr(event, "run_id", None):
            invocation.response_id = getattr(event, "run_id")
        if getattr(event, "agent_id", None):
            invocation.agent_id = invocation.agent_id or getattr(
                event, "agent_id"
            )
        if getattr(event, "session_id", None):
            invocation.conversation_id = invocation.conversation_id or getattr(
                event, "session_id"
            )
        event_content = getattr(event, "content", None)
        if event_content:
            content.append(_stringify(event_content))
        event_reasoning = getattr(event, "reasoning_content", None)
        if event_reasoning:
            reasoning.append(_stringify(event_reasoning))
        if getattr(event, "event", None) == "RunCompleted":
            completed = event

    if completed is not None:
        update_agent_invocation_from_response(invocation, completed)
    elif content or reasoning:
        parts = _text_parts("".join(content))
        if reasoning:
            parts.append(Reasoning(content="".join(reasoning)))
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=parts,
                finish_reason=_finish_reason(completed or events[-1]),
            )
        ]
        finish_reason = _finish_reason(completed or events[-1])
        if finish_reason:
            invocation.finish_reasons = [finish_reason]


def create_llm_invocation(
    model: Any, arguments: Mapping[str, Any]
) -> LLMInvocation:
    request_model = getattr(model, "id", None) or getattr(model, "name", None)
    invocation = LLMInvocation(
        request_model=request_model,
        provider=getattr(model, "provider", None) or _PROVIDER,
        input_messages=convert_model_messages(arguments.get("messages")),
        tool_definitions=convert_tool_definitions(arguments.get("tools")),
    )
    session_id = _apply_current_identity(invocation)
    if session_id and not getattr(invocation, "conversation_id", None):
        invocation.conversation_id = session_id

    for name in (
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "seed",
    ):
        value = getattr(model, name, None)
        if value is not None:
            setattr(invocation, name, value)

    stop = getattr(model, "stop", None)
    if isinstance(stop, str):
        invocation.stop_sequences = [stop]
    elif isinstance(stop, list):
        invocation.stop_sequences = stop

    response_format = arguments.get("response_format")
    if response_format is not None:
        invocation.output_type = (
            response_format.__name__
            if hasattr(response_format, "__name__")
            else _stringify(response_format)
        )
    return invocation


def update_llm_invocation_from_response(
    invocation: LLMInvocation, response: Any
) -> None:
    if response is None:
        return

    invocation.response_id = getattr(response, "id", None)
    invocation.response_model_name = (
        getattr(response, "model", None) or invocation.request_model
    )
    invocation.provider = (
        getattr(response, "model_provider", None)
        or getattr(response, "provider", None)
        or invocation.provider
    )
    invocation.output_messages = [_message_to_output(response)]
    finish_reason = _finish_reason(response)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]
    invocation.input_tokens = _usage_value(
        response, "input_tokens", "prompt_tokens"
    )
    invocation.output_tokens = _usage_value(
        response, "output_tokens", "completion_tokens"
    )
    invocation.usage_cache_read_input_tokens = _usage_value(
        response, "cache_read_tokens"
    )
    invocation.usage_cache_creation_input_tokens = _usage_value(
        response, "cache_write_tokens"
    )

    usage = getattr(response, "response_usage", None)
    if usage is not None:
        invocation.input_tokens = invocation.input_tokens or _usage_value(
            usage, "input_tokens", "prompt_tokens"
        )
        invocation.output_tokens = invocation.output_tokens or _usage_value(
            usage, "output_tokens", "completion_tokens"
        )
        invocation.usage_cache_read_input_tokens = (
            invocation.usage_cache_read_input_tokens
            or _usage_value(usage, "cache_read_tokens")
        )
        invocation.usage_cache_creation_input_tokens = (
            invocation.usage_cache_creation_input_tokens
            or _usage_value(usage, "cache_write_tokens")
        )


def create_tool_invocation(function_call: Any) -> ExecuteToolInvocation:
    function = getattr(function_call, "function", None)
    invocation = ExecuteToolInvocation(
        tool_name=getattr(function, "name", None) or "unknown_tool",
        provider=_PROVIDER,
        tool_call_id=getattr(function_call, "call_id", None),
        tool_description=getattr(function, "description", None),
        tool_type="function",
        tool_call_arguments=getattr(function_call, "arguments", None),
    )
    _apply_current_identity(invocation)
    _apply_agno_skill_tool_metadata(invocation, invocation.tool_call_arguments)
    return invocation


def update_tool_invocation_from_response(
    invocation: ExecuteToolInvocation, response: Any
) -> None:
    result = getattr(response, "result", response)
    invocation.tool_call_result = (
        result if isinstance(result, str) else _stringify(result)
    )
    _apply_agno_skill_tool_metadata(invocation, result)
