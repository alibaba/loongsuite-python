# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Utility helpers for deepagents telemetry."""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

try:
    from langchain_core.runnables.config import ensure_config as _ensure_config
except ModuleNotFoundError:
    _ensure_config = None  # type: ignore[assignment]

from ._attributes import (
    ENTRY_PARENT_KINDS,
    FRAMEWORK_NAME,
    GEN_AI_AGENT_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_FRAMEWORK_VERSION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_SPAN_KIND,
    GRAPH_METADATA_ATTR,
    LANGGRAPH_REACT_AGENT_METADATA_KEY,
    METADATA_DEEPAGENTS_VERSION,
    METADATA_LC_AGENT_NAME,
    METADATA_LS_AGENT_TYPE,
    METADATA_LS_INTEGRATION,
    METADATA_SUBAGENT_DESCRIPTION,
    METADATA_VERSIONS,
    SPAN_KIND_ENTRY,
)

_logger = logging.getLogger(__name__)
_CURRENT_SUBAGENT_REGISTRY: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar(
        "opentelemetry_deepagents_subagent_registry",
        default=None,
    )
)


def obj_get(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def set_current_subagent_registry(registry: dict[str, str] | None):
    return _CURRENT_SUBAGENT_REGISTRY.set(registry or None)


def reset_current_subagent_registry(token: Any) -> None:
    _CURRENT_SUBAGENT_REGISTRY.reset(token)


def current_subagent_registry() -> dict[str, str]:
    return _CURRENT_SUBAGENT_REGISTRY.get() or {}


def detect_deepagents_version(metadata: Mapping[str, Any] | None = None) -> str:
    versions = obj_get(metadata or {}, METADATA_VERSIONS, {})
    version = obj_get(versions, METADATA_DEEPAGENTS_VERSION)
    if version:
        return str(version)
    try:
        return str(getattr(import_module("deepagents"), "__version__"))
    except Exception:  # noqa: BLE001
        return ""


def graph_metadata(graph: Any, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    existing = getattr(graph, GRAPH_METADATA_ATTR, None)
    if isinstance(existing, Mapping):
        return dict(existing)

    config = getattr(graph, "config", None)
    metadata = obj_get(config or {}, "metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)

    return dict(fallback or {})


def root_agent_name(graph: Any, metadata: Mapping[str, Any] | None = None) -> str | None:
    name = obj_get(metadata or {}, METADATA_LC_AGENT_NAME)
    if name:
        return str(name)
    config = getattr(graph, "config", None)
    metadata_from_config = obj_get(config or {}, "metadata", {})
    name = obj_get(metadata_from_config, METADATA_LC_AGENT_NAME)
    if name:
        return str(name)
    name = getattr(graph, "name", None)
    return str(name) if name else None


def create_graph_metadata(
    graph: Any,
    *,
    name: Any = None,
) -> dict[str, Any]:
    metadata = graph_metadata(graph)
    metadata.setdefault(METADATA_LS_INTEGRATION, FRAMEWORK_NAME)
    versions = dict(obj_get(metadata, METADATA_VERSIONS, {}) or {})
    version = detect_deepagents_version(metadata)
    if version:
        versions.setdefault(METADATA_DEEPAGENTS_VERSION, version)
    if versions:
        metadata[METADATA_VERSIONS] = versions
    if name is not None:
        metadata.setdefault(METADATA_LC_AGENT_NAME, str(name))
    return metadata


def extract_subagent_registry(
    subagents: Any,
) -> dict[str, str]:
    registry: dict[str, str] = {}
    for spec in subagents or ():
        name = obj_get(spec, "name")
        description = obj_get(spec, "description")
        if name and description:
            registry[str(name)] = str(description)
    return registry


def active_span_is_entry_or_agent() -> bool:
    current_span = trace.get_current_span()
    attributes = getattr(current_span, "attributes", None)
    if isinstance(attributes, Mapping):
        if attributes.get(GEN_AI_SPAN_KIND) in ENTRY_PARENT_KINDS:
            return True

    # util-genai sets the span kind on finish. The name check prevents nested
    # ENTRY creation while an unfinished util-genai ENTRY/AGENT span is active.
    span_name = str(getattr(current_span, "name", "") or "")
    return span_name.startswith(("enter_ai_application_system", "invoke_agent"))


def safe_set_attribute(span: Any, key: str, value: Any) -> None:
    if value is None:
        return
    setter = getattr(span, "set_attribute", None)
    if setter is None:
        return
    try:
        setter(key, value)
    except Exception:  # noqa: BLE001
        _logger.debug("Failed to set span attribute %s", key, exc_info=True)


def prime_entry_span(
    span: Span | None,
    *,
    method_name: str,
    metadata: Mapping[str, Any],
) -> None:
    if span is None:
        return
    version = detect_deepagents_version(metadata)
    safe_set_attribute(span, GEN_AI_SPAN_KIND, SPAN_KIND_ENTRY)
    safe_set_attribute(span, GEN_AI_OPERATION_NAME, method_name)
    safe_set_attribute(span, GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
    if version:
        safe_set_attribute(span, GEN_AI_FRAMEWORK_VERSION, version)
    safe_set_attribute(span, GEN_AI_AGENT_NAME, obj_get(metadata, METADATA_LC_AGENT_NAME))


def entry_attributes(
    *,
    method_name: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        GEN_AI_OPERATION_NAME: method_name,
        GEN_AI_FRAMEWORK: FRAMEWORK_NAME,
    }
    version = detect_deepagents_version(metadata)
    if version:
        attributes[GEN_AI_FRAMEWORK_VERSION] = version
    agent_name = obj_get(metadata, METADATA_LC_AGENT_NAME)
    if agent_name:
        attributes[GEN_AI_AGENT_NAME] = str(agent_name)
    return attributes


def config_from_call(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    if len(args) > 1:
        return args[1]
    return kwargs.get("config")


def _merge_deepagents_metadata(
    target: dict[str, Any],
    source: Mapping[str, Any] | None,
) -> None:
    if not source:
        return
    for key, value in source.items():
        if key == METADATA_VERSIONS and isinstance(value, Mapping):
            versions = dict(obj_get(target, METADATA_VERSIONS, {}) or {})
            for version_key, version_value in value.items():
                versions.setdefault(version_key, version_value)
            if versions:
                target[METADATA_VERSIONS] = versions
            continue
        if key == METADATA_SUBAGENT_DESCRIPTION and value:
            target[key] = value
            continue
        if key in {
            METADATA_LS_INTEGRATION,
            METADATA_LS_AGENT_TYPE,
            METADATA_LC_AGENT_NAME,
            METADATA_SUBAGENT_DESCRIPTION,
        }:
            target[key] = value
            continue
        target.setdefault(key, value)


def config_with_langgraph_react_metadata(
    config: Any,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    if _ensure_config is None:
        ensured = config or {}
    else:
        try:
            ensured = _ensure_config(config)
        except Exception:  # noqa: BLE001
            ensured = config or {}

    if not isinstance(ensured, Mapping):
        return ensured

    updated = dict(ensured)
    updated_metadata = dict(obj_get(updated, "metadata", {}) or {})
    _merge_deepagents_metadata(updated_metadata, metadata)
    updated_metadata.setdefault(LANGGRAPH_REACT_AGENT_METADATA_KEY, True)
    updated["metadata"] = updated_metadata
    return updated


def inject_langgraph_react_metadata(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    updated_kwargs = dict(kwargs)
    if len(args) > 1:
        config = config_with_langgraph_react_metadata(args[1], metadata)
        return (args[0], config) + args[2:], updated_kwargs

    updated_kwargs["config"] = config_with_langgraph_react_metadata(
        updated_kwargs.get("config"),
        metadata,
    )
    return args, updated_kwargs


def session_id_from_config(config: Any) -> str | None:
    configurable = obj_get(config or {}, "configurable", {})
    thread_id = obj_get(configurable or {}, "thread_id")
    return str(thread_id) if thread_id is not None else None


def input_value_from_call(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    if args:
        return args[0]
    if "input" in kwargs:
        return kwargs["input"]
    return None


def _flatten_content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        parts = []
        for item in value:
            text = obj_get(item, "text")
            content = obj_get(item, "content")
            if text:
                parts.append(str(text))
            elif content:
                parts.append(str(content))
        return "\n".join(parts).strip() or None
    return str(value)


def _tool_call_parts(message: Any) -> list[ToolCall]:
    parts: list[ToolCall] = []
    for tool_call in obj_get(message, "tool_calls", []) or []:
        function = obj_get(tool_call, "function", {})
        name = obj_get(tool_call, "name") or obj_get(function, "name")
        if not name:
            continue
        arguments = (
            obj_get(tool_call, "args")
            if obj_get(tool_call, "args") is not None
            else obj_get(function, "arguments", obj_get(tool_call, "arguments", {}))
        )
        parts.append(
            ToolCall(
                id=obj_get(tool_call, "id"),
                name=str(name),
                arguments=arguments,
            )
        )
    return parts


def _message_role(message: Any) -> str | None:
    role = obj_get(message, "role") or obj_get(message, "type")
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    if role == "function":
        return "tool"
    return str(role) if role else None


def _message_to_input(message: Any) -> InputMessage | None:
    role = _message_role(message)
    if not role:
        return None

    if role == "tool":
        return InputMessage(
            role="tool",
            parts=[
                ToolCallResponse(
                    id=obj_get(message, "tool_call_id"),
                    response=_flatten_content(obj_get(message, "content")),
                )
            ],
        )

    parts: list[Any] = []
    content = _flatten_content(obj_get(message, "content", message))
    if content:
        parts.append(Text(content=content))
    parts.extend(_tool_call_parts(message))
    return InputMessage(role=role, parts=parts) if parts else None


def _message_to_output(message: Any) -> OutputMessage | None:
    input_message = _message_to_input(message)
    if input_message is None:
        return None
    finish_reason = obj_get(message, "finish_reason") or obj_get(
        obj_get(message, "response_metadata", {}),
        "finish_reason",
    )
    return OutputMessage(
        role=input_message.role,
        parts=input_message.parts,
        finish_reason=str(finish_reason or "stop"),
    )


def _messages_from_state(value: Any) -> list[Any]:
    if isinstance(value, Mapping) and "messages" in value:
        messages = value.get("messages")
        if isinstance(messages, list):
            return messages
        return [messages]
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [{"role": "user", "content": _safe_json(value)}]


def input_messages_from_value(value: Any) -> list[InputMessage]:
    messages = []
    for message in _messages_from_state(value):
        converted = _message_to_input(message)
        if converted is not None:
            messages.append(converted)
    return messages


def output_messages_from_value(value: Any) -> list[OutputMessage]:
    messages = _messages_from_state(value)
    if messages:
        messages = [messages[-1]]
    converted_messages = []
    for message in messages:
        converted = _message_to_output(message)
        if converted is not None:
            converted_messages.append(converted)
    if converted_messages:
        return converted_messages
    if value is None:
        return []
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=_safe_json(value))],
            finish_reason="stop",
        )
    ]


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(value)
