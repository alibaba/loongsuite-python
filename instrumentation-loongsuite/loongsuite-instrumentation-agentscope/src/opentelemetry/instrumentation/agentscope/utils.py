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

# -*- coding: utf-8 -*-
"""Attributes processor for span attributes."""

from __future__ import annotations

import datetime
import enum
import inspect
import json
import logging
import mimetypes
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agentscope import _config
from agentscope.agent import AgentBase
from agentscope.embedding import EmbeddingModelBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase, ChatResponse
from pydantic import BaseModel

from opentelemetry import baggage
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)
from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import (
    Base64Blob,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
    Uri,
)

from .message_converter import get_message_converter

logger = logging.getLogger(__name__)


class AgentScopeGenAiProviderName(str, Enum):
    """Extended provider names not in standard OpenTelemetry semantic conventions."""

    OLLAMA = "ollama"
    DASHSCOPE = "dashscope"
    MOONSHOT = "moonshot"


# Provider name mapping based on class names
_PROVIDER_NAME_MAP = {
    "openai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "gemini": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "anthropic": GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value,
    "dashscope": AgentScopeGenAiProviderName.DASHSCOPE.value,
    "ollama": AgentScopeGenAiProviderName.OLLAMA.value,
}

# Base URL to provider mapping for OpenAI-compatible APIs
_BASE_URL_PROVIDER_MAP = [
    ("openai.com", GenAIAttributes.GenAiProviderNameValues.OPENAI.value),
    (
        "api.deepseek.com",
        GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value,
    ),
    ("dashscope.aliyuncs.com", AgentScopeGenAiProviderName.DASHSCOPE.value),
]


def _current_baggage_value(key: str) -> str | None:
    try:
        value = baggage.get_baggage(key)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def entry_baggage_identity_attributes() -> dict[str, str]:
    """Return entry-level identity from current OpenTelemetry Baggage.

    QwenPaw opens an Entry span before AgentScope runs and writes
    ``gen_ai.session.id`` / ``gen_ai.user.id`` into Baggage. AgentScope has
    its own ``run_id``; when both instrumentations are active, entry baggage is
    the request-level identity and should color downstream spans.
    """
    attributes: dict[str, str] = {}
    session_id = _current_baggage_value(GEN_AI_SESSION_ID)
    user_id = _current_baggage_value(GEN_AI_USER_ID)
    if session_id:
        attributes[GEN_AI_SESSION_ID] = session_id
    if user_id:
        attributes[GEN_AI_USER_ID] = user_id
    return attributes


def apply_entry_baggage_identity(invocation: Any) -> str | None:
    """Copy entry-level identity baggage onto a GenAI invocation."""
    attributes = entry_baggage_identity_attributes()
    for key, value in attributes.items():
        invocation.attributes.setdefault(key, value)
    return attributes.get(GEN_AI_SESSION_ID)


def get_provider_name(chat_model: ChatModelBase) -> str:
    """Parse chat model provider name"""
    classname = chat_model.__class__.__name__
    prefix = classname.removesuffix("ChatModel").lower()

    # Special handling for DashScopeChatModel with base_http_api_url
    if (
        prefix == "dashscope"
        and hasattr(chat_model, "base_http_api_url")
        and chat_model.base_http_api_url
    ):
        base_url = chat_model.base_http_api_url
        for url_fragment, provider in _BASE_URL_PROVIDER_MAP:
            if url_fragment in base_url:
                return provider

    return _PROVIDER_NAME_MAP.get(prefix, "unknown")


def get_embedding_provider_name(embedding_model: EmbeddingModelBase) -> str:
    """Parse embedding model provider name"""
    class_name = embedding_model.__class__.__name__

    if "DashScope" in class_name:
        return AgentScopeGenAiProviderName.DASHSCOPE.value
    elif "OpenAI" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    elif "Gemini" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
    elif "Ollama" in class_name:
        return AgentScopeGenAiProviderName.OLLAMA.value
    else:
        model_name = getattr(embedding_model, "model_name", "")
        if "text-embedding" in model_name:
            if "qwen" in model_name or "dashscope" in model_name:
                return AgentScopeGenAiProviderName.DASHSCOPE.value
            else:
                return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
        return "agentscope"


def _get_tool_definitions(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
) -> str | None:
    """Extract and serialize tool definitions for tracing."""
    if tools is None or not isinstance(tools, list) or len(tools) == 0:
        return None

    if tool_choice == "none":
        return None

    try:
        flat_tools = []
        for tool in tools:
            if not isinstance(tool, dict) or "function" not in tool:
                continue

            func_def = tool["function"]
            flat_tool = {
                "type": tool.get("type", "function"),
                "name": func_def.get("name"),
                "description": func_def.get("description"),
                "parameters": func_def.get("parameters"),
            }
            flat_tool = {k: v for k, v in flat_tool.items() if v is not None}
            flat_tools.append(flat_tool)

        if flat_tools:
            return _serialize_to_str(flat_tools)
        return None

    except Exception:
        return None


def _to_serializable(obj: Any) -> Any:
    """Convert an object to a JSON serializable type."""
    if isinstance(obj, (str, int, bool, float, type(None))):
        return obj
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(key): _to_serializable(val) for (key, val) in obj.items()}
    elif isinstance(obj, (Msg, BaseModel)) or is_dataclass(obj):
        return repr(obj)
    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        return repr(obj)
    elif isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        return obj.isoformat()
    elif isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    elif isinstance(obj, enum.Enum):
        return _to_serializable(obj.value)
    else:
        return str(obj)


def _serialize_to_str(value: Any) -> str:
    """Serialize value to JSON string"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(_to_serializable(value), ensure_ascii=False)


def _convert_block_to_part(block: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert content block to standardized part format."""
    block_type = block.get("type")

    if block_type == "text":
        return {"type": "text", "content": block.get("text", "")}

    elif block_type == "thinking":
        return {
            "type": "text",
            "content": f"[Thinking] {block.get('thinking', '')}",
        }

    elif block_type == "tool_use":
        return {
            "type": "tool_call",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "arguments": block.get("input", {}),
        }

    elif block_type == "tool_result":
        output = block.get("output", "")
        result = (
            _serialize_to_str(output)
            if isinstance(output, (list, dict))
            else str(output)
        )
        return {
            "type": "tool_call_response",
            "id": block.get("id", ""),
            "result": result,
        }

    elif block_type in ["image", "audio", "video"]:
        source = block.get("source", {})
        source_type = source.get("type")
        media_type = source.get("media_type")

        if source_type == "url":
            url = source.get("url", "")
            # Infer mime_type from URL extension if not provided
            if not media_type:
                parsed_url = urlparse(url)
                media_type, _ = mimetypes.guess_type(parsed_url.path)
            return {
                "type": "uri",
                "uri": url,
                "modality": block_type,
                "mime_type": media_type,
            }
        elif source_type == "base64":
            if not media_type:
                default_media_types = {
                    "image": "image/jpeg",
                    "audio": "audio/wav",
                    "video": "video/mp4",
                }
                media_type = default_media_types.get(block_type, "unknown")
            return {
                "type": "blob",
                "content": source.get("data", ""),
                "media_type": media_type,
                "modality": block_type,
            }

    return None


def _format_msg_to_parts(msg: Msg) -> dict[str, Any]:
    """Convert Msg to standard format (parts structure)"""
    try:
        parts = []
        for block in msg.get_content_blocks():
            part = _convert_block_to_part(block)
            if part:
                parts.append(part)

        formatted_msg = {"role": msg.role, "parts": parts}
        if msg.name:
            formatted_msg["name"] = msg.name

        return formatted_msg

    except Exception as e:
        logger.debug(f"Error formatting message: {e}")
        return {
            "role": msg.role,
            "parts": [
                {
                    "type": "text",
                    "content": str(msg.content) if msg.content else "",
                }
            ],
        }


def create_llm_invocation(
    call_instance: ChatModelBase,
    call_args: Tuple[Any, ...],
    call_kwargs: Dict[str, Any],
) -> LLMInvocation:
    """Create LLM invocation from call context."""
    provider_name = get_provider_name(call_instance)
    request_model = getattr(call_instance, "model_name", "unknown_model")

    # Convert input messages
    input_messages = []
    messages = call_args[0] if call_args else call_kwargs.get("messages")
    if messages:
        try:
            converted_messages = get_message_converter(provider_name)(messages)
            input_messages = convert_agentscope_messages_to_genai_format(
                converted_messages, provider_name=provider_name
            )
        except Exception as e:
            logger.warning(
                f"Failed to convert input messages: {e}", exc_info=True
            )

    invocation = LLMInvocation(
        request_model=request_model,
        provider=provider_name,
        input_messages=input_messages,
    )
    entry_session_id = apply_entry_baggage_identity(invocation)
    if entry_session_id and invocation.conversation_id is None:
        invocation.conversation_id = entry_session_id

    # Set optional request parameters if present
    if call_kwargs.get("max_tokens"):
        invocation.max_tokens = call_kwargs["max_tokens"]
    if call_kwargs.get("temperature"):
        invocation.temperature = call_kwargs["temperature"]
    if call_kwargs.get("top_p"):
        invocation.top_p = call_kwargs["top_p"]

    # Set tool definitions if applicable
    if not call_kwargs.get("structured_model", False):
        tool_definitions = _get_tool_definitions(
            call_kwargs.get("tools"), call_kwargs.get("tool_choice")
        )
        if tool_definitions:
            invocation.attributes["gen_ai.request.tool_definitions"] = (
                tool_definitions
            )

    return invocation


def create_embedding_invocation(
    call_instance: EmbeddingModelBase,
    call_args: Tuple[Any, ...],
    call_kwargs: Dict[str, Any],
) -> EmbeddingInvocation:
    """Create Embedding invocation from call context."""
    provider_name = get_embedding_provider_name(call_instance)
    request_model = getattr(call_instance, "model_name", "unknown_model")

    invocation = EmbeddingInvocation(
        request_model=request_model,
        provider=provider_name,
    )
    apply_entry_baggage_identity(invocation)

    # Set encoding formats if present
    if call_kwargs.get("encoding_formats"):
        invocation.encoding_formats = call_kwargs["encoding_formats"]

    return invocation


def create_agent_invocation(
    reply_instance: AgentBase,
    reply_args: Tuple[Any, ...],
    reply_kwargs: Dict[str, Any],
) -> InvokeAgentInvocation:
    """Create Agent invocation from reply call context."""
    # Get provider and model info
    provider_name = None
    request_model = None
    if hasattr(reply_instance, "model") and reply_instance.model:
        model = reply_instance.model
        request_model = getattr(model, "model_name", None)
        if isinstance(model, ChatModelBase):
            provider_name = get_provider_name(model)

    # Convert input messages
    input_messages = []
    msg = reply_args[0] if reply_args else reply_kwargs.get("msg")
    if msg:
        try:
            if isinstance(msg, Msg):
                input_messages = convert_agentscope_messages_to_genai_format(
                    [msg]
                )
            elif isinstance(msg, list):
                input_messages = convert_agentscope_messages_to_genai_format(
                    msg
                )
        except Exception as e:
            logger.debug(f"Error converting agent input messages: {e}")

    entry_session_id = _current_baggage_value(GEN_AI_SESSION_ID)
    invocation = InvokeAgentInvocation(
        provider=provider_name,
        agent_name=getattr(reply_instance, "name", "unknown_agent"),
        agent_id=getattr(reply_instance, "id", "unknown"),
        agent_description=inspect.getdoc(reply_instance.__class__)
        or "No description available",
        conversation_id=entry_session_id or _config.run_id,
        request_model=request_model,
        input_messages=input_messages,
    )
    apply_entry_baggage_identity(invocation)

    # Set system instruction if available
    if hasattr(reply_instance, "sys_prompt") and reply_instance.sys_prompt:
        sys_prompt = reply_instance.sys_prompt
        if isinstance(sys_prompt, str):
            invocation.system_instruction = [Text(content=sys_prompt)]
        elif isinstance(sys_prompt, list):
            invocation.system_instruction = sys_prompt

    return invocation


def get_chatmodel_output_messages(chat_response: Any) -> List[Dict[str, Any]]:
    """Convert ChatResponse to OpenTelemetry standard output message format."""
    try:
        if not isinstance(chat_response, ChatResponse):
            return []

        parts = []
        for block in chat_response.content:
            part = _convert_block_to_part(block)
            if part:
                parts.append(part)

        if not parts:
            parts.append({"type": "text", "content": ""})

        return [
            {
                "role": "assistant",
                "parts": parts,
                "finish_reason": "stop",
            }
        ]

    except Exception as e:
        logger.warning(
            f"Error processing ChatResponse to output messages: {e}"
        )
        return [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "<error processing response>"}
                ],
                "finish_reason": "error",
            }
        ]


def convert_agentscope_messages_to_genai_format(
    messages: Any,
    provider_name: Optional[str] = None,
) -> List[InputMessage]:
    """Convert AgentScope messages to opentelemetry-util-genai InputMessage format.
       parse _convert_block_to_part result from agentscope/tracing/_converter.py

    This function is used by ExtendedTelemetryHandler which requires InputMessage objects.
    """
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    input_messages = []
    for msg in messages:
        if isinstance(msg, Msg):
            msg_dict = _format_msg_to_parts(msg)
        elif isinstance(msg, dict):
            msg_dict = msg
        else:
            continue

        role = msg_dict.get("role", "user")
        parts = msg_dict.get("parts", [])

        converted_parts = []
        for part in parts:
            part_type = part.get("type")
            if part_type == "text":
                converted_parts.append(Text(content=part.get("content", "")))
            elif part_type == "tool_call":
                converted_parts.append(
                    ToolCall(
                        id=part.get("id"),
                        name=part.get("name", ""),
                        arguments=part.get("arguments", {}),
                    )
                )
            elif part_type == "tool_call_response":
                converted_parts.append(
                    ToolCallResponse(
                        id=part.get("id"),
                        response=part["response"]
                        if "response" in part
                        else part.get("result", ""),
                    )
                )
            elif part_type == "reasoning":
                converted_parts.append(
                    Reasoning(content=part.get("content", ""))
                )
            elif part_type == "uri":
                converted_parts.append(
                    Uri(
                        uri=part.get("uri"),
                        modality=part.get("modality"),
                        mime_type=part.get("mime_type"),
                    )
                )
            elif part_type == "blob":
                converted_parts.append(
                    Base64Blob(
                        content=part.get("content", ""),
                        modality=part.get("modality"),
                        mime_type=part.get("media_type")
                        if "media_type" in part
                        else part.get("mime_type"),
                    )
                )
            else:
                # Keep other types as-is
                converted_parts.append(part)

        input_messages.append(InputMessage(role=role, parts=converted_parts))

    return input_messages


def convert_chatresponse_to_output_messages(
    chat_response: Any,
) -> List[OutputMessage]:
    """Convert ChatResponse to opentelemetry-util-genai OutputMessage format.

    This function is used by ExtendedTelemetryHandler which requires OutputMessage objects.
    """
    output_dicts = get_chatmodel_output_messages(chat_response)
    if not output_dicts:
        return []

    output_messages = []
    for msg_dict in output_dicts:
        role = msg_dict.get("role", "assistant")
        parts_dicts = msg_dict.get("parts", [])
        finish_reason = msg_dict.get("finish_reason", "stop")

        converted_parts = []
        for part in parts_dicts:
            part_type = part.get("type")
            if part_type == "text":
                converted_parts.append(Text(content=part.get("content", "")))
            elif part_type == "tool_call":
                converted_parts.append(
                    ToolCall(
                        id=part.get("id"),
                        name=part.get("name", ""),
                        arguments=part.get("arguments", {}),
                    )
                )
            else:
                converted_parts.append(part)

        output_messages.append(
            OutputMessage(
                role=role,
                parts=converted_parts,
                finish_reason=finish_reason,
            )
        )

    return output_messages


def convert_agent_response_to_output_messages(
    agent_response: Any,
) -> List[OutputMessage]:
    """Convert agent Msg response to opentelemetry-util-genai OutputMessage format.

    This function is used by ExtendedTelemetryHandler which requires OutputMessage objects.
    """
    if not hasattr(agent_response, "content"):
        return []

    try:
        # Use _format_msg_to_parts to convert Msg to dict, then convert to OutputMessage
        msg_dict = _format_msg_to_parts(agent_response)
        parts_dicts = msg_dict.get("parts", [])

        converted_parts = []
        for part in parts_dicts:
            part_type = part.get("type")
            if part_type == "text":
                converted_parts.append(Text(content=part.get("content", "")))
            elif part_type == "reasoning":
                converted_parts.append(
                    Reasoning(content=part.get("content", ""))
                )
            elif part_type == "tool_call":
                converted_parts.append(
                    ToolCall(
                        id=part.get("id"),
                        name=part.get("name", ""),
                        arguments=part.get("arguments", {}),
                    )
                )
            elif part_type in ("uri", "blob"):
                converted_parts.append(part)
            else:
                converted_parts.append(Text(content=str(part)))

        if not converted_parts:
            converted_parts.append(Text(content=""))

        return [
            OutputMessage(
                role="assistant",
                parts=converted_parts,
                finish_reason="stop",
            )
        ]
    except Exception as e:
        logger.debug(f"Failed to convert agent response: {e}")
        return []
