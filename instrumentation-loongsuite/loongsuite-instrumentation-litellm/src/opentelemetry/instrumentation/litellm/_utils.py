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
Utility functions for LiteLLM instrumentation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
)
from opentelemetry.util.genai.extended_types import EmbeddingInvocation
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)


def convert_messages_to_structured_format(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert LiteLLM message format to structured format required by semantic conventions.

    Converts from:
        {"role": "user", "content": "..."}
    To:
        {"role": "user", "parts": [{"type": "text", "content": "..."}]}
    """
    if not isinstance(messages, list):
        return []

    structured_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        structured_msg = {"role": role, "parts": []}

        # Handle text content
        if "content" in msg and msg["content"]:
            content = msg["content"]
            if isinstance(content, str):
                structured_msg["parts"].append(
                    {"type": "text", "content": content}
                )
            elif isinstance(content, list):
                # Handle multi-modal content
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            structured_msg["parts"].append(
                                {
                                    "type": "text",
                                    "content": item.get("text", ""),
                                }
                            )
                        else:
                            structured_msg["parts"].append(item)

        # Handle tool calls
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue

                tool_part = {"type": "tool_call"}
                if "id" in tool_call:
                    tool_part["id"] = tool_call["id"]
                if "function" in tool_call:
                    func = tool_call["function"]
                    if isinstance(func, dict):
                        if "name" in func:
                            tool_part["name"] = func["name"]
                        if "arguments" in func:
                            try:
                                # Try to parse arguments if it's a JSON string
                                args_str = func["arguments"]
                                if isinstance(args_str, str):
                                    tool_part["arguments"] = json.loads(
                                        args_str
                                    )
                                else:
                                    tool_part["arguments"] = args_str
                            except Exception:
                                tool_part["arguments"] = func.get(
                                    "arguments", ""
                                )

                structured_msg["parts"].append(tool_part)

        # Handle tool call responses
        if role == "tool" and "content" in msg:
            tool_response_part = {
                "type": "tool_call_response",
                "response": msg["content"],
            }
            if "tool_call_id" in msg:
                tool_response_part["id"] = msg["tool_call_id"]
            structured_msg["parts"].append(tool_response_part)

        structured_messages.append(structured_msg)

    return structured_messages


def parse_provider_from_model(model: str) -> Optional[str]:
    """
    Parse provider name from model string.

    LiteLLM uses format like "openai/gpt-4", "dashscope/qwen-turbo", etc.
    """
    if not model:
        return None

    if "/" in model:
        return model.split("/")[0]

    # Fallback: try to infer from model name patterns
    if "gpt" in model.lower():
        return "openai"
    elif "qwen" in model.lower():
        return "dashscope"
    elif "claude" in model.lower():
        return "anthropic"
    elif "gemini" in model.lower():
        return "google"

    return "unknown"


def parse_model_name(model: str) -> str:
    """
    Parse model name by removing provider prefix.

    Examples:
        "openai/gpt-4" -> "gpt-4"
        "dashscope/qwen-turbo" -> "qwen-turbo"
        "gpt-4" -> "gpt-4"
    """
    if not model:
        return "unknown"

    if "/" in model:
        return model.split("/", 1)[1]

    return model


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON string.
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Failed to serialize object to JSON: {e}")
        return default


def convert_tool_definitions(tools: List[Dict[str, Any]]) -> str:
    """
    Convert tool definitions to JSON string format.
    """
    if not tools:
        return "[]"

    try:
        # Tools are typically in format: [{"type": "function", "function": {...}}]
        return json.dumps(tools, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Failed to convert tool definitions: {e}")
        return "[]"


def convert_litellm_messages_to_genai_format(
    messages: List[Dict[str, Any]],
) -> List:
    """
    Convert LiteLLM message format to OpenTelemetry GenAI InputMessage format.

    This function converts LiteLLM's message structure to the standardized
    InputMessage format required by ExtendedTelemetryHandler.
    """

    if not isinstance(messages, list):
        return []

    input_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "user")
        parts = []

        # Handle text content
        if "content" in msg and msg["content"]:
            content = msg["content"]
            if isinstance(content, str):
                parts.append(Text(content=content))
            elif isinstance(content, list):
                # Handle multi-modal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(Text(content=item.get("text", "")))
                    # Other content types (image, etc.) can be added here

        # Handle tool calls
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue

                func = tool_call.get("function", {})
                if isinstance(func, dict):
                    # Parse arguments if it's a JSON string
                    arguments = func.get("arguments", "")
                    if isinstance(arguments, str) and arguments:
                        try:
                            arguments = json.loads(arguments)
                        except Exception:
                            # If arguments are not valid JSON, keep the original string
                            pass

                    parts.append(
                        ToolCall(
                            id=tool_call.get("id"),
                            name=func.get("name", ""),
                            arguments=arguments,
                        )
                    )

        # Handle tool call responses
        if role == "tool" and "content" in msg:
            parts.append(
                ToolCallResponse(
                    id=msg.get("tool_call_id"), response=msg["content"]
                )
            )

        # If no parts added, add empty text
        if not parts:
            parts.append(Text(content=""))

        input_messages.append(InputMessage(role=role, parts=parts))

    return input_messages


def extract_output_from_litellm_response(response: Any) -> List:
    """
    Extract output messages from LiteLLM response.

    Converts LiteLLM response to OpenTelemetry GenAI OutputMessage format.
    """

    if not hasattr(response, "choices") or not response.choices:
        return []

    output_messages = []
    for choice in response.choices:
        if not hasattr(choice, "message"):
            continue

        msg = choice.message
        parts = []

        # Extract text content
        if hasattr(msg, "content") and msg.content:
            parts.append(Text(content=msg.content))

        # Extract tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                # Parse arguments if it's a JSON string
                arguments = getattr(tc.function, "arguments", "")
                if isinstance(arguments, str) and arguments:
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        # If arguments are not valid JSON, keep the original string
                        pass

                parts.append(
                    ToolCall(
                        id=getattr(tc, "id", None),
                        name=getattr(tc.function, "name", ""),
                        arguments=arguments,
                    )
                )

        # If no parts, add empty text
        if not parts:
            parts.append(Text(content=""))

        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        output_messages.append(
            OutputMessage(
                role=getattr(msg, "role", "assistant"),
                parts=parts,
                finish_reason=finish_reason,
            )
        )

    return output_messages


def create_llm_invocation_from_litellm(**kwargs):
    """
    Create LLMInvocation from LiteLLM request parameters.

    Args:
        model: The model name (e.g., "gpt-4", "openai/gpt-4")
        provider: The provider name (e.g., "openai", "dashscope")
        messages: List of message dictionaries
        **kwargs: Additional request parameters (temperature, max_tokens, etc.)

    Returns:
        LLMInvocation object ready for use with ExtendedTelemetryHandler
    """

    # Parse model name (remove provider prefix if present)
    model = kwargs.get("model", "unknown_model")
    provider = parse_provider_from_model(model) or "unknown"
    messages = kwargs.get("messages", [])

    # Convert messages to GenAI format
    input_messages = convert_litellm_messages_to_genai_format(messages)

    request_model = parse_model_name(model)

    invocation = LLMInvocation(
        request_model=request_model,
        provider=provider or "unknown",
        operation_name=GenAiOperationNameValues.CHAT.value,
        input_messages=input_messages,
    )

    # Set optional request parameters
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        invocation.temperature = kwargs["temperature"]
    if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
        invocation.max_tokens = kwargs["max_tokens"]
    if "top_p" in kwargs and kwargs["top_p"] is not None:
        invocation.top_p = kwargs["top_p"]
    if (
        "frequency_penalty" in kwargs
        and kwargs["frequency_penalty"] is not None
    ):
        invocation.frequency_penalty = kwargs["frequency_penalty"]
    if "presence_penalty" in kwargs and kwargs["presence_penalty"] is not None:
        invocation.presence_penalty = kwargs["presence_penalty"]
    if "seed" in kwargs and kwargs["seed"] is not None:
        invocation.seed = kwargs["seed"]
    if "stop" in kwargs and kwargs["stop"] is not None:
        stop = kwargs["stop"]
        if isinstance(stop, str):
            invocation.stop_sequences = [stop]
        elif isinstance(stop, list):
            invocation.stop_sequences = stop

    # Handle tool definitions
    if "tools" in kwargs and kwargs["tools"]:
        tools = kwargs["tools"]
        tool_definitions = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                func = tool["function"]
                tool_definitions.append(
                    FunctionToolDefinition(
                        name=func.get("name", ""),
                        description=func.get("description"),
                        parameters=func.get("parameters"),
                    )
                )
        if tool_definitions:
            invocation.tool_definitions = tool_definitions

    return invocation


def create_embedding_invocation_from_litellm(**kwargs):
    """
    Create EmbeddingInvocation from LiteLLM embedding request parameters.

    Args:
        model: The embedding model name
        provider: The provider name
        **kwargs: Additional request parameters

    Returns:
        EmbeddingInvocation object ready for use with ExtendedTelemetryHandler
    """

    # Extract request parameters
    model = kwargs.get("model", "unknown")
    provider = parse_provider_from_model(model) or "unknown"

    # Parse model name (remove provider prefix if present)
    request_model = parse_model_name(model)

    invocation = EmbeddingInvocation(
        request_model=request_model,
        provider=provider or "unknown",
    )

    # Set encoding formats if present
    if "encoding_format" in kwargs and kwargs["encoding_format"]:
        invocation.encoding_formats = [kwargs["encoding_format"]]

    return invocation
