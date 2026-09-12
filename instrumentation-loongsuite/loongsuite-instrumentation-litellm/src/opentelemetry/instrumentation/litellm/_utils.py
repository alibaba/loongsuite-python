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

import inspect
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, List, Optional
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
)
from opentelemetry.util.genai.extended_types import EmbeddingInvocation
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)

_COMPLETION_POSITIONAL_PARAMETERS = ("model", "messages")
_EMBEDDING_POSITIONAL_PARAMETERS = ("model", "input")
_BASE_URL_PROVIDER_MAP = (
    ("dashscope.aliyuncs.com", "dashscope"),
    ("api.openai.com", "openai"),
    ("api.deepseek.com", "deepseek"),
    ("anthropic.com", "anthropic"),
    ("generativelanguage.googleapis.com", "google"),
)
_BASE_URL_KWARG_NAMES = (
    "api_base",
    "base_url",
    "api_endpoint",
    "endpoint",
)
_SYSTEM_INSTRUCTION_ROLES = frozenset(("system", "developer"))


def get_litellm_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a value from LiteLLM dict, pydantic, or object responses."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_litellm_completion_kwargs(
    original_func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return request kwargs with positional LiteLLM completion args included."""
    return _normalize_litellm_kwargs(
        original_func, args, kwargs, _COMPLETION_POSITIONAL_PARAMETERS
    )


def normalize_litellm_embedding_kwargs(
    original_func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return request kwargs with positional LiteLLM embedding args included."""
    return _normalize_litellm_kwargs(
        original_func, args, kwargs, _EMBEDDING_POSITIONAL_PARAMETERS
    )


def _normalize_litellm_kwargs(
    original_func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    positional_names: tuple[str, ...],
) -> dict[str, Any]:
    normalized = dict(kwargs)

    for name, value in zip(positional_names, args):
        normalized.setdefault(name, value)

    try:
        signature = inspect.signature(original_func)
        bound_arguments = signature.bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        return normalized

    extra_kwargs = bound_arguments.pop("kwargs", None)
    bound_arguments.pop("args", None)
    normalized.update(bound_arguments)
    if isinstance(extra_kwargs, Mapping):
        normalized.update(extra_kwargs)
    return normalized


def parse_tool_call_arguments(arguments: Any) -> Any:
    """Parse JSON tool-call arguments when LiteLLM returns them as strings."""
    if isinstance(arguments, str) and arguments:
        try:
            return json.loads(arguments)
        except (TypeError, ValueError):
            return arguments
    return arguments


def extract_litellm_text_parts(content: Any) -> list[str]:
    """Extract text strings from LiteLLM text or multimodal content."""
    if isinstance(content, str):
        return [content] if content else []

    if not isinstance(content, list):
        return []

    text_parts = []
    for item in content:
        if isinstance(item, str):
            if item:
                text_parts.append(item)
            continue

        if not isinstance(item, Mapping) or item.get("type") != "text":
            continue

        text = item.get("text", item.get("content", ""))
        if isinstance(text, str) and text:
            text_parts.append(text)

    return text_parts


def apply_litellm_llm_response_to_invocation(
    invocation: LLMInvocation,
    response: Any,
    *,
    include_output_messages: bool = True,
) -> None:
    """Populate a GenAI LLMInvocation from a LiteLLM response or stream chunk."""
    if include_output_messages:
        output_messages = extract_output_from_litellm_response(response)
        if output_messages:
            invocation.output_messages = output_messages

    usage = get_litellm_value(response, "usage")
    _apply_usage_to_invocation(invocation, usage)

    response_id = get_litellm_value(response, "id")
    if response_id:
        invocation.response_id = response_id

    response_model = get_litellm_value(response, "model")
    if response_model:
        invocation.response_model_name = response_model

    finish_reasons = extract_finish_reasons_from_litellm_response(response)
    if finish_reasons:
        invocation.finish_reasons = finish_reasons


def apply_litellm_embedding_response_to_invocation(
    invocation: EmbeddingInvocation,
    response: Any,
) -> None:
    """Populate a GenAI EmbeddingInvocation from a LiteLLM response."""
    response_model = get_litellm_value(response, "model")
    if response_model:
        invocation.response_model_name = response_model

    usage = get_litellm_value(response, "usage")
    _apply_usage_to_invocation(invocation, usage, include_output_tokens=False)

    data = get_litellm_value(response, "data")
    if not data:
        return

    try:
        first_embedding = data[0]
        embedding_vector = get_litellm_value(first_embedding, "embedding")
        if isinstance(embedding_vector, list):
            invocation.dimension_count = len(embedding_vector)
    except (IndexError, AttributeError, KeyError, TypeError):
        logger.debug("Failed to extract LiteLLM embedding dimension count")


def extract_finish_reasons_from_litellm_response(response: Any) -> list[str]:
    """Extract non-empty finish reasons from LiteLLM choices."""
    choices = get_litellm_value(response, "choices") or []
    finish_reasons = []
    for choice in choices:
        finish_reason = get_litellm_value(choice, "finish_reason")
        if finish_reason:
            finish_reasons.append(finish_reason)
    return finish_reasons


def _apply_usage_to_invocation(
    invocation: Any,
    usage: Any,
    *,
    include_output_tokens: bool = True,
) -> None:
    if not usage:
        return

    input_tokens = get_litellm_value(usage, "prompt_tokens")
    output_tokens = get_litellm_value(usage, "completion_tokens")
    total_tokens = get_litellm_value(usage, "total_tokens")

    if (
        include_output_tokens
        and output_tokens is None
        and input_tokens is not None
        and total_tokens
    ):
        output_tokens = max(total_tokens - input_tokens, 0)

    if input_tokens is not None:
        invocation.input_tokens = input_tokens
    if include_output_tokens and output_tokens is not None:
        invocation.output_tokens = output_tokens

    prompt_details = get_litellm_value(usage, "prompt_tokens_details")
    cached_tokens = get_litellm_value(prompt_details, "cached_tokens")
    if cached_tokens is not None and hasattr(
        invocation, "usage_cache_read_input_tokens"
    ):
        invocation.usage_cache_read_input_tokens = cached_tokens

    cache_creation_tokens = get_litellm_value(
        prompt_details, "cache_creation_tokens"
    )
    if cache_creation_tokens is not None and hasattr(
        invocation, "usage_cache_creation_input_tokens"
    ):
        invocation.usage_cache_creation_input_tokens = cache_creation_tokens


def parse_provider_from_model(model: str) -> Optional[str]:
    """
    Parse provider name from model string.

    LiteLLM uses format like "openai/gpt-4", "dashscope/qwen-turbo", etc.
    """
    if not model or not isinstance(model, str):
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


def parse_provider_from_base_url(base_url: Any) -> Optional[str]:
    """Infer provider from known OpenAI-compatible service endpoints."""
    if not base_url or not isinstance(base_url, str):
        return None

    try:
        host = urlparse(base_url).hostname or base_url
    except ValueError:
        host = base_url

    host = host.lower()
    for fragment, provider in _BASE_URL_PROVIDER_MAP:
        if fragment in host:
            return provider
    return None


def resolve_litellm_provider(model: Any, kwargs: Mapping[str, Any]) -> str:
    """Resolve the actual GenAI provider for a LiteLLM request."""
    for name in _BASE_URL_KWARG_NAMES:
        provider = parse_provider_from_base_url(kwargs.get(name))
        if provider:
            return provider

    custom_provider = kwargs.get("custom_llm_provider")
    if custom_provider:
        return custom_provider

    provider = parse_provider_from_model(model)
    if provider not in (None, "unknown"):
        return provider

    return provider or "unknown"


def parse_model_name(model: str) -> str:
    """
    Parse model name by removing provider prefix.

    Examples:
        "openai/gpt-4" -> "gpt-4"
        "dashscope/qwen-turbo" -> "qwen-turbo"
        "gpt-4" -> "gpt-4"
    """
    if not model or not isinstance(model, str):
        return "unknown"

    if "/" in model:
        return model.split("/", 1)[1]

    return model


def convert_litellm_messages_to_genai_format(
    messages: list[dict[str, Any]],
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
        if role in _SYSTEM_INSTRUCTION_ROLES:
            continue

        parts = _extract_message_parts(msg, role)

        # If no parts added, add empty text
        if not parts:
            parts.append(Text(content=""))

        input_messages.append(InputMessage(role=role, parts=parts))

    return input_messages


def extract_system_instruction_from_litellm_messages(
    messages: list[dict[str, Any]],
) -> list:
    """Extract system/developer instructions from LiteLLM messages."""
    if not isinstance(messages, list):
        return []

    system_instruction = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") not in _SYSTEM_INSTRUCTION_ROLES:
            continue

        for text in extract_litellm_text_parts(msg.get("content")):
            system_instruction.append(Text(content=text))

    return system_instruction


def _extract_message_parts(msg: Mapping[str, Any], role: str) -> list:
    parts = []

    for text in extract_litellm_text_parts(msg.get("content")):
        parts.append(Text(content=text))

    # Handle tool calls
    if "tool_calls" in msg and msg["tool_calls"]:
        for tool_call in msg["tool_calls"]:
            if not isinstance(tool_call, Mapping):
                continue

            func = tool_call.get("function", {})
            if isinstance(func, Mapping):
                arguments = parse_tool_call_arguments(
                    func.get("arguments", "")
                )

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

    return parts


def extract_output_from_litellm_response(response: Any) -> List:
    """
    Extract output messages from LiteLLM response.

    Converts LiteLLM response to OpenTelemetry GenAI OutputMessage format.
    """

    choices = get_litellm_value(response, "choices") or []
    if not choices:
        return []

    output_messages = []
    for choice in choices:
        msg = get_litellm_value(choice, "message")
        parts = []
        role = "assistant"

        if msg is not None:
            role = get_litellm_value(msg, "role", "assistant")

            reasoning_content = get_litellm_value(msg, "reasoning_content")
            for text in extract_litellm_text_parts(reasoning_content):
                parts.append(Reasoning(content=text))

            # Extract text content
            content = get_litellm_value(msg, "content")
            for text in extract_litellm_text_parts(content):
                parts.append(Text(content=text))

            # Extract tool calls
            tool_calls = get_litellm_value(msg, "tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    function = get_litellm_value(tc, "function")
                    arguments = parse_tool_call_arguments(
                        get_litellm_value(function, "arguments", "")
                    )

                    parts.append(
                        ToolCall(
                            id=get_litellm_value(tc, "id"),
                            name=get_litellm_value(function, "name", ""),
                            arguments=arguments,
                        )
                    )

        # If no parts, add empty text
        if not parts:
            parts.append(Text(content=""))

        finish_reason = get_litellm_value(choice, "finish_reason") or "stop"

        output_messages.append(
            OutputMessage(
                role=role,
                parts=parts,
                finish_reason=finish_reason,
            )
        )

    return output_messages


def create_llm_invocation_from_litellm(**kwargs):
    """
    Create LLMInvocation from LiteLLM request parameters.

    The provider is resolved from known base URLs, custom_llm_provider, or the
    model name.

    Args:
        model: The model name (e.g., "gpt-4", "openai/gpt-4")
        messages: List of message dictionaries
        **kwargs: Additional request parameters (temperature, max_tokens, etc.)

    Returns:
        LLMInvocation object ready for use with ExtendedTelemetryHandler
    """

    # Parse model name (remove provider prefix if present)
    model = kwargs.get("model", "unknown_model")
    provider = resolve_litellm_provider(model, kwargs)
    messages = kwargs.get("messages", [])

    # Convert messages to GenAI format
    input_messages = convert_litellm_messages_to_genai_format(messages)
    system_instruction = extract_system_instruction_from_litellm_messages(
        messages
    )

    request_model = parse_model_name(model)

    invocation = LLMInvocation(
        request_model=request_model,
        provider=provider,
        operation_name=GenAiOperationNameValues.CHAT.value,
        input_messages=input_messages,
    )
    if system_instruction:
        invocation.system_instruction = system_instruction

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
    if "n" in kwargs and kwargs["n"] is not None:
        invocation.choice_count = kwargs["n"]
    if "top_k" in kwargs and kwargs["top_k"] is not None:
        invocation.top_k = kwargs["top_k"]
    if "response_format" in kwargs and kwargs["response_format"] is not None:
        response_format = kwargs["response_format"]
        if isinstance(response_format, Mapping):
            invocation.output_type = response_format.get("type")
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

    The provider is resolved from known base URLs, custom_llm_provider, or the
    model name.

    Args:
        model: The embedding model name
        **kwargs: Additional request parameters

    Returns:
        EmbeddingInvocation object ready for use with ExtendedTelemetryHandler
    """

    # Extract request parameters
    model = kwargs.get("model", "unknown")
    provider = resolve_litellm_provider(model, kwargs)

    # Parse model name (remove provider prefix if present)
    request_model = parse_model_name(model)

    invocation = EmbeddingInvocation(
        request_model=request_model,
        provider=provider,
    )

    # Set encoding formats if present
    if "encoding_format" in kwargs and kwargs["encoding_format"]:
        invocation.encoding_formats = [kwargs["encoding_format"]]

    return invocation
