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

"""Utility functions for DashScope Generation API instrumentation."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
)

from .common import _extract_usage, _get_parameter

logger = logging.getLogger(__name__)

_MISSING = object()


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from dict-like DashScope objects without leaking KeyError."""
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    try:
        get = getattr(obj, "get", None)
    except (AttributeError, KeyError):
        get = None

    if callable(get):
        try:
            return get(key, default)
        except (AttributeError, KeyError, TypeError):
            pass

    try:
        return getattr(obj, key)
    except (AttributeError, KeyError):
        return default


def _extract_input_messages(kwargs: dict) -> List[InputMessage]:
    """Extract input messages from DashScope API kwargs.

    DashScope supports both `prompt` (string) and `messages` (list) formats.
    Also supports tool call responses in messages.
    """
    input_messages = []

    # Check for messages format (preferred)
    if "messages" in kwargs and kwargs["messages"]:
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id")
                tool_calls = msg.get("tool_calls")

                parts = []

                # Handle tool call response (role="tool")
                if role == "tool":
                    # For tool role, use ToolCallResponse
                    parts.append(
                        ToolCallResponse(
                            response=content,
                            id=tool_call_id,
                            type="tool_call_response",
                        )
                    )
                else:
                    # Add text content if present (for non-tool roles)
                    if content:
                        if isinstance(content, str):
                            parts.append(Text(content=content, type="text"))
                        elif isinstance(content, list):
                            # Handle multimodal content (qwen-vl)
                            for part in content:
                                if isinstance(part, dict):
                                    if "text" in part:
                                        parts.append(
                                            Text(
                                                content=part["text"],
                                                type="text",
                                            )
                                        )
                                elif isinstance(part, str):
                                    parts.append(
                                        Text(content=part, type="text")
                                    )

                # Add tool calls if present (for assistant messages with tool calls)
                if tool_calls and isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tc_id = tool_call.get("id")
                            function = tool_call.get("function", {})
                            tool_name = (
                                function.get("name")
                                if isinstance(function, dict)
                                else None
                            )
                            tool_args = (
                                function.get("arguments")
                                if isinstance(function, dict)
                                else None
                            )

                            if tool_name:
                                parts.append(
                                    ToolCall(
                                        name=tool_name,
                                        arguments=tool_args,
                                        id=tc_id,
                                        type="tool_call",
                                    )
                                )

                if parts:
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=parts,
                        )
                    )
                elif content:  # Fallback for text-only messages
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=[Text(content=str(content), type="text")],
                        )
                    )
            elif _safe_get(msg, "role", _MISSING) is not _MISSING:
                # Handle message objects
                role = _safe_get(msg, "role", "user")
                content = _safe_get(msg, "content", "")
                tool_call_id = _safe_get(msg, "tool_call_id")
                tool_calls = _safe_get(msg, "tool_calls")

                parts = []

                # Handle tool call response (role="tool")
                if role == "tool":
                    # For tool role, use ToolCallResponse
                    parts.append(
                        ToolCallResponse(
                            response=str(content),
                            id=tool_call_id,
                            type="tool_call_response",
                        )
                    )
                else:
                    # Add text content if present (for non-tool roles)
                    if content:
                        if isinstance(content, str):
                            parts.append(Text(content=content, type="text"))
                        elif isinstance(content, list):
                            # TODO: Handle multimodal content
                            for part in content:
                                if isinstance(part, dict) and "text" in part:
                                    parts.append(
                                        Text(content=part["text"], type="text")
                                    )
                                elif isinstance(part, str):
                                    parts.append(
                                        Text(content=part, type="text")
                                    )

                # Add tool calls if present
                if tool_calls:
                    for tool_call in tool_calls:
                        function = _safe_get(tool_call, "function", _MISSING)
                        if function is _MISSING:
                            continue
                        if function:
                            tool_name = _safe_get(function, "name")
                            tool_args = _safe_get(function, "arguments")
                            tc_id = _safe_get(tool_call, "id")

                            if tool_name:
                                parts.append(
                                    ToolCall(
                                        name=tool_name,
                                        arguments=tool_args,
                                        id=tc_id,
                                        type="tool_call",
                                    )
                                )

                if parts:
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=parts,
                        )
                    )
                elif content:
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=[Text(content=str(content), type="text")],
                        )
                    )

    # Check for prompt format (legacy)
    elif "prompt" in kwargs and kwargs["prompt"]:
        prompt = kwargs["prompt"]
        if isinstance(prompt, str):
            input_messages.append(
                InputMessage(
                    role="user",
                    parts=[Text(content=prompt, type="text")],
                )
            )

    return input_messages


def _extract_tool_definitions(kwargs: dict) -> list[ToolDefinition]:
    """Extract tool definitions from DashScope API kwargs and convert to FunctionToolDefinition objects.

    DashScope supports both `tools` and `plugins` parameters for tool definitions.
    - `tools`: Direct list of tool definitions (preferred)
    - `plugins`: Can be a string (JSON) or a dict containing tools

    Args:
        kwargs: Generation.call kwargs

    Returns:
        List of FunctionToolDefinition objects, or empty list if not found
    """
    tool_definitions: list[ToolDefinition] = []

    # Check for tools parameter first (preferred)
    tools = kwargs.get("tools")
    if not tools:
        # Check for plugins parameter
        plugins = kwargs.get("plugins")
        if plugins:
            try:
                # If plugins is a string, parse it as JSON
                if isinstance(plugins, str):
                    plugins = json.loads(plugins)

                # If plugins is a dict, extract tools
                if isinstance(plugins, dict):
                    # DashScope plugins format: {"tools": [...]} or direct list
                    if "tools" in plugins:
                        tools = plugins["tools"]
                    # Check if plugins itself is a list-like structure
                    elif isinstance(plugins, list):
                        tools = plugins

                # If plugins is already a list, use it
                if isinstance(plugins, list):
                    tools = plugins
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                # If parsing fails, return empty list
                logger.debug(
                    "Failed to parse tool definitions from response: %s", e
                )
                return tool_definitions

    # Convert tool definitions to FunctionToolDefinition objects
    if tools and isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                # Extract function definition (DashScope format: {"type": "function", "function": {...}})
                function = tool.get("function", {})
                if isinstance(function, dict):
                    tool_def = FunctionToolDefinition(
                        name=function.get("name", ""),
                        description=function.get("description"),
                        parameters=function.get("parameters"),
                        type="function",
                    )
                    tool_definitions.append(tool_def)
                # Handle case where tool itself is a function definition
                elif "name" in tool:
                    tool_def = FunctionToolDefinition(
                        name=tool.get("name", ""),
                        description=tool.get("description"),
                        parameters=tool.get("parameters"),
                        type="function",
                    )
                    tool_definitions.append(tool_def)
            elif isinstance(tool, FunctionToolDefinition):
                # Already a FunctionToolDefinition, add directly
                tool_definitions.append(tool)

    return tool_definitions


def _extract_output_messages(response: Any) -> List[OutputMessage]:
    """Extract output messages from DashScope GenerationResponse.

    Supports both standard format (output.text) and qwen-vl format (output.choices).

    Args:
        response: DashScope GenerationResponse object

    Returns:
        List of OutputMessage objects
    """
    output_messages = []

    if not response:
        return output_messages

    try:
        output = _safe_get(response, "output")
        if not output:
            return output_messages

        # Check for choices format (qwen-vl and some models)
        choices = _safe_get(output, "choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            # Process each choice
            for choice in choices:
                if not choice:
                    continue

                # Extract message from choice
                message = _safe_get(choice, "message")
                if not message:
                    continue

                # Extract content and tool_calls
                content = _safe_get(message, "content")
                tool_calls = _safe_get(message, "tool_calls")
                finish_reason = _safe_get(
                    choice, "finish_reason"
                ) or _safe_get(output, "finish_reason", "stop")

                parts = []

                # Add text content if present
                if content:
                    if isinstance(content, str):
                        parts.append(Text(content=content, type="text"))
                    elif isinstance(content, list):
                        # Handle multimodal content (qwen-vl)
                        for part in content:
                            if isinstance(part, dict):
                                if "text" in part:
                                    parts.append(
                                        Text(content=part["text"], type="text")
                                    )
                            elif isinstance(part, str):
                                parts.append(Text(content=part, type="text"))

                # Add tool calls if present
                if tool_calls and isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id")
                            function = tool_call.get("function", {})
                            tool_name = (
                                function.get("name")
                                if isinstance(function, dict)
                                else None
                            )
                            tool_args = (
                                function.get("arguments")
                                if isinstance(function, dict)
                                else None
                            )

                            if tool_name:
                                parts.append(
                                    ToolCall(
                                        name=tool_name,
                                        arguments=tool_args,
                                        id=tool_call_id,
                                        type="tool_call",
                                    )
                                )
                            continue

                        # Handle tool call objects
                        function = _safe_get(tool_call, "function", _MISSING)
                        if function is _MISSING:
                            continue
                        if function:
                            tool_name = _safe_get(function, "name")
                            tool_args = _safe_get(function, "arguments")
                            tool_call_id = _safe_get(tool_call, "id")

                            if tool_name:
                                parts.append(
                                    ToolCall(
                                        name=tool_name,
                                        arguments=tool_args,
                                        id=tool_call_id,
                                        type="tool_call",
                                    )
                                )

                # Create output message if we have parts OR if finish_reason indicates tool_calls
                # (even if content is empty, tool calls should be captured)
                if parts or (finish_reason == "tool_calls" and tool_calls):
                    output_messages.append(
                        OutputMessage(
                            role="assistant",
                            parts=parts,
                            finish_reason=finish_reason or "stop",
                        )
                    )
        else:
            # Standard format: output.text
            text = _safe_get(output, "text") or _safe_get(output, "content")
            finish_reason = _safe_get(output, "finish_reason", "stop")

            if text:
                output_messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=text, type="text")],
                        finish_reason=finish_reason or "stop",
                    )
                )
    except (KeyError, AttributeError) as e:
        # If any attribute access fails, return empty list
        logger.debug("Failed to extract output messages from response: %s", e)
        return output_messages

    return output_messages


def _create_invocation_from_generation(
    kwargs: dict, model: Optional[str] = None
) -> LLMInvocation:
    """Create LLMInvocation from Generation.call kwargs.

    Args:
        kwargs: Generation.call kwargs
        model: Model name (if not in kwargs)

    Returns:
        LLMInvocation object
    """
    request_model = kwargs.get("model") or model
    if not request_model:
        raise ValueError("Model name is required")

    invocation = LLMInvocation(request_model=request_model)
    invocation.provider = "dashscope"
    # DashScope SDK always targets dashscope.aliyuncs.com; if custom endpoint
    # support is added in the future, extract from instance/env instead.
    invocation.server_address = "dashscope.aliyuncs.com"
    invocation.server_port = 443
    invocation.input_messages = _extract_input_messages(kwargs)

    # Extract tool definitions and convert to FunctionToolDefinition objects
    invocation.tool_definitions = _extract_tool_definitions(kwargs)

    # Extract parameters from kwargs or kwargs["parameters"] dict
    # Parameters can be passed directly as kwargs (e.g., temperature=0.7) or
    # explicitly in a parameters dict. Check kwargs first, then parameters dict.
    parameters = kwargs.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    # Temperature
    temperature = _get_parameter(kwargs, "temperature", parameters)
    if temperature is not None:
        invocation.attributes["gen_ai.request.temperature"] = temperature

    # Top-p
    top_p = _get_parameter(kwargs, "top_p", parameters)
    if top_p is not None:
        invocation.attributes["gen_ai.request.top_p"] = top_p

    # Top-k
    top_k = _get_parameter(kwargs, "top_k", parameters)
    if top_k is not None:
        invocation.attributes["gen_ai.request.top_k"] = top_k

    # Max tokens
    max_tokens = _get_parameter(kwargs, "max_tokens", parameters)
    if max_tokens is not None:
        invocation.attributes["gen_ai.request.max_tokens"] = max_tokens

    # Repetition penalty
    repetition_penalty = _get_parameter(
        kwargs, "repetition_penalty", parameters
    )
    if repetition_penalty is not None:
        invocation.attributes["gen_ai.request.repetition_penalty"] = (
            repetition_penalty
        )
    else:
        # Fallback to frequency_penalty and presence_penalty if repetition_penalty not present
        frequency_penalty = _get_parameter(
            kwargs, "frequency_penalty", parameters
        )
        if frequency_penalty is not None:
            invocation.attributes["gen_ai.request.frequency_penalty"] = (
                frequency_penalty
            )

        presence_penalty = _get_parameter(
            kwargs, "presence_penalty", parameters
        )
        if presence_penalty is not None:
            invocation.attributes["gen_ai.request.presence_penalty"] = (
                presence_penalty
            )

    # Stop sequences
    stop_sequences = _get_parameter(kwargs, "stop", parameters)
    if stop_sequences is not None:
        if isinstance(stop_sequences, list):
            invocation.attributes["gen_ai.request.stop_sequences"] = (
                stop_sequences
            )
        elif isinstance(stop_sequences, str):
            invocation.attributes["gen_ai.request.stop_sequences"] = [
                stop_sequences
            ]

    # Seed
    seed = _get_parameter(kwargs, "seed", parameters)
    if seed is not None:
        invocation.attributes["gen_ai.request.seed"] = seed

    return invocation


def _update_invocation_from_response(
    invocation: LLMInvocation, response: Any
) -> None:
    """Update LLMInvocation with response data.

    Args:
        invocation: LLMInvocation to update
        response: DashScope response object
    """
    if not response:
        return

    try:
        # Extract output messages
        invocation.output_messages = _extract_output_messages(response)

        # Extract token usage
        input_tokens, output_tokens = _extract_usage(response)
        invocation.input_tokens = input_tokens
        invocation.output_tokens = output_tokens

        # Extract cache token usage
        from ..utils.common import _extract_cache_tokens  # noqa: PLC0415

        cache_creation, cache_read = _extract_cache_tokens(response)
        if cache_creation is not None:
            invocation.usage_cache_creation_input_tokens = cache_creation
        if cache_read is not None:
            invocation.usage_cache_read_input_tokens = cache_read

        # Extract response model name (if available)
        response_model = _safe_get(response, "model")
        if response_model:
            invocation.response_model_name = response_model

        # Extract request ID (if available)
        request_id = _safe_get(response, "request_id")
        if request_id:
            invocation.response_id = request_id
    except (KeyError, AttributeError) as e:
        # If any attribute access fails, silently continue with available data
        logger.debug(
            "Failed to extract response model name or request id from response: %s",
            e,
        )


def _create_accumulated_response(original_response, accumulated_text):
    """Create a response object with accumulated text for incremental output mode.

    Args:
        original_response: The last chunk response object
        accumulated_text: The accumulated text from all chunks

    Returns:
        A response object with accumulated text, or original_response if modification fails
    """
    try:
        output = _safe_get(original_response, "output")
        if output and _safe_get(output, "text") is not None:
            # Try to set the accumulated text directly
            try:
                output.text = accumulated_text
                return original_response
            except (AttributeError, TypeError) as e:
                # If we can't modify, create a wrapper object
                logger.debug(
                    "Failed to modify output.text directly, creating wrapper: %s",
                    e,
                )

        # Create wrapper objects with accumulated text
        class AccumulatedOutput:
            def __init__(self, original_output, accumulated_text):
                self.text = accumulated_text
                self.finish_reason = _safe_get(
                    original_output, "finish_reason", "stop"
                )
                self.content = accumulated_text

        class AccumulatedResponse:
            def __init__(self, original_response, accumulated_output):
                self.output = accumulated_output
                # Copy other attributes from original response
                for attr in ["usage", "request_id", "model"]:
                    value = _safe_get(original_response, attr)
                    if value is not None:
                        setattr(self, attr, value)

        accumulated_output = AccumulatedOutput(output, accumulated_text)
        return AccumulatedResponse(original_response, accumulated_output)
    except (KeyError, AttributeError) as e:
        # If modification fails, return original response
        logger.debug(
            "Failed to create accumulated response, returning original: %s", e
        )
        return original_response
