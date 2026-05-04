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

"""Utility functions for Anthropic instrumentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
)

if TYPE_CHECKING:
    from anthropic.resources.messages import Messages


def is_streaming(kwargs: dict[str, Any]) -> bool:
    """Check if the request is a streaming request."""
    return bool(kwargs.get("stream"))


def get_server_address_and_port(
    client_instance: "Messages",
) -> tuple[str | None, int | None]:
    """Extract server address and port from the Anthropic client instance."""
    base_client = getattr(client_instance, "_client", None)
    base_url = getattr(base_client, "base_url", None)
    if not base_url:
        return None, None

    address: str | None = None
    port: int | None = None

    if hasattr(base_url, "host"):
        # httpx.URL object
        address = base_url.host
        port = getattr(base_url, "port", None)
    elif isinstance(base_url, str):
        url = urlparse(base_url)
        address = url.hostname
        port = url.port

    if port == 443:
        port = None

    return address, port


def create_anthropic_invocation(
    kwargs: dict[str, Any],
    client_instance: "Messages",
    capture_content: bool,
) -> LLMInvocation:
    """Create an LLMInvocation from Anthropic Messages.create() parameters.

    This populates the LLMInvocation with all semantic convention attributes
    from the request parameters, including content capture if enabled.
    """
    invocation = LLMInvocation(
        request_model=kwargs.get("model", ""),
    )
    invocation.provider = (
        GenAIAttributes.GenAiSystemValues.ANTHROPIC.value  # pyright: ignore[reportDeprecated]
    )

    # Request parameters
    invocation.max_tokens = kwargs.get("max_tokens")
    invocation.temperature = kwargs.get("temperature")
    invocation.top_p = kwargs.get("top_p")
    invocation.top_k = kwargs.get("top_k")

    stop_sequences = kwargs.get("stop_sequences")
    if stop_sequences is not None:
        invocation.stop_sequences = list(stop_sequences)

    # Server address
    address, port = get_server_address_and_port(client_instance)
    if address:
        invocation.server_address = address
    if port:
        invocation.server_port = port

    # Content capture
    if capture_content:
        # Input messages
        messages = kwargs.get("messages", [])
        invocation.input_messages = _convert_messages_to_input(messages)

        # System instruction
        system = kwargs.get("system")
        if system:
            invocation.system_instruction = _convert_system_to_parts(system)

        # Tool definitions
        tools = kwargs.get("tools")
        if tools:
            invocation.tool_definitions = _convert_tools_to_definitions(tools)

    return invocation


def populate_response(
    invocation: LLMInvocation,
    result: Any,
    capture_content: bool,
) -> None:
    """Populate an LLMInvocation with response data from an Anthropic Message."""
    if result is None:
        return

    if getattr(result, "model", None):
        invocation.response_model_name = result.model

    if getattr(result, "id", None):
        invocation.response_id = result.id

    if getattr(result, "stop_reason", None):
        invocation.finish_reasons = [result.stop_reason]

    # Token usage
    usage = getattr(result, "usage", None)
    if usage:
        if hasattr(usage, "input_tokens"):
            invocation.input_tokens = usage.input_tokens
        if hasattr(usage, "output_tokens"):
            invocation.output_tokens = usage.output_tokens
        # Cache token usage (Anthropic-specific)
        if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens:
            invocation.usage_cache_creation_input_tokens = usage.cache_creation_input_tokens
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            invocation.usage_cache_read_input_tokens = usage.cache_read_input_tokens

    # Output messages (content blocks)
    if capture_content:
        content = getattr(result, "content", None)
        if content:
            invocation.output_messages = _convert_content_blocks_to_output(
                content, getattr(result, "stop_reason", None)
            )


def _convert_messages_to_input(
    messages: list[dict[str, Any]],
) -> list[InputMessage]:
    """Convert Anthropic message format to InputMessage list."""
    input_messages: list[InputMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        parts: list[MessagePart] = []
        content = msg.get("content")

        if isinstance(content, str):
            parts.append(Text(content=content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        parts.append(Text(content=block.get("text", "")))
                    elif block_type == "tool_use":
                        parts.append(
                            ToolCall(
                                name=block.get("name", ""),
                                id=block.get("id"),
                                arguments=block.get("input"),
                            )
                        )
                    elif block_type == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            # Extract text from content blocks
                            text_parts = []
                            for sub_block in tool_content:
                                if isinstance(sub_block, dict) and sub_block.get("type") == "text":
                                    text_parts.append(sub_block.get("text", ""))
                            tool_content = "\n".join(text_parts)
                        parts.append(
                            ToolCallResponse(
                                id=block.get("tool_use_id"),
                                response=tool_content,
                            )
                        )
                    elif block_type == "image":
                        # Skip image blocks for now (binary data)
                        pass
                elif isinstance(block, str):
                    parts.append(Text(content=block))

        input_messages.append(InputMessage(role=role, parts=parts))
    return input_messages


def _convert_system_to_parts(
    system: Any,
) -> list[MessagePart]:
    """Convert Anthropic system prompt to MessagePart list."""
    parts: list[MessagePart] = []
    if isinstance(system, str):
        parts.append(Text(content=system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(Text(content=block.get("text", "")))
            elif isinstance(block, str):
                parts.append(Text(content=block))
    return parts


def _convert_tools_to_definitions(
    tools: list[Any],
) -> list[FunctionToolDefinition]:
    """Convert Anthropic tool definitions to FunctionToolDefinition list."""
    definitions: list[FunctionToolDefinition] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description")
            input_schema = tool.get("input_schema")
            definitions.append(
                FunctionToolDefinition(
                    name=name,
                    description=description,
                    parameters=input_schema,
                )
            )
        else:
            # Pydantic model or other object
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", None)
            input_schema = getattr(tool, "input_schema", None)
            tool_type = getattr(tool, "type", "custom")
            if tool_type == "custom" or name:
                definitions.append(
                    FunctionToolDefinition(
                        name=name,
                        description=description,
                        parameters=input_schema,
                    )
                )
    return definitions


def _convert_content_blocks_to_output(
    content: list[Any],
    stop_reason: str | None,
) -> list[OutputMessage]:
    """Convert Anthropic content blocks to OutputMessage list."""
    parts: list[MessagePart] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            parts.append(Text(content=text))
        elif block_type == "tool_use":
            name = getattr(block, "name", "")
            tool_id = getattr(block, "id", None)
            tool_input = getattr(block, "input", None)
            parts.append(
                ToolCall(
                    name=name,
                    id=tool_id,
                    arguments=tool_input,
                )
            )
        elif block_type == "thinking":
            # Extended thinking block - capture as text for now
            thinking_text = getattr(block, "thinking", "")
            if thinking_text:
                parts.append(Reasoning(content=thinking_text))

    if parts:
        return [
            OutputMessage(
                role="assistant",
                parts=parts,
                finish_reason=stop_reason or "error",
            )
        ]
    return []
