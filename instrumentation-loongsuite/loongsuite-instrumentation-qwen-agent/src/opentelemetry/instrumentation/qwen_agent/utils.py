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

"""Utility functions for Qwen-Agent instrumentation.

Handles conversion between qwen-agent Message types and
OpenTelemetry GenAI semantic convention types.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GenAiExtendedProviderNameValues,
)
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
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

# Map qwen-agent model_type to provider name.
_MODEL_TYPE_PROVIDER_MAP = {
    "qwen_dashscope": GenAiExtendedProviderNameValues.DASHSCOPE.value,
    "qwenvl_dashscope": GenAiExtendedProviderNameValues.DASHSCOPE.value,
    "qwenaudio_dashscope": GenAiExtendedProviderNameValues.DASHSCOPE.value,
    "qwenvlo_dashscope": GenAiExtendedProviderNameValues.DASHSCOPE.value,
    "oai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "azure": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "qwenvl_oai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "qwenomni_oai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
}


def _get_provider_name(llm_instance: Any) -> str:
    """Extract provider name from a qwen-agent LLM instance.

    Args:
        llm_instance: A BaseChatModel instance from qwen-agent.

    Returns:
        Provider name string.
    """
    model_type = getattr(llm_instance, "model_type", "")
    if model_type in _MODEL_TYPE_PROVIDER_MAP:
        return _MODEL_TYPE_PROVIDER_MAP[model_type]

    # Fallback: infer from class name
    class_name = type(llm_instance).__name__.lower()
    if "dashscope" in class_name:
        return GenAiExtendedProviderNameValues.DASHSCOPE.value
    if "openai" in class_name or "oai" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    if "azure" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value

    return GenAiExtendedProviderNameValues.DASHSCOPE.value


def _extract_content_text(content: Any) -> str:
    """Extract text from qwen-agent Message content field.

    Content can be str or List[ContentItem].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if hasattr(item, "text") and item.text is not None:
                texts.append(item.text)
            elif hasattr(item, "get_type_and_value"):
                t, v = item.get_type_and_value()
                if t == "text":
                    texts.append(v)
        return "\n".join(texts)
    return str(content) if content else ""


def _convert_qwen_messages_to_input_messages(
    messages: Any,
) -> List[InputMessage]:
    """Convert qwen-agent Message list to GenAI InputMessage format.

    Args:
        messages: List of qwen-agent Message objects or dicts.

    Returns:
        List of InputMessage objects for ExtendedTelemetryHandler.
    """
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    input_messages = []
    for msg in messages:
        try:
            role = (
                msg.role if hasattr(msg, "role") else msg.get("role", "user")
            )
            content = (
                msg.content
                if hasattr(msg, "content")
                else msg.get("content", "")
            )
            function_call = (
                msg.function_call
                if hasattr(msg, "function_call")
                else msg.get("function_call")
            )
            name = msg.name if hasattr(msg, "name") else msg.get("name")

            parts = []

            # Handle function_call (tool call from assistant)
            if function_call:
                fc_name = (
                    function_call.name
                    if hasattr(function_call, "name")
                    else function_call.get("name", "")
                )
                fc_args = (
                    function_call.arguments
                    if hasattr(function_call, "arguments")
                    else function_call.get("arguments", "{}")
                )
                # Parse arguments if string
                if isinstance(fc_args, str):
                    try:
                        fc_args = json.loads(fc_args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                parts.append(
                    ToolCall(name=fc_name, arguments=fc_args, id=None)
                )

            # Handle function/tool role (tool response).
            # qwen-agent uses role="function" internally, but DashScope API
            # converts it to role="tool" (see base.py:445). Handle both.
            if role in ("function", "tool") and content:
                text = _extract_content_text(content)

                # Extract tool_call_id: prefer msg.id, then msg.extra.function_id,
                # then fall back to tool name.
                tool_call_id: str = ""
                if hasattr(msg, "id"):
                    tool_call_id = getattr(msg, "id", "") or ""
                elif isinstance(msg, dict):
                    tool_call_id = msg.get("id") or ""

                if not tool_call_id:
                    extra = (
                        getattr(msg, "extra", None)
                        if not isinstance(msg, dict)
                        else msg.get("extra")
                    )
                    if extra is not None:
                        if isinstance(extra, dict):
                            tool_call_id = extra.get("function_id") or ""
                        else:
                            tool_call_id = (
                                getattr(extra, "function_id", "") or ""
                            )

                if not tool_call_id:
                    tool_call_id = name or ""

                parts.append(
                    ToolCallResponse(
                        id=tool_call_id,
                        response=text,
                    )
                )
            elif content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if parts:
                input_messages.append(InputMessage(role=role, parts=parts))

        except Exception as e:
            logger.debug(f"Error converting message: {e}")
            continue

    return input_messages


def _convert_qwen_messages_to_output_messages(
    messages: Any,
) -> List[OutputMessage]:
    """Convert qwen-agent response messages to GenAI OutputMessage format.

    Args:
        messages: List of qwen-agent Message objects (LLM output).

    Returns:
        List of OutputMessage objects.
    """
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    output_messages = []
    for msg in messages:
        try:
            content = (
                msg.content
                if hasattr(msg, "content")
                else msg.get("content", "")
            )
            function_call = (
                msg.function_call
                if hasattr(msg, "function_call")
                else msg.get("function_call")
            )

            parts = []
            finish_reason = "stop"

            if function_call:
                fc_name = (
                    function_call.name
                    if hasattr(function_call, "name")
                    else function_call.get("name", "")
                )
                fc_args = (
                    function_call.arguments
                    if hasattr(function_call, "arguments")
                    else function_call.get("arguments", "{}")
                )
                if isinstance(fc_args, str):
                    try:
                        fc_args = json.loads(fc_args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                parts.append(
                    ToolCall(name=fc_name, arguments=fc_args, id=None)
                )
                finish_reason = "tool_calls"

            if content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if not parts:
                parts.append(Text(content=""))

            output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=finish_reason,
                )
            )

        except Exception as e:
            logger.debug(f"Error converting output message: {e}")
            continue

    return output_messages


def _get_tool_definitions(
    functions: Optional[List[Dict]],
) -> Optional[List[FunctionToolDefinition]]:
    """Extract tool definitions for tracing as FunctionToolDefinition objects.

    Args:
        functions: List of function dicts in qwen-agent format
                   (each with 'name', 'description', 'parameters').

    Returns:
        List of FunctionToolDefinition objects, or None.
    """
    if not functions:
        return None

    try:
        tool_defs = []
        for func in functions:
            if not isinstance(func, dict):
                continue
            name = func.get("name")
            if not name:
                continue
            tool_defs.append(
                FunctionToolDefinition(
                    name=name,
                    description=func.get("description"),
                    parameters=func.get("parameters"),
                )
            )
        if tool_defs:
            return tool_defs
    except Exception:
        pass

    return None


def _create_llm_invocation(
    llm_instance: Any,
    messages: Any,
    functions: Optional[List[Dict]] = None,
    stream: bool = True,
    extra_generate_cfg: Optional[Dict] = None,
) -> LLMInvocation:
    """Create LLMInvocation from qwen-agent BaseChatModel.chat() parameters.

    Args:
        llm_instance: The BaseChatModel instance.
        messages: Input messages.
        functions: Tool function definitions.
        stream: Whether streaming.
        extra_generate_cfg: Extra generation config.

    Returns:
        LLMInvocation for ExtendedTelemetryHandler.
    """
    provider_name = _get_provider_name(llm_instance)
    request_model = getattr(llm_instance, "model", "unknown_model")

    input_messages = _convert_qwen_messages_to_input_messages(messages)

    invocation = LLMInvocation(
        request_model=request_model,
        provider=provider_name,
        input_messages=input_messages,
    )

    # Set generation parameters
    if extra_generate_cfg:
        if extra_generate_cfg.get("max_tokens"):
            invocation.max_tokens = extra_generate_cfg["max_tokens"]
        if extra_generate_cfg.get("temperature"):
            invocation.temperature = extra_generate_cfg["temperature"]
        if extra_generate_cfg.get("top_p"):
            invocation.top_p = extra_generate_cfg["top_p"]

    # Set tool definitions
    tool_definitions = _get_tool_definitions(functions)
    if tool_definitions:
        invocation.tool_definitions = tool_definitions

    return invocation


def _create_agent_invocation(
    agent_instance: Any,
    messages: Any,
) -> InvokeAgentInvocation:
    """Create InvokeAgentInvocation from qwen-agent Agent.run() parameters.

    Args:
        agent_instance: The Agent instance.
        messages: Input messages.

    Returns:
        InvokeAgentInvocation for ExtendedTelemetryHandler.
    """
    # Get provider and model from agent's LLM
    provider_name = None
    request_model = None
    if hasattr(agent_instance, "llm") and agent_instance.llm:
        provider_name = _get_provider_name(agent_instance.llm)
        request_model = getattr(agent_instance.llm, "model", None)

    input_messages = _convert_qwen_messages_to_input_messages(messages)

    agent_name = (
        getattr(agent_instance, "name", None) or type(agent_instance).__name__
    )
    agent_description = getattr(agent_instance, "description", None) or ""

    invocation = InvokeAgentInvocation(
        provider=provider_name,
        agent_name=agent_name,
        agent_id=agent_name,
        agent_description=agent_description,
        request_model=request_model,
        input_messages=input_messages,
    )

    # Set system instruction if available.
    # Qwen-Agent's Agent.system_message is a plain string (Optional[str])
    # that serves as the system prompt prepended to messages (see agent.py:113).
    # This conforms to the gen_ai.system_instructions semantic convention:
    # "The full system instructions (also known as system prompt)".
    # Note: agent.py:117 may concatenate system_message with an existing
    # system message in the input, but we capture the original system_message
    # as the agent's configured instruction.
    if (
        hasattr(agent_instance, "system_message")
        and agent_instance.system_message
    ):
        invocation.system_instruction = [
            Text(content=agent_instance.system_message)
        ]

    return invocation


def _create_tool_invocation(
    tool_name: str,
    tool_args: Any = None,
    tool_instance: Any = None,
) -> ExecuteToolInvocation:
    """Create ExecuteToolInvocation from qwen-agent tool call parameters.

    Args:
        tool_name: Name of the tool.
        tool_args: Tool arguments (str or dict).
        tool_instance: The BaseTool instance, if available.

    Returns:
        ExecuteToolInvocation for ExtendedTelemetryHandler.
    """
    # Parse tool_args
    if isinstance(tool_args, str):
        try:
            parsed_args = json.loads(tool_args)
        except (json.JSONDecodeError, ValueError):
            parsed_args = {"raw_args": tool_args}
    elif isinstance(tool_args, dict):
        parsed_args = tool_args
    else:
        parsed_args = {}

    tool_description = None
    if tool_instance:
        tool_description = getattr(tool_instance, "description", None)

    return ExecuteToolInvocation(
        tool_name=tool_name,
        tool_type="function",
        tool_call_arguments=parsed_args,
        tool_description=tool_description,
    )
