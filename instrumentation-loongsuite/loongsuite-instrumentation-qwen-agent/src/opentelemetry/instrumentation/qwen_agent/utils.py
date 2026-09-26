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


def _field_value(value: Any, *names: str) -> Any:
    """Read the first present field from a mapping or SDK response object."""
    if value is None:
        return None

    for name in names:
        if isinstance(value, dict):
            if name in value:
                return value[name]
            continue

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


def _int_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_token_values(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}

    input_tokens = _int_value(
        _field_value(usage, "input_tokens", "prompt_tokens")
    )
    output_tokens = _int_value(
        _field_value(usage, "output_tokens", "completion_tokens")
    )
    cache_read_tokens = _int_value(
        _field_value(usage, "cache_read_input_tokens", "cached_prompt_tokens")
    )
    cache_creation_tokens = _int_value(
        _field_value(usage, "cache_creation_input_tokens")
    )

    for detail_name in ("prompt_tokens_details", "input_tokens_details"):
        details = _field_value(usage, detail_name)
        if details is not None and cache_read_tokens is None:
            cache_read_tokens = _int_value(
                _field_value(details, "cached_tokens")
            )

    values: Dict[str, int] = {}
    if input_tokens is not None:
        values["input_tokens"] = input_tokens
    if output_tokens is not None:
        values["output_tokens"] = output_tokens
    if cache_read_tokens is not None and cache_read_tokens > 0:
        values["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None and cache_creation_tokens > 0:
        values["cache_creation_input_tokens"] = cache_creation_tokens

    return values


def _usage_score(usage_values: Dict[str, int]) -> int:
    return (usage_values.get("input_tokens") or 0) + (
        usage_values.get("output_tokens") or 0
    )


def _usage_sources(value: Any) -> List[Any]:
    sources = []
    usage = _field_value(value, "usage")
    if usage is not None:
        sources.append(usage)

    extra = _field_value(value, "extra")
    if extra is not None:
        extra_usage = _field_value(extra, "usage", "usage_metadata")
        if extra_usage is not None:
            sources.append(extra_usage)

        service_info = _field_value(extra, "model_service_info")
        if service_info is not None:
            sources.append(service_info)

    service_info = _field_value(value, "model_service_info")
    if service_info is not None:
        sources.append(service_info)

    return sources


def _extract_usage_values(value: Any, depth: int = 0) -> Dict[str, int]:
    """Extract token usage from qwen-agent Message/extra/model_service_info."""
    if value is None or depth > 4:
        return {}

    best_values: Dict[str, int] = {}
    values = _usage_token_values(value)
    if values:
        best_values = values

    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            item_values = _extract_usage_values(item, depth + 1)
            if _usage_score(item_values) > _usage_score(best_values):
                best_values = item_values
        return best_values

    for source in _usage_sources(value):
        source_values = _extract_usage_values(source, depth + 1)
        if _usage_score(source_values) > _usage_score(best_values):
            best_values = source_values

    return best_values


def _apply_usage_to_llm_invocation(
    invocation: LLMInvocation, value: Any
) -> None:
    """Apply qwen-agent token usage metadata to an LLMInvocation.

    Qwen-Agent stores DashScope responses under Message.extra["model_service_info"]
    for both streaming and non-streaming calls. Streaming chunks can carry
    cumulative usage, so only replace existing values when the candidate usage
    has at least as many observed tokens as the current invocation.
    """
    usage_values = _extract_usage_values(value)
    if not usage_values:
        return

    current_score = (invocation.input_tokens or 0) + (
        invocation.output_tokens or 0
    )
    if current_score and _usage_score(usage_values) < current_score:
        return

    if "input_tokens" in usage_values:
        invocation.input_tokens = usage_values["input_tokens"]
    if "output_tokens" in usage_values:
        invocation.output_tokens = usage_values["output_tokens"]
    if "cache_read_input_tokens" in usage_values:
        invocation.usage_cache_read_input_tokens = usage_values[
            "cache_read_input_tokens"
        ]
    if "cache_creation_input_tokens" in usage_values:
        invocation.usage_cache_creation_input_tokens = usage_values[
            "cache_creation_input_tokens"
        ]


def apply_token_usage_from_qwen_messages(
    invocation: LLMInvocation,
    messages: Any,
) -> None:
    """Populate token usage from qwen-agent Message metadata.

    Kept as a compatibility entrypoint for callers that used the previous
    helper name; the instrumentation wrapper now calls the generic extractor
    directly so it can process individual streaming chunks.
    """
    _apply_usage_to_llm_invocation(invocation, messages)


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


def _convert_qwen_agent_final_output_messages(
    messages: Any,
) -> List[OutputMessage]:
    """Convert only the final qwen-agent answer to GenAI OutputMessage format."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    for msg in reversed(messages):
        try:
            role = _field_value(msg, "role") or "assistant"
            function_call = _field_value(msg, "function_call")
            content = _field_value(msg, "content") or ""

            if role in ("function", "tool") or function_call:
                continue

            text = _extract_content_text(content)
            if text:
                return _convert_qwen_messages_to_output_messages([msg])
        except Exception as e:
            logger.debug(f"Error extracting final agent output message: {e}")
            continue

    logger.debug("No final qwen-agent assistant text output message found")
    return []


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
        stream=True if stream else None,
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
