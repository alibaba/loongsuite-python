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

"""Utility functions for WideSearch instrumentation."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolCallResponse,
)
from opentelemetry.util.genai.types import (
    ToolCall as GenAIToolCall,
)

logger = logging.getLogger(__name__)


_FRAMEWORK = "widesearch"


def _create_entry_invocation(
    query: str,
    *,
    system_prompt: Optional[str] = None,
    tools_desc: Optional[List[dict[str, Any]]] = None,
) -> EntryInvocation:
    invocation = EntryInvocation()
    invocation.input_messages = [
        InputMessage(role="user", parts=[Text(content=query)])
    ]
    invocation.attributes["gen_ai.framework"] = _FRAMEWORK
    if system_prompt:
        invocation.system_instruction = [Text(content=system_prompt)]

    defs = None
    if tools_desc:
        defs = _convert_tools_desc(tools_desc)
        if defs is not None:
            invocation.tool_definitions = defs

    return invocation


def _create_agent_invocation(
    agent: Any, user_input: str, system_prompt: Optional[str] = None
) -> InvokeAgentInvocation:
    agent_name = getattr(agent, "name", None) or "widesearch-agent"

    request_model = None
    model_config_name = getattr(agent, "model_config_name", None)
    if model_config_name:
        try:
            from src.utils.config import model_config

            request_model = model_config.get(model_config_name, {}).get(
                "model_name"
            )
        except Exception:
            pass
    request_model = request_model or model_config_name

    instructions = system_prompt or getattr(agent, "instructions", None) or ""

    invocation = InvokeAgentInvocation(
        provider="widesearch",
        agent_name=agent_name,
        agent_description=instructions[:200] if instructions else "",
        request_model=request_model,
        input_messages=[
            InputMessage(role="user", parts=[Text(content=user_input)])
        ],
    )
    invocation.attributes["gen_ai.framework"] = _FRAMEWORK

    if instructions:
        invocation.system_instruction = [Text(content=instructions)]

    tools_desc = getattr(agent, "tools_desc", None)
    if tools_desc:
        invocation.tool_definitions = _convert_tools_desc(tools_desc)

    return invocation


def _create_tool_invocation(
    tool_call: Any, agent: Any
) -> ExecuteToolInvocation:
    args = tool_call.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {"raw": args}

    description = None
    if hasattr(agent, "tools_desc"):
        for td in agent.tools_desc:
            func = td.get("function", {})
            if func.get("name") == tool_call.tool_name:
                description = func.get("description")
                break

    invocation = ExecuteToolInvocation(
        tool_name=tool_call.tool_name,
        tool_call_id=getattr(tool_call, "tool_call_id", None),
        tool_call_arguments=args,
        tool_description=description,
        tool_type="function",
    )
    invocation.attributes["gen_ai.framework"] = _FRAMEWORK
    return invocation


def _extract_output_messages(messages: Any) -> List[OutputMessage]:
    """Extract output messages from run_single_query return value."""
    if not messages:
        return []
    last_msg = messages[-1]
    content = ""
    if isinstance(last_msg, dict):
        c = last_msg.get("content", {})
        if isinstance(c, dict):
            content = c.get("content", "")
        elif isinstance(c, str):
            content = c
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=content)],
            finish_reason="stop",
        )
    ]


def _step_to_output_messages(step: Any) -> List[OutputMessage]:
    """Extract output messages from an ActionStep."""
    content = getattr(step, "content", None) or ""
    parts = []
    if content:
        parts.append(Text(content=content))

    for tool_call in getattr(step, "tool_calls", []) or []:
        args = getattr(tool_call, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                pass
        parts.append(
            GenAIToolCall(
                id=getattr(tool_call, "tool_call_id", None),
                name=getattr(tool_call, "tool_name", ""),
                arguments=args,
            )
        )

    for tool_result in getattr(step, "tool_call_results", []) or []:
        result = getattr(tool_result, "content", None)
        if result is None and getattr(tool_result, "error_marker", None):
            result = getattr(tool_result, "error_marker", {}).get("message")
        parts.append(
            ToolCallResponse(
                id=getattr(tool_result, "tool_call_id", None),
                response=result,
            )
        )

    finish_reason = (
        "tool_calls" if getattr(step, "tool_calls", None) else "stop"
    )
    return [
        OutputMessage(
            role="assistant",
            parts=parts or [Text(content="")],
            finish_reason=finish_reason,
        )
    ]


def _convert_tools_desc(
    tools_desc: List[dict],
) -> Optional[List[FunctionToolDefinition]]:
    """Convert WideSearch tools_desc to FunctionToolDefinition list."""
    result = []
    for td in tools_desc:
        if td.get("type") == "function":
            func = td.get("function", {})
            result.append(
                FunctionToolDefinition(
                    name=func.get("name", ""),
                    description=func.get("description"),
                    parameters=func.get("parameters"),
                )
            )
    return result if result else None
