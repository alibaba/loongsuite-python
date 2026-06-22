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

"""Utility functions for VitaBench instrumentation.

Handles conversion between vitabench Message types and
OpenTelemetry GenAI semantic convention types.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolCallResponse,
)
from opentelemetry.util.genai.types import (
    ToolCall as OTelToolCall,
)

logger = logging.getLogger(__name__)

_MAX_CONTENT_LEN = 4096


def _convert_vita_messages_to_input(messages: Any) -> List[InputMessage]:
    """Convert vita Message list to OTel InputMessage list."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    result = []
    for msg in messages:
        try:
            role = getattr(msg, "role", None)
            if role is None:
                continue

            parts = []
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)

            if role == "tool":
                msg_id = getattr(msg, "id", None) or ""
                if content:
                    parts.append(
                        ToolCallResponse(
                            id=msg_id,
                            response=str(content)[:_MAX_CONTENT_LEN],
                        )
                    )
            else:
                if content:
                    parts.append(Text(content=str(content)[:_MAX_CONTENT_LEN]))
                if tool_calls:
                    for tc in tool_calls:
                        tc_args = getattr(tc, "arguments", {})
                        if isinstance(tc_args, dict):
                            tc_args = json.dumps(
                                tc_args, ensure_ascii=False, default=str
                            )
                        parts.append(
                            OTelToolCall(
                                name=getattr(tc, "name", ""),
                                id=getattr(tc, "id", None),
                                arguments=tc_args,
                            )
                        )

            if parts:
                result.append(InputMessage(role=role, parts=parts))
        except Exception as e:
            logger.debug(f"Error converting vita message: {e}")
            continue

    return result


def _convert_vita_assistant_to_output(msg: Any) -> List[OutputMessage]:
    """Convert vita AssistantMessage to OTel OutputMessage list."""
    if not msg:
        return []

    parts = []
    content = getattr(msg, "content", None)
    tool_calls = getattr(msg, "tool_calls", None)

    if content:
        parts.append(Text(content=str(content)[:_MAX_CONTENT_LEN]))
    if tool_calls:
        for tc in tool_calls:
            tc_args = getattr(tc, "arguments", {})
            if isinstance(tc_args, dict):
                tc_args = json.dumps(tc_args, ensure_ascii=False, default=str)
            parts.append(
                OTelToolCall(
                    name=getattr(tc, "name", ""),
                    id=getattr(tc, "id", None),
                    arguments=tc_args,
                )
            )

    finish_reason = "tool_calls" if tool_calls else "stop"

    if not parts:
        parts.append(Text(content=""))

    return [
        OutputMessage(
            role="assistant", parts=parts, finish_reason=finish_reason
        )
    ]


def _infer_provider(model_name: str) -> str:
    """Infer provider from model name string."""
    if not model_name:
        return "unknown"
    m = model_name.lower()
    if "gpt" in m or "o1" in m or "o3" in m:
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "qwen" in m:
        return "alibaba_cloud"
    if "deepseek" in m:
        return "deepseek"
    if "gemini" in m:
        return "google"
    return "unknown"


def _get_tool_definitions(
    tools: Any,
) -> Optional[List[FunctionToolDefinition]]:
    """Extract tool definitions from vita Tool list."""
    if not tools:
        return None

    try:
        defs = []
        for t in tools:
            name = getattr(t, "name", None)
            if not name:
                continue
            parameters = None
            openai_schema = getattr(t, "openai_schema", None)
            if isinstance(openai_schema, dict):
                function_schema = openai_schema.get("function", openai_schema)
                parameters = function_schema.get("parameters")
            defs.append(
                FunctionToolDefinition(
                    name=name,
                    description=getattr(t, "short_desc", None),
                    parameters=parameters,
                )
            )
        return defs if defs else None
    except Exception:
        return None
