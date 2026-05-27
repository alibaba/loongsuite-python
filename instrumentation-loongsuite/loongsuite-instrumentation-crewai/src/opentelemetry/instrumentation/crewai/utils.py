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

import dataclasses
import logging
from typing import Any, Dict, List, Optional

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import Span
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCall,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
)

logger = logging.getLogger(__name__)

OP_NAME_CREW = "crew.kickoff"
OP_NAME_AGENT = "agent.execute"
OP_NAME_TASK = "task.execute"
OP_NAME_TOOL = "tool.execute"


class GenAIHookHelper:
    def __init__(self, capture_content: bool = True):
        self.capture_content = capture_content

    def on_completion(
        self,
        span: Span,
        inputs: List[InputMessage],
        outputs: List[OutputMessage],
        system_instructions: Optional[List[MessagePart]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        if not self.capture_content or not span.is_recording():
            return

        if not is_experimental_mode():
            return

        capturing_mode = get_content_capturing_mode()
        should_capture_span = capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )

        if should_capture_span:
            if inputs:
                span.set_attribute(
                    gen_ai_attributes.GEN_AI_INPUT_MESSAGES,
                    gen_ai_json_dumps([dataclasses.asdict(i) for i in inputs]),
                )
            if outputs:
                span.set_attribute(
                    gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES,
                    gen_ai_json_dumps(
                        [dataclasses.asdict(o) for o in outputs]
                    ),
                )
            if system_instructions:
                span.set_attribute(
                    gen_ai_attributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                    gen_ai_json_dumps(
                        [dataclasses.asdict(i) for i in system_instructions]
                    ),
                )

        if attributes:
            span.set_attributes(attributes)


def to_input_message(role: str, content: Any) -> List[InputMessage]:
    if not content:
        return []

    text_content = content if isinstance(content, str) else str(content)

    return [InputMessage(role=role, parts=[Text(content=text_content)])]


def to_output_message(
    role: str, content: Any, finish_reason: str = "stop"
) -> List[OutputMessage]:
    if not content:
        return []

    text_content = content if isinstance(content, str) else str(content)

    return [
        OutputMessage(
            role=role,
            parts=[Text(content=text_content)],
            finish_reason=finish_reason,
        )
    ]


def extract_agent_inputs(
    task_obj: Any, context: str, tools: List[Any]
) -> List[InputMessage]:
    description = getattr(task_obj, "description", "")

    parts = []
    if description:
        parts.append(Text(content=f"Task: {description}"))
    if context:
        parts.append(Text(content=f"Context: {context}"))
    if tools:
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        parts.append(Text(content=f"Tools Available: {', '.join(tool_names)}"))

    return [InputMessage(role="user", parts=parts)]


def extract_tool_inputs(tool_name: str, arguments: Any) -> List[InputMessage]:
    args_str = (
        gen_ai_json_dumps(arguments)
        if isinstance(arguments, dict)
        else str(arguments)
    )

    return [
        InputMessage(
            role="assistant",
            parts=[ToolCall(id=tool_name, name=tool_name, arguments=args_str)],
        )
    ]
