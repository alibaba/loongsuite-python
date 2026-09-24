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

"""Helpers for building util-genai invocations from CrewAI objects."""

from __future__ import annotations

import dataclasses
from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import Span
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCall,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
)

CREWAI_PROVIDER = "crewai"
CREWAI_OPERATION = "gen_ai.crewai.operation"
CREWAI_COMPONENT = "gen_ai.crewai.component"
CREWAI_TASK_DESCRIPTION = "gen_ai.crewai.task.description"
CREWAI_TASK_EXPECTED_OUTPUT = "gen_ai.crewai.task.expected_output"
CREWAI_AGENT_GOAL = "gen_ai.crewai.agent.goal"
CREWAI_AGENT_BACKSTORY = "gen_ai.crewai.agent.backstory"
CREWAI_PROCESS = "gen_ai.crewai.process"
CREWAI_TOOLS_COUNT = "gen_ai.crewai.tools.count"
CREWAI_USAGE_SUCCESSFUL_REQUESTS = "gen_ai.crewai.usage.successful_requests"
CREWAI_INPUT_METADATA_KEYS = {
    "conversation_id",
    "session_id",
    "streaming",
    "thread_id",
    "user_id",
}

OP_NAME_CREW = "crew.kickoff"
OP_NAME_FLOW = "flow.kickoff"
OP_NAME_AGENT = "agent.execute"
OP_NAME_TASK = "task.execute"
OP_NAME_TOOL = "tool.execute"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "result"):
        try:
            return _stringify(value.result)
        except Exception:
            pass
    raw = getattr(value, "raw", None)
    if raw is not None:
        return str(raw)
    return str(value)


def _non_empty_string(value: Any) -> str | None:
    text = _stringify(value).strip()
    return text or None


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field_value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        try:
            members = vars(value)
        except TypeError:
            members = {}
        if name in members:
            return members[name]
        if name not in dir(value):
            continue
        try:
            return getattr(value, name)
        except AttributeError:
            pass
    return None


def _should_capture_content() -> bool:
    if not is_experimental_mode():
        return False
    try:
        return get_content_capturing_mode() in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    except ValueError:
        return False


def _usage_metrics(value: Any) -> Any:
    if value is None:
        return None

    direct = _field_value(value, "token_usage", "usage_metrics", "usage")
    if direct is not None and direct is not value:
        return _usage_metrics(direct)

    for method_name in ("get_token_usage_summary", "get_summary"):
        method = _field_value(value, method_name)
        if method is None:
            continue
        try:
            summary = method()
        except Exception:
            continue
        if summary is not None and summary is not value:
            return _usage_metrics(summary)

    has_usage_fields = any(
        _field_value(value, name) is not None
        for name in (
            "prompt_tokens",
            "input_tokens",
            "completion_tokens",
            "output_tokens",
            "total_tokens",
        )
    )
    return value if has_usage_fields else None


def _usage_token_values_from_values(*values: Any) -> dict[str, int]:
    first_observed: dict[str, int] = {}
    for value in values:
        usage_values = _usage_token_values(_usage_metrics(value))
        if not usage_values:
            continue
        if not first_observed:
            first_observed = usage_values
        if (usage_values.get("input_tokens") or 0) + (
            usage_values.get("output_tokens") or 0
        ) > 0:
            return usage_values
    return first_observed


def _usage_token_values(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}

    values: dict[str, int] = {}

    input_tokens = _int_value(
        _field_value(usage, "prompt_tokens", "input_tokens")
    )
    output_tokens = _int_value(
        _field_value(usage, "completion_tokens", "output_tokens")
    )
    cache_read_tokens = _int_value(
        _field_value(usage, "cached_prompt_tokens", "cache_read_input_tokens")
    )
    cache_creation_tokens = _int_value(
        _field_value(
            usage, "cache_creation_tokens", "cache_creation_input_tokens"
        )
    )
    successful_requests = _int_value(
        _field_value(usage, "successful_requests")
    )
    total_tokens = _int_value(_field_value(usage, "total_tokens"))

    prompt_details = _field_value(usage, "prompt_tokens_details")
    if cache_read_tokens is None and prompt_details is not None:
        cache_read_tokens = _int_value(
            _field_value(prompt_details, "cached_tokens")
        )

    if input_tokens is not None:
        values["input_tokens"] = input_tokens
    if output_tokens is not None:
        values["output_tokens"] = output_tokens
    if total_tokens is not None:
        values["total_tokens"] = total_tokens
    if cache_read_tokens is not None and cache_read_tokens > 0:
        values["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None and cache_creation_tokens > 0:
        values["cache_creation_input_tokens"] = cache_creation_tokens
    if successful_requests is not None and successful_requests > 0:
        values["successful_requests"] = successful_requests

    return values


def apply_usage_metrics(invocation: Any, *values: Any) -> None:
    usage_values = _usage_token_values_from_values(*values)
    if not usage_values:
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
    if "successful_requests" in usage_values:
        invocation.attributes[CREWAI_USAGE_SUCCESSFUL_REQUESTS] = usage_values[
            "successful_requests"
        ]


def usage_metric_attributes(*values: Any) -> dict[str, int]:
    usage_values = _usage_token_values_from_values(*values)
    if not usage_values:
        return {}

    attributes: dict[str, int] = {}
    input_tokens = usage_values.get("input_tokens")
    output_tokens = usage_values.get("output_tokens")

    if input_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
    if input_tokens is not None or output_tokens is not None:
        attributes["gen_ai.usage.total_tokens"] = usage_values.get(
            "total_tokens",
            (input_tokens or 0) + (output_tokens or 0),
        )
    if "cache_read_input_tokens" in usage_values:
        attributes["gen_ai.usage.cache_read.input_tokens"] = usage_values[
            "cache_read_input_tokens"
        ]
    if "cache_creation_input_tokens" in usage_values:
        attributes["gen_ai.usage.cache_creation.input_tokens"] = usage_values[
            "cache_creation_input_tokens"
        ]
    if "successful_requests" in usage_values:
        attributes[CREWAI_USAGE_SUCCESSFUL_REQUESTS] = usage_values[
            "successful_requests"
        ]
    return attributes


def _message_parts(*values: Any) -> list[MessagePart]:
    parts: list[MessagePart] = []
    for value in values:
        text = _non_empty_string(value)
        if text:
            parts.append(Text(content=text))
    return parts


def to_input_messages(role: str, content: Any) -> list[InputMessage]:
    parts = _message_parts(content)
    if not parts:
        return []
    return [InputMessage(role=role, parts=parts)]


def to_output_messages(
    role: str, content: Any, finish_reason: str = "stop"
) -> list[OutputMessage]:
    parts = _message_parts(content)
    if not parts:
        return []
    return [
        OutputMessage(
            role=role,
            parts=parts,
            finish_reason=finish_reason,
        )
    ]


def _tool_parameters(tool: Any) -> Any:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return None
    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()
    if hasattr(args_schema, "schema"):
        return args_schema.schema()
    return None


def tool_definitions(
    tools: list[Any] | tuple[Any, ...] | None,
) -> list[ToolDefinition]:
    if not tools:
        return []

    definitions: list[ToolDefinition] = []
    for tool in tools:
        name = _non_empty_string(getattr(tool, "name", None))
        if not name:
            continue
        definitions.append(
            FunctionToolDefinition(
                name=name,
                description=_non_empty_string(
                    getattr(tool, "description", None)
                ),
                parameters=_tool_parameters(tool),
            )
        )
    return definitions


def _agent_model(agent: Any) -> str | None:
    llm = getattr(agent, "llm", None)
    if llm is None:
        return None
    for attr in ("model", "model_name", "deployment_name"):
        value = _non_empty_string(getattr(llm, attr, None))
        if value:
            return value
    return _non_empty_string(llm)


def _agent_id(agent: Any) -> str | None:
    return _non_empty_string(
        getattr(agent, "key", None) or getattr(agent, "id", None)
    )


def _crew_process(crew: Any) -> str | None:
    process = getattr(crew, "process", None)
    value = getattr(process, "value", process)
    return _non_empty_string(value)


def _crew_name(crew: Any, default: str) -> str:
    return _non_empty_string(getattr(crew, "name", None)) or default


def _task_description(task: Any) -> str | None:
    return _non_empty_string(getattr(task, "description", None))


def _task_expected_output(task: Any) -> str | None:
    return _non_empty_string(getattr(task, "expected_output", None))


def _task_agent(task: Any) -> Any:
    return getattr(task, "agent", None)


def _session_id_from_inputs(inputs: Any) -> str | None:
    if isinstance(inputs, dict):
        return _non_empty_string(
            inputs.get("session_id")
            or inputs.get("conversation_id")
            or inputs.get("thread_id")
        )
    return None


def _user_id_from_inputs(inputs: Any) -> str | None:
    if isinstance(inputs, dict):
        return _non_empty_string(inputs.get("user_id"))
    return None


def _input_content(inputs: Any) -> Any:
    if not isinstance(inputs, dict):
        return inputs
    return {
        key: value
        for key, value in inputs.items()
        if key not in CREWAI_INPUT_METADATA_KEYS
    }


def create_entry_invocation(
    instance: Any,
    inputs: Any,
    operation_name: str,
) -> EntryInvocation:
    attributes: dict[str, Any] = {
        CREWAI_OPERATION: operation_name,
        CREWAI_COMPONENT: "crew" if operation_name == OP_NAME_CREW else "flow",
    }

    process = _crew_process(instance)
    if process:
        attributes[CREWAI_PROCESS] = process

    tasks = getattr(instance, "tasks", None)
    if tasks is not None:
        try:
            attributes["gen_ai.crewai.tasks.count"] = len(tasks)
        except TypeError:
            pass

    agents = getattr(instance, "agents", None)
    if agents is not None:
        try:
            attributes["gen_ai.crewai.agents.count"] = len(agents)
        except TypeError:
            pass

    attributes[GenAI.GEN_AI_PROVIDER_NAME] = CREWAI_PROVIDER
    attributes[GenAI.GEN_AI_AGENT_NAME] = _crew_name(instance, operation_name)

    return EntryInvocation(
        session_id=_session_id_from_inputs(inputs),
        user_id=_user_id_from_inputs(inputs),
        input_messages=(
            to_input_messages("user", _input_content(inputs))
            if _should_capture_content()
            else []
        ),
        attributes=attributes,
    )


def create_task_invocation(
    task: Any, agent: Any = None
) -> InvokeAgentInvocation:
    description = _task_description(task)
    expected_output = _task_expected_output(task)
    agent = agent or _task_agent(task)
    agent_name = _non_empty_string(getattr(agent, "role", None))
    capture_content = _should_capture_content()

    attributes: dict[str, Any] = {
        CREWAI_OPERATION: OP_NAME_TASK,
        CREWAI_COMPONENT: "task",
    }
    if description and capture_content:
        attributes[CREWAI_TASK_DESCRIPTION] = description
    if expected_output and capture_content:
        attributes[CREWAI_TASK_EXPECTED_OUTPUT] = expected_output
    if agent_name:
        attributes["gen_ai.crewai.task.agent"] = agent_name

    input_parts = (
        _message_parts(
            f"Task: {description}" if description else None,
            f"Expected output: {expected_output}" if expected_output else None,
        )
        if capture_content
        else []
    )

    return InvokeAgentInvocation(
        provider=CREWAI_PROVIDER,
        agent_id=_agent_id(agent),
        agent_name=agent_name or "task",
        agent_description=(
            (description or expected_output) if capture_content else None
        ),
        request_model=_agent_model(agent),
        response_model_name=_agent_model(agent),
        input_messages=(
            [InputMessage(role="user", parts=input_parts)]
            if input_parts
            else []
        ),
        attributes=attributes,
    )


def create_agent_invocation(
    agent: Any,
    task: Any,
    context: Any,
    tools: list[Any] | tuple[Any, ...] | None,
) -> InvokeAgentInvocation:
    role = _non_empty_string(getattr(agent, "role", None)) or "agent"
    goal = _non_empty_string(getattr(agent, "goal", None))
    backstory = _non_empty_string(getattr(agent, "backstory", None))
    description = _task_description(task)
    capture_content = _should_capture_content()

    attributes: dict[str, Any] = {
        CREWAI_OPERATION: OP_NAME_AGENT,
        CREWAI_COMPONENT: "agent",
    }
    if goal and capture_content:
        attributes[CREWAI_AGENT_GOAL] = goal
    if backstory and capture_content:
        attributes[CREWAI_AGENT_BACKSTORY] = backstory
    if tools:
        attributes[CREWAI_TOOLS_COUNT] = len(tools)

    input_parts = (
        _message_parts(
            f"Task: {description}" if description else None,
            f"Context: {context}" if _non_empty_string(context) else None,
        )
        if capture_content
        else []
    )

    return InvokeAgentInvocation(
        provider=CREWAI_PROVIDER,
        agent_id=_agent_id(agent),
        agent_name=role,
        agent_description=goal if capture_content else None,
        request_model=_agent_model(agent),
        response_model_name=_agent_model(agent),
        input_messages=(
            [InputMessage(role="user", parts=input_parts)]
            if input_parts
            else []
        ),
        system_instruction=(
            _message_parts(backstory) if capture_content else None
        ),
        tool_definitions=tool_definitions(tools),
        attributes=attributes,
    )


def _tool_call_id(tool_name: str, tool_calling: Any) -> str:
    if isinstance(tool_calling, dict):
        return (
            _non_empty_string(tool_calling.get("id"))
            or _non_empty_string(tool_calling.get("tool_call_id"))
            or tool_name
        )
    return (
        _non_empty_string(getattr(tool_calling, "id", None))
        or _non_empty_string(getattr(tool_calling, "tool_call_id", None))
        or tool_name
    )


def _tool_arguments(tool_calling: Any) -> Any:
    if tool_calling is None:
        return None
    if isinstance(tool_calling, dict):
        return tool_calling.get("arguments") or tool_calling.get("tool_input")
    return getattr(tool_calling, "arguments", None)


def create_tool_invocation(
    tool: Any,
    tool_calling: Any,
    *,
    tool_call_arguments: Any = None,
) -> ExecuteToolInvocation:
    tool_name = (
        _non_empty_string(getattr(tool, "name", None)) or "unknown_tool"
    )
    invocation = ExecuteToolInvocation(
        provider=CREWAI_PROVIDER,
        tool_name=tool_name,
        tool_call_id=_tool_call_id(tool_name, tool_calling),
        tool_description=_non_empty_string(getattr(tool, "description", None)),
        tool_type="function",
        tool_call_arguments=(
            tool_call_arguments
            if tool_call_arguments is not None
            else _tool_arguments(tool_calling)
        ),
        attributes={
            CREWAI_OPERATION: OP_NAME_TOOL,
            CREWAI_COMPONENT: "tool",
        },
    )
    return invocation


class GenAIHookHelper:
    """Backward-compatible helper for older direct-span tests.

    Runtime instrumentation uses ``ExtendedTelemetryHandler``. This shim remains
    for external tests that import it directly from previous CrewAI releases.
    """

    def __init__(self, capture_content: bool = True):
        self.capture_content = capture_content

    def on_completion(
        self,
        span: Span,
        inputs: list[InputMessage],
        outputs: list[OutputMessage],
        system_instructions: list[MessagePart] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
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
                    GenAI.GEN_AI_INPUT_MESSAGES,
                    gen_ai_json_dumps([dataclasses.asdict(i) for i in inputs]),
                )
            if outputs:
                span.set_attribute(
                    GenAI.GEN_AI_OUTPUT_MESSAGES,
                    gen_ai_json_dumps(
                        [dataclasses.asdict(o) for o in outputs]
                    ),
                )
            if system_instructions:
                span.set_attribute(
                    GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
                    gen_ai_json_dumps(
                        [dataclasses.asdict(i) for i in system_instructions]
                    ),
                )

        if attributes:
            span.set_attributes(attributes)


def extract_tool_inputs(tool_name: str, arguments: Any) -> list[InputMessage]:
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
