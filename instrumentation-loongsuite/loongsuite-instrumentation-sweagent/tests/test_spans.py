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

from __future__ import annotations

import json
from unittest.mock import MagicMock

from sweagent.agent.hooks.abstract import CombinedAgentHook
from sweagent.run.hooks.abstract import CombinedRunHooks
from sweagent.types import AgentInfo, AgentRunResult, StepOutput

from opentelemetry.instrumentation.sweagent.patch import (
    SWEAGENT_AGENT_NAME,
    SWEAGENT_BASH_TOOL_NAME,
    tool_call_arguments_from_sweagent_step,
    tool_name_from_sweagent_step,
    wrap_default_agent_handle_action,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)

ENTRY_SPAN_NAME = "enter_ai_application_system"
REACT_SPAN_NAME = "react step"
TOOL_SPAN_PREFIX = "execute_tool "
INVOKE_AGENT_SPAN_NAME = f"{GenAI.GenAiOperationNameValues.INVOKE_AGENT.value} {SWEAGENT_AGENT_NAME}"


def test_tool_name_from_step_llm_tool_calls():
    openai_style = {
        "type": "function",
        "id": "call_1",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
    step = StepOutput(
        action="ls",
        tool_calls=[openai_style],
    )
    assert tool_name_from_sweagent_step(step) == "bash"

    step2 = StepOutput(
        action="submit",
        tool_calls=[
            {
                "type": "function",
                "id": "call_s",
                "function": {"name": "submit", "arguments": "{}"},
            }
        ],
    )
    assert tool_name_from_sweagent_step(step2) == "submit"


def test_tool_name_from_step_fallback_without_tool_calls():
    assert tool_name_from_sweagent_step(None) == SWEAGENT_BASH_TOOL_NAME
    assert (
        tool_name_from_sweagent_step(StepOutput(action="ls -la"))
        == SWEAGENT_BASH_TOOL_NAME
    )


def test_tool_call_arguments_from_step_llm_json():
    step = StepOutput(
        action="ls  # rendered for bash",
        tool_calls=[
            {
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "ls -la"}',
                }
            }
        ],
    )
    assert tool_call_arguments_from_sweagent_step(step) == {
        "command": "ls -la"
    }

    empty_args = StepOutput(
        action="touch x",
        tool_calls=[{"function": {"name": "bash", "arguments": "{}"}}],
    )
    assert tool_call_arguments_from_sweagent_step(empty_args) == {}


def test_tool_call_arguments_from_step_non_json_string_kept():
    step = StepOutput(
        action="fallback_action",
        tool_calls=[
            {"function": {"name": "bash", "arguments": "not valid json {"}}
        ],
    )
    assert tool_call_arguments_from_sweagent_step(step) == "not valid json {"


def test_tool_call_arguments_fallback_without_tool_calls():
    assert tool_call_arguments_from_sweagent_step(None) is None
    assert (
        tool_call_arguments_from_sweagent_step(StepOutput(action="ls -la"))
        == "ls -la"
    )


def _get_attrs(span):
    return dict(span.attributes or {})


def test_entry_run_hooks_span(instrumented_sweagent, span_exporter):
    hooks = CombinedRunHooks()
    prob = MagicMock()
    prob.id = "issue-42"
    prob.get_problem_statement.return_value = "Fix the crash"

    hooks.on_instance_start(index=0, env=MagicMock(), problem_statement=prob)
    result = AgentRunResult(
        info=AgentInfo(exit_status="Submitted"), trajectory=[]
    )
    hooks.on_instance_completed(result=result)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == ENTRY_SPAN_NAME
    attrs = _get_attrs(span)
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "enter"
    assert attrs.get(GEN_AI_SPAN_KIND) == GenAiSpanKindValues.ENTRY.value
    assert attrs.get(GEN_AI_SESSION_ID) == "issue-42"


def test_react_step_span(instrumented_sweagent, span_exporter):
    hooks = CombinedAgentHook()
    hooks.on_step_start()
    step = StepOutput(done=False, exit_status=None)
    hooks.on_step_done(step=step, info=AgentInfo())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == REACT_SPAN_NAME
    attrs = _get_attrs(span)
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "react"
    assert attrs.get(GEN_AI_SPAN_KIND) == GenAiSpanKindValues.STEP.value
    assert attrs.get("gen_ai.react.round") == 1


def test_invoke_agent_span(instrumented_sweagent, span_exporter):
    hooks = CombinedAgentHook()
    hooks.on_run_start()
    hooks.on_run_done(trajectory=[], info=AgentInfo())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == INVOKE_AGENT_SPAN_NAME
    attrs = _get_attrs(span)
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    assert attrs.get(GenAI.GEN_AI_AGENT_NAME) == SWEAGENT_AGENT_NAME
    assert attrs.get(GEN_AI_SPAN_KIND) == GenAiSpanKindValues.AGENT.value
    in_raw = attrs.get(GenAI.GEN_AI_INPUT_MESSAGES)
    out_raw = attrs.get(GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert in_raw is not None and out_raw is not None
    in_msgs = json.loads(in_raw)
    assert in_msgs[0]["parts"][0]["content"] == "(empty)"
    out_msgs = json.loads(out_raw)
    assert out_msgs[0]["role"] == "assistant"


def test_handle_action_execute_tool_span(tracer_provider, span_exporter):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    step = StepOutput(
        action="ls -la",
        observation="",
        tool_calls=[
            {
                "type": "function",
                "id": "call_abc",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ],
    )

    def fake_handle_action(*args, **kwargs):
        step.observation = "file.txt"
        return step

    wrap_default_agent_handle_action(
        handler, fake_handle_action, MagicMock(), (step,), {}
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == f"{TOOL_SPAN_PREFIX}bash"
    attrs = _get_attrs(span)
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "execute_tool"
    assert attrs.get(GenAI.GEN_AI_TOOL_NAME) == "bash"
    assert attrs.get(GEN_AI_SPAN_KIND) == GenAiSpanKindValues.TOOL.value


def test_nested_hook_hierarchy(instrumented_sweagent, span_exporter):
    handler = instrumented_sweagent._handler
    run_hooks = CombinedRunHooks()
    agent_hooks = CombinedAgentHook()
    prob = MagicMock()
    prob.id = "nested-1"
    prob.get_problem_statement.return_value = "task"

    run_hooks.on_instance_start(
        index=0, env=MagicMock(), problem_statement=prob
    )
    agent_hooks.on_run_start()
    agent_hooks.on_step_start()
    step = StepOutput(action="true", observation="")

    def fake_handle_action(*args, **kwargs):
        step.observation = "ok"
        return step

    wrap_default_agent_handle_action(
        handler, fake_handle_action, MagicMock(), (step,), {}
    )

    agent_hooks.on_step_done(step=step, info=AgentInfo())
    agent_hooks.on_run_done(trajectory=[], info=AgentInfo())
    result = AgentRunResult(info=AgentInfo(exit_status="ok"), trajectory=[])
    run_hooks.on_instance_completed(result=result)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    by_name = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)

    assert ENTRY_SPAN_NAME in by_name
    assert INVOKE_AGENT_SPAN_NAME in by_name
    assert REACT_SPAN_NAME in by_name
    tool_name = f"execute_tool {SWEAGENT_BASH_TOOL_NAME}"
    assert tool_name in by_name

    entry_span = by_name[ENTRY_SPAN_NAME][0]
    invoke_span = by_name[INVOKE_AGENT_SPAN_NAME][0]
    react_span = by_name[REACT_SPAN_NAME][0]
    tool_span = by_name[tool_name][0]

    assert invoke_span.parent.span_id == entry_span.context.span_id
    assert react_span.parent.span_id == invoke_span.context.span_id
    assert tool_span.parent.span_id == react_span.context.span_id

    assert (
        invoke_span.attributes.get(GenAI.GEN_AI_CONVERSATION_ID) == "nested-1"
    )
    in_msgs = json.loads(invoke_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert in_msgs[0]["parts"][0]["content"] == "task"
