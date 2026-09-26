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

"""Tests for Agent span creation — verifying AGENT_RUN_NAMES detection."""

from uuid import uuid4

import pytest

from opentelemetry import baggage, context
from opentelemetry.instrumentation.langchain.internal._agent_semantics import (
    AGENT_FRAMEWORK_METADATA_KEY,
    AGENT_STEP_NODE_METADATA_KEY,
    DEERFLOW_FRAMEWORK,
)
from opentelemetry.instrumentation.langchain.internal._tracer import (
    LoongsuiteTracer,
    _extract_langgraph_input_message,
)
from opentelemetry.instrumentation.langchain.internal._utils import (
    AGENT_RUN_NAMES,
    _is_agent_run,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


class _FakeRun:
    """Minimal stub that looks like a langchain Run for unit tests."""

    def __init__(
        self,
        name: str,
        parent_run_id=None,
        inputs=None,
        outputs=None,
        extra=None,
    ):
        self.id = uuid4()
        self.name = name
        self.parent_run_id = parent_run_id
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.extra = extra or {}
        self.metadata = {}
        self.serialized = {}
        self.error = None


class TestAgentDetection:
    def test_agent_executor_detected(self):
        assert _is_agent_run(_FakeRun("AgentExecutor"))

    def test_mrkl_chain_detected(self):
        assert _is_agent_run(_FakeRun("MRKLChain"))

    def test_react_chain_detected(self):
        assert _is_agent_run(_FakeRun("ReActChain"))

    def test_self_ask_chain_detected(self):
        assert _is_agent_run(_FakeRun("SelfAskWithSearchChain"))

    def test_regular_chain_not_detected(self):
        assert not _is_agent_run(_FakeRun("RunnableSequence"))

    def test_empty_name_not_detected(self):
        assert not _is_agent_run(_FakeRun(""))

    def test_none_name_not_detected(self):
        assert not _is_agent_run(_FakeRun(None))

    def test_agent_run_names_immutable(self):
        assert isinstance(AGENT_RUN_NAMES, frozenset)


def test_extract_langgraph_input_message_from_openai_style_dict():
    message = _extract_langgraph_input_message(
        {"role": "user", "content": "hello"}
    )

    assert message is not None
    assert message.role == "user"
    assert len(message.parts) == 1
    assert message.parts[0].type == "text"
    assert message.parts[0].content == "hello"


def test_extract_langgraph_input_message_keeps_empty_content_tool_call():
    message = _extract_langgraph_input_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query":"hello"}',
                    },
                }
            ],
        }
    )

    assert message is not None
    assert message.role == "assistant"
    assert len(message.parts) == 1
    assert message.parts[0].type == "tool_call"
    assert message.parts[0].name == "search"
    assert message.parts[0].arguments == '{"query":"hello"}'
    assert message.parts[0].id == "call_1"


def test_agent_context_colors_child_llm_and_tool_spans(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(
        handler=handler,
        tracer_provider=tracer_provider,
    )

    agent_run = _FakeRun(
        "AgentExecutor",
        inputs={"input": "plan a search"},
    )
    tracer._start_agent(agent_run)

    llm_run = _FakeRun(
        "ChatOpenAI",
        parent_run_id=agent_run.id,
        inputs={"prompts": ["plan a search"]},
        extra={"invocation_params": {"model_name": "gpt-4o-mini"}},
    )
    tracer._handle_llm_start(llm_run)
    llm_run.outputs = {"generations": [[{"text": "call search"}]]}
    tracer._on_llm_end(llm_run)

    tool_run = _FakeRun(
        "search",
        parent_run_id=agent_run.id,
        inputs={"input": "query"},
        outputs={"output": "result"},
    )
    tracer._on_tool_start(tool_run)
    tracer._on_tool_end(tool_run)

    agent_run.outputs = {"output": "done"}
    tracer._on_chain_end(agent_run)

    spans = span_exporter.get_finished_spans()
    llm_span = next(span for span in spans if span.name == "chat gpt-4o-mini")
    tool_span = next(
        span for span in spans if span.name == "execute_tool search"
    )

    assert llm_span.attributes[GenAI.GEN_AI_AGENT_NAME] == "AgentExecutor"
    assert tool_span.attributes[GenAI.GEN_AI_AGENT_NAME] == "AgentExecutor"


def test_deerflow_agent_semantics_sets_framework_attribute(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(
        handler=handler,
        tracer_provider=tracer_provider,
    )
    agent_run = _FakeRun("lead-agent", inputs={"input": "research"})
    agent_run.metadata = {
        AGENT_FRAMEWORK_METADATA_KEY: DEERFLOW_FRAMEWORK,
        AGENT_STEP_NODE_METADATA_KEY: "model",
    }

    tracer._start_agent(agent_run)
    agent_run.outputs = {"output": "done"}
    tracer._on_chain_end(agent_run)

    agent_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.span.kind") == "AGENT"
    )
    assert agent_span.attributes["gen_ai.framework"] == "deerflow"


@pytest.mark.parametrize(
    ("name", "metadata", "tags", "expected"),
    [
        ("research-agent", {}, [], "research-agent"),
        (
            "LangGraph",
            {"agent_name": "configured-agent"},
            [],
            "configured-agent",
        ),
        (
            "LangGraph",
            {"langfuse_trace_name": "metadata-agent"},
            ["subagent:researcher"],
            "subagent:researcher",
        ),
        (
            "LangGraph",
            {"langfuse_trace_name": "metadata-agent"},
            [],
            "metadata-agent",
        ),
        ("default", {}, [], "lead-agent"),
    ],
)
def test_deerflow_agent_name_resolution(
    name,
    metadata,
    tags,
    expected,
    tracer_provider,
):
    tracer = LoongsuiteTracer(
        handler=ExtendedTelemetryHandler(tracer_provider=tracer_provider),
        tracer_provider=tracer_provider,
    )
    run = _FakeRun(name)
    run.metadata = {
        AGENT_FRAMEWORK_METADATA_KEY: DEERFLOW_FRAMEWORK,
        AGENT_STEP_NODE_METADATA_KEY: "model",
        **metadata,
    }
    run.tags = tags

    assert tracer._resolve_agent_name(run) == expected


def test_deerflow_agent_name_falls_back_to_entry_baggage(tracer_provider):
    tracer = LoongsuiteTracer(
        handler=ExtendedTelemetryHandler(tracer_provider=tracer_provider),
        tracer_provider=tracer_provider,
    )
    run = _FakeRun("LangGraph")
    run.metadata = {
        AGENT_FRAMEWORK_METADATA_KEY: DEERFLOW_FRAMEWORK,
        AGENT_STEP_NODE_METADATA_KEY: "model",
    }
    run.tags = []
    ctx = baggage.set_baggage("gen_ai.agent.name", "embedded-agent")
    token = context.attach(ctx)
    try:
        assert tracer._resolve_agent_name(run) == "embedded-agent"
    finally:
        context.detach(token)
