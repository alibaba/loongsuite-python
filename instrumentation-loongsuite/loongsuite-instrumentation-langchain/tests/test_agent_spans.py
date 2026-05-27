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

from opentelemetry.instrumentation.langchain.internal._tracer import (
    LoongsuiteTracer,
    _is_deepagents_subagent_candidate,
    _RunData,
)
from opentelemetry.instrumentation.langchain.internal._utils import (
    AGENT_RUN_NAMES,
    _is_agent_run,
)
from opentelemetry.trace import get_current_span, set_span_in_context
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


class _FakeRun:
    """Minimal stub that looks like a langchain Run for unit tests."""

    def __init__(self, name: str, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.name = name
        self.parent_run_id = kwargs.get("parent_run_id")
        self.metadata = kwargs.get("metadata", {})
        self.inputs = kwargs.get("inputs", {})
        self.outputs = kwargs.get("outputs", {})
        self.extra = kwargs.get("extra", {})


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


def test_deepagents_subagent_prefers_current_task_tool_parent(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    otel_tracer = tracer_provider.get_tracer(__name__)

    with otel_tracer.start_as_current_span("execute_tool task") as tool_span:
        tool_span.set_attribute("gen_ai.span.kind", "TOOL")
        tool_span.set_attribute("gen_ai.tool.name", "task")
        run = _FakeRun(
            "LangGraph",
            parent_run_id=uuid4(),
            metadata={
                "ls_integration": "deepagents",
                "lc_agent_name": "researcher",
            },
        )

        context = tracer._get_parent_context(run)

    assert get_current_span(context) is tool_span


def test_deepagents_task_tool_parent_ignores_non_langgraph_runs(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    otel_tracer = tracer_provider.get_tracer(__name__)

    with otel_tracer.start_as_current_span("execute_tool task") as tool_span:
        tool_span.set_attribute("gen_ai.span.kind", "TOOL")
        tool_span.set_attribute("gen_ai.tool.name", "task")
        run = _FakeRun(
            "RetrievalQA",
            parent_run_id=uuid4(),
            metadata={
                "ls_integration": "deepagents",
                "lc_agent_name": "researcher",
            },
        )

        context = tracer._get_parent_context(run)

    assert context is None


def test_parentless_chain_does_not_inherit_ambient_context(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    otel_tracer = tracer_provider.get_tracer(__name__)

    with otel_tracer.start_as_current_span("ambient") as ambient_span:
        run = _FakeRun("RetrievalQA")

        tracer._on_chain_start(run)
        try:
            span = tracer._runs[run.id].span
            assert span.parent is None
            assert span.context.trace_id != ambient_span.context.trace_id
        finally:
            tracer._on_chain_end(run)


def test_parentless_handler_spans_do_not_inherit_ambient_context(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    otel_tracer = tracer_provider.get_tracer(__name__)
    cases = (
        (tracer._handle_llm_start, _FakeRun("ChatOpenAI")),
        (
            tracer._start_agent,
            _FakeRun("AgentExecutor", inputs={"input": "hello"}),
        ),
        (tracer._on_tool_start, _FakeRun("search", inputs={"input": "q"})),
        (
            tracer._on_retriever_start,
            _FakeRun("retriever", inputs={"query": "q"}),
        ),
    )

    for start, run in cases:
        with otel_tracer.start_as_current_span("ambient") as ambient_span:
            start(run)
            try:
                span = tracer._runs[run.id].span
                assert span.parent is None
                assert span.context.trace_id != ambient_span.context.trace_id
            finally:
                tracer._runs.pop(run.id).span.end()


def test_deepagents_explicit_subagent_metadata_bypasses_langgraph_name():
    run = _FakeRun(
        "CustomAgent",
        metadata={
            "ls_integration": "deepagents",
            "ls_agent_type": "subagent",
        },
    )

    assert _is_deepagents_subagent_candidate(run)


def test_deepagents_subagent_falls_back_to_active_task_tool_run(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    otel_tracer = tracer_provider.get_tracer(__name__)
    parent_run_id = uuid4()

    tool_span = otel_tracer.start_span("execute_tool task")
    try:
        tracer._runs[uuid4()] = _RunData(
            run_kind="tool",
            span=tool_span,
            context=set_span_in_context(tool_span),
            parent_run_id=parent_run_id,
            tool_name="task",
        )
        run = _FakeRun(
            "LangGraph",
            parent_run_id=parent_run_id,
            metadata={
                "ls_integration": "deepagents",
                "lc_agent_name": "researcher",
                "ls_agent_type": "subagent",
            },
        )

        context = tracer._get_parent_context(run)
    finally:
        tool_span.end()

    assert get_current_span(context) is tool_span


def test_deepagents_subagent_falls_back_to_latest_task_tool_when_parent_differs(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    otel_tracer = tracer_provider.get_tracer(__name__)

    tool_span = otel_tracer.start_span("execute_tool task")
    try:
        tracer._runs[uuid4()] = _RunData(
            run_kind="tool",
            span=tool_span,
            context=set_span_in_context(tool_span),
            parent_run_id=uuid4(),
            tool_name="task",
        )
        run = _FakeRun(
            "LangGraph",
            parent_run_id=uuid4(),
            metadata={
                "ls_integration": "deepagents",
                "lc_agent_name": "researcher",
                "ls_agent_type": "subagent",
            },
        )

        context = tracer._get_parent_context(run)
    finally:
        tool_span.end()

    assert get_current_span(context) is tool_span


def test_start_agent_writes_deepagents_framework_attributes(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    run = _FakeRun(
        "LangGraph",
        metadata={
            "_loongsuite_react_agent": True,
            "ls_integration": "deepagents",
            "versions": {"deepagents": "0.6.3"},
            "lc_agent_name": "deep_agent",
        },
    )

    tracer._start_agent(run)
    try:
        span = tracer._runs[run.id].span
        attributes = span.attributes
    finally:
        tracer._runs.pop(run.id).span.end()

    assert attributes["gen_ai.framework"] == "deepagents"
    assert attributes["gen_ai.framework.version"] == "0.6.3"
    assert attributes["gen_ai.agent.name"] == "deep_agent"


def test_start_agent_writes_deepagents_subagent_attributes(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    run = _FakeRun(
        "LangGraph",
        metadata={
            "_loongsuite_react_agent": True,
            "ls_integration": "deepagents",
            "versions": {"deepagents": "0.6.3"},
            "lc_agent_name": "country-researcher",
            "ls_agent_type": "subagent",
            "loongsuite_deepagents_subagent_description": (
                "Research country facts"
            ),
        },
    )

    tracer._start_agent(run)
    try:
        span = tracer._runs[run.id].span
        attributes = span.attributes
    finally:
        tracer._runs.pop(run.id).span.end()

    assert attributes["gen_ai.framework"] == "deepagents"
    assert attributes["gen_ai.framework.version"] == "0.6.3"
    assert attributes["gen_ai.agent.name"] == "country-researcher"
    assert attributes["gen_ai.agent.type"] == "subagent"
    assert (
        attributes["gen_ai.agent.description"]
        == "Research country facts"
    )


def test_langgraph_router_treats_deepagents_subagent_root_as_agent(
    tracer_provider,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = LoongsuiteTracer(handler, tracer_provider=tracer_provider)
    parent_run_id = uuid4()
    tracer._runs[parent_run_id] = _RunData(
        run_kind="chain",
        inside_langgraph_react=True,
    )
    run = _FakeRun(
        "country-researcher",
        parent_run_id=parent_run_id,
        metadata={
            "_loongsuite_react_agent": True,
            "ls_integration": "deepagents",
            "versions": {"deepagents": "0.6.3"},
            "lc_agent_name": "country-researcher",
            "ls_agent_type": "subagent",
            "loongsuite_deepagents_subagent_description": (
                "Research country facts"
            ),
        },
    )

    tracer._on_chain_start(run)
    try:
        run_data = tracer._runs[run.id]
        attributes = run_data.span.attributes
    finally:
        tracer._runs.pop(run.id).span.end()
        tracer._runs.pop(parent_run_id)

    assert run_data.run_kind == "agent"
    assert attributes["gen_ai.framework"] == "deepagents"
    assert attributes["gen_ai.agent.type"] == "subagent"
