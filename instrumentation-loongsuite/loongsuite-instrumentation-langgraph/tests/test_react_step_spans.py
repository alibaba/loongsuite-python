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

"""Integration tests: LangGraph ReAct agent → LangChain tracer → OTel spans.

Requires both ``loongsuite-instrumentation-langgraph`` (patches
``create_react_agent``) and ``loongsuite-instrumentation-langchain``
(``LoongsuiteTracer`` that creates Agent / ReAct Step / LLM spans).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, TypedDict

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Fake chat models
# ---------------------------------------------------------------------------


class _FakeChatModelWithTools(BaseChatModel):
    """Chat model that returns pre-configured responses and supports
    ``bind_tools`` (required by ``create_react_agent``).
    """

    responses: List[AIMessage]
    _call_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self._call_count % len(self.responses)
        self._call_count += 1
        msg = self.responses[idx]
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=msg,
                    generation_info={"finish_reason": "stop"},
                )
            ],
            llm_output={"model_name": "fake-tool-calling"},
        )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self

    @property
    def _identifying_params(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@tool
def _dummy_tool(query: str) -> str:
    """A no-op tool used only for testing."""
    return f"result for {query}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_react_agent(responses: list[AIMessage], **extra: Any):
    # Import inside function so each call picks up the current (possibly
    # patched) module-level attribute rather than a stale top-level reference.
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

    llm = _FakeChatModelWithTools(responses=responses)
    return create_react_agent(llm, [_dummy_tool], **extra)


def _build_opt_in_harness():
    """Build a plain StateGraph marked by a framework adapter."""
    from langgraph.graph import StateGraph  # noqa: PLC0415

    class _State(TypedDict):
        result: str

    llm = _FakeChatModelWithTools(
        responses=[AIMessage(content="Harness answer.")]
    )

    def decision_node(state: _State) -> _State:
        message = llm.invoke("hello")
        return {**state, "result": str(message.content)}

    builder = StateGraph(_State)
    builder.add_node("decision", decision_node)
    builder.set_entry_point("decision")
    builder.set_finish_point("decision")
    graph = builder.compile(name="example-harness")
    graph._loongsuite_agent_framework = "example-harness"
    graph._loongsuite_agent_step_node = "decision"
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLangGraphReActSingleRound:
    """Single-round agent: LLM returns a plain answer without tool_calls."""

    def test_agent_span_created(self, instrument, span_exporter):
        """The graph-level span is recognised as an Agent."""
        graph = _build_react_agent([AIMessage(content="Final answer.")])
        graph.invoke({"messages": [("user", "hello")]})

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent_spans) == 1, (
            f"Expected 1 agent span, got {len(agent_spans)}: "
            f"{[s.name for s in spans]}"
        )

    def test_single_react_step_created(self, instrument, span_exporter):
        """One ReAct step span with finish_reason='stop'."""
        graph = _build_react_agent([AIMessage(content="Final answer.")])
        graph.invoke({"messages": [("user", "hello")]})

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step_spans) == 1, (
            f"Expected 1 step span, got {len(step_spans)}: "
            f"{[s.name for s in spans]}"
        )
        step = step_spans[0]
        assert step.name == "react step"
        assert step.attributes.get("gen_ai.operation.name") == "react"
        assert step.attributes.get("gen_ai.react.round") == 1
        assert step.attributes.get("gen_ai.react.finish_reason") == "stop"

    def test_span_hierarchy(self, instrument, span_exporter):
        """Hierarchy: Agent > ReAct Step > LLM."""
        graph = _build_react_agent([AIMessage(content="Final answer.")])
        graph.invoke({"messages": [("user", "hello")]})

        spans = span_exporter.get_finished_spans()

        agent_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(agent_spans) >= 1
        assert len(step_spans) >= 1

        agent = agent_spans[0]
        step = step_spans[0]

        # Step is child of Agent
        assert step.parent is not None
        assert step.parent.span_id == agent.context.span_id


def test_opt_in_state_graph_uses_declared_decision_node(
    instrument, span_exporter
):
    """A LangGraph-based harness can opt in without changing prebuilt rules."""
    graph = _build_opt_in_harness()

    result = graph.invoke({"result": ""})

    assert result["result"] == "Harness answer."
    spans = span_exporter.get_finished_spans()
    agent_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "AGENT"
    ]
    step_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "STEP"
    ]
    llm_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "LLM"
    ]

    assert len(agent_spans) == 1
    assert agent_spans[0].attributes["gen_ai.framework"] == "example-harness"
    assert len(step_spans) == 1
    assert step_spans[0].parent.span_id == agent_spans[0].context.span_id
    assert len(llm_spans) == 1
    assert llm_spans[0].context.trace_id == agent_spans[0].context.trace_id


class TestLangGraphReActMultiRound:
    """Multi-round agent: LLM uses a tool in round 1, answers in round 2."""

    def _invoke_two_rounds(self, instrument, span_exporter):
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "_dummy_tool",
                        "args": {"query": "test"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Final answer based on tool result."),
        ]
        graph = _build_react_agent(responses)
        graph.invoke({"messages": [("user", "hello")]})
        return span_exporter.get_finished_spans()

    def test_two_react_steps(self, instrument, span_exporter):
        """Two ReAct step spans: round 1 → tool_calls, round 2 → stop."""
        spans = self._invoke_two_rounds(instrument, span_exporter)
        step_spans = sorted(
            [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "STEP"
            ],
            key=lambda s: s.attributes.get("gen_ai.react.round", 0),
        )
        assert len(step_spans) == 2, (
            f"Expected 2 step spans, got {len(step_spans)}: "
            f"{[(s.name, s.attributes) for s in spans]}"
        )

        assert step_spans[0].attributes.get("gen_ai.react.round") == 1
        assert (
            step_spans[0].attributes.get("gen_ai.react.finish_reason")
            == "tool_calls"
        )

        assert step_spans[1].attributes.get("gen_ai.react.round") == 2
        assert (
            step_spans[1].attributes.get("gen_ai.react.finish_reason")
            == "stop"
        )

    def test_llm_spans_parented_under_steps(self, instrument, span_exporter):
        """LLM spans should be descendants of their corresponding step spans."""
        spans = self._invoke_two_rounds(instrument, span_exporter)
        step_spans = sorted(
            [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "STEP"
            ],
            key=lambda s: s.attributes.get("gen_ai.react.round", 0),
        )
        assert len(step_spans) >= 2

        step_1_id = step_spans[0].context.span_id
        step_2_id = step_spans[1].context.span_id

        # Collect all descendant span IDs for each step
        step_1_descendants = _collect_descendants(spans, step_1_id)
        step_2_descendants = _collect_descendants(spans, step_2_id)

        # There should be at least one LLM span under each step
        llm_in_step1 = [
            s
            for s in spans
            if s.context.span_id in step_1_descendants and _is_llm_span(s)
        ]
        llm_in_step2 = [
            s
            for s in spans
            if s.context.span_id in step_2_descendants and _is_llm_span(s)
        ]
        assert len(llm_in_step1) >= 1, "Expected LLM span under step 1"
        assert len(llm_in_step2) >= 1, "Expected LLM span under step 2"


def _collect_descendants(spans, parent_span_id: int) -> set[int]:
    """Return all span IDs that are descendants of *parent_span_id*."""
    children: dict[int, list[int]] = {}
    for s in spans:
        if s.parent is not None:
            children.setdefault(s.parent.span_id, []).append(s.context.span_id)

    result: set[int] = set()
    queue = list(children.get(parent_span_id, []))
    while queue:
        sid = queue.pop()
        result.add(sid)
        queue.extend(children.get(sid, []))
    return result


def _is_llm_span(span) -> bool:
    kind = span.attributes.get("gen_ai.span.kind", "")
    return (
        kind == "LLM" or span.attributes.get("gen_ai.operation.name") == "chat"
    )
