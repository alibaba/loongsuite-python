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

"""Runtime matrix against the pinned official DeerFlow 2.x source."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import pytest
from pydantic import PrivateAttr

from opentelemetry.instrumentation.deerflow import DeerFlowInstrumentor
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.langgraph import LangGraphInstrumentor
from opentelemetry.trace import StatusCode

deerflow = pytest.importorskip("deerflow")

from deerflow.agents.factory import create_deerflow_agent  # noqa: E402
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import (  # noqa: E402
    BaseChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402


def _final_message(content: str = "finished") -> AIMessage:
    return AIMessage(
        content=content,
        response_metadata={"finish_reason": "stop"},
    )


def _tool_call_message(
    tool_name: str,
    *,
    call_id: str = "call-1",
    args: dict[str, Any] | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": args or {},
                "id": call_id,
                "type": "tool_call",
            }
        ],
        response_metadata={"finish_reason": "tool_calls"},
    )


class _ScriptedChatModel(BaseChatModel):
    """Small deterministic model supporting sync, stream, and async graphs."""

    _responses: list[AIMessage | BaseException] = PrivateAttr()
    _repeat_tool_name: str | None = PrivateAttr()
    _calls: int = PrivateAttr(default=0)

    def __init__(
        self,
        responses: Sequence[AIMessage | BaseException] = (),
        *,
        repeat_tool_name: str | None = None,
    ) -> None:
        super().__init__()
        self._responses = list(responses)
        self._repeat_tool_name = repeat_tool_name

    @property
    def _llm_type(self) -> str:
        return "fake-deerflow"

    @property
    def _identifying_params(self) -> dict:
        return {}

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def _next_response(self) -> AIMessage:
        self._calls += 1
        if self._responses:
            response = self._responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        if self._repeat_tool_name is not None:
            return _tool_call_message(
                self._repeat_tool_name,
                call_id=f"call-{self._calls}",
            )
        return _final_message()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        message = self._next_response()
        finish_reason = message.response_metadata.get(
            "finish_reason",
            "stop",
        )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={"finish_reason": finish_reason},
                )
            ],
            llm_output={"model_name": "fake-deerflow"},
        )


class _ProviderCallbackRecorder:
    """Record provider-style callbacks without contacting either backend."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any] | None]] = []
        # LangGraph recognizes LangSmith tracers by type and reads this map.
        self.run_map: dict[str, Any] = {}

    def _record(self, event: str, kwargs: dict[str, Any]) -> None:
        metadata = kwargs.get("metadata")
        self.events.append(
            (event, metadata if isinstance(metadata, dict) else None)
        )

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("chain_start", kwargs)

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("chain_end", kwargs)

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("chain_error", kwargs)

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("chat_model_start", kwargs)

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("llm_start", kwargs)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("llm_end", kwargs)

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("llm_error", kwargs)

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("tool_start", kwargs)

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("tool_end", kwargs)

    def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("tool_error", kwargs)

    def on_agent_action(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("agent_action", kwargs)

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        del args
        self._record("agent_finish", kwargs)


@tool
def lookup(query: str) -> str:
    """Look up a deterministic test value."""
    return f"result:{query}"


@tool
def fail_tool() -> str:
    """Fail deterministically for lifecycle verification."""
    raise ValueError("tool failed")


@tool
def ping() -> str:
    """Return a deterministic value for recursion tests."""
    return "pong"


@pytest.fixture
def instrumented_deerflow(tracer_provider):
    instrumentor = DeerFlowInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        yield
    finally:
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def _spans_of_kind(spans, kind: str):
    return [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == kind
    ]


def _is_descendant(child, ancestor, spans) -> bool:
    spans_by_id = {span.context.span_id: span for span in spans}
    parent = child.parent
    while parent is not None:
        if parent.span_id == ancestor.context.span_id:
            return True
        parent_span = spans_by_id.get(parent.span_id)
        parent = parent_span.parent if parent_span is not None else None
    return False


def _assert_direct_sdk_tree(spans, *, model_decisions: int) -> None:
    assert _spans_of_kind(spans, "ENTRY") == []

    agents = _spans_of_kind(spans, "AGENT")
    steps = _spans_of_kind(spans, "STEP")
    llms = _spans_of_kind(spans, "LLM")
    assert len(agents) == 1
    assert len(steps) == model_decisions
    assert len(llms) == model_decisions

    agent = agents[0]
    assert agent.parent is None
    assert agent.attributes["gen_ai.framework"] == "deerflow"
    assert agent.attributes["gen_ai.agent.name"] == "sdk-agent"
    assert all(step.parent.span_id == agent.context.span_id for step in steps)
    assert all(
        sum(_is_descendant(llm, step, spans) for step in steps) == 1
        for llm in llms
    )


def _new_graph(model, tools=None):
    return create_deerflow_agent(
        model=model,
        tools=tools,
        middleware=[],
        name="sdk-agent",
    )


@pytest.mark.parametrize("operation", ["invoke", "stream"])
def test_direct_sdk_sync_operations_have_agent_step_without_entry(
    operation,
    instrumented_deerflow,
    span_exporter,
):
    graph = _new_graph(_ScriptedChatModel([_final_message()]))

    if operation == "invoke":
        graph.invoke({"messages": [{"role": "user", "content": "hello"}]})
    else:
        list(
            graph.stream({"messages": [{"role": "user", "content": "hello"}]})
        )

    assert getattr(graph, "_loongsuite_agent_flavor") == "deerflow"
    _assert_direct_sdk_tree(
        span_exporter.get_finished_spans(),
        model_decisions=1,
    )


def test_langsmith_and_langfuse_callbacks_keep_user_config(
    instrumented_deerflow,
    span_exporter,
):
    langfuse_langchain = pytest.importorskip("langfuse.langchain")
    from langchain_core.tracers.langchain import (  # noqa: PLC0415
        LangChainTracer,
    )

    class _LangSmithRecorder(_ProviderCallbackRecorder, LangChainTracer):
        pass

    class _LangfuseRecorder(
        _ProviderCallbackRecorder,
        langfuse_langchain.CallbackHandler,
    ):
        pass

    langsmith_callback = _LangSmithRecorder()
    langfuse_callback = _LangfuseRecorder()
    callbacks = [langsmith_callback, langfuse_callback]
    metadata = {
        "customer": "kept",
        "langfuse_trace_name": "deerflow-coexistence",
    }
    config = {"callbacks": callbacks, "metadata": metadata}
    graph = _new_graph(_ScriptedChatModel([_final_message()]))

    graph.invoke(
        {"messages": [{"role": "user", "content": "hello"}]},
        config=config,
    )

    assert config["callbacks"] is callbacks
    assert config["metadata"] is metadata
    assert "_loongsuite_agent_flavor" not in metadata
    for callback in callbacks:
        assert callback.events
        assert any(
            event_metadata is not None
            and event_metadata.get("customer") == "kept"
            for _, event_metadata in callback.events
        )
    _assert_direct_sdk_tree(
        span_exporter.get_finished_spans(),
        model_decisions=1,
    )


def test_dependency_instrumentors_do_not_duplicate_agent_or_step(
    tracer_provider,
    span_exporter,
):
    langchain_instrumentor = LangChainInstrumentor()
    langgraph_instrumentor = LangGraphInstrumentor()
    deerflow_instrumentor = DeerFlowInstrumentor()
    langchain_instrumentor.instrument(tracer_provider=tracer_provider)
    langgraph_instrumentor.instrument()
    deerflow_instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        graph = _new_graph(_ScriptedChatModel([_final_message()]))
        graph.invoke({"messages": [{"role": "user", "content": "hello"}]})

        spans = span_exporter.get_finished_spans()
        assert len(_spans_of_kind(spans, "AGENT")) == 1
        assert len(_spans_of_kind(spans, "STEP")) == 1
        assert len(_spans_of_kind(spans, "LLM")) == 1
    finally:
        deerflow_instrumentor.uninstrument()
        langgraph_instrumentor.uninstrument()
        langchain_instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_direct_sdk_ainvoke_has_agent_step_without_entry(
    instrumented_deerflow,
    span_exporter,
):
    graph = _new_graph(_ScriptedChatModel([_final_message()]))

    await graph.ainvoke({"messages": [{"role": "user", "content": "hello"}]})

    _assert_direct_sdk_tree(
        span_exporter.get_finished_spans(),
        model_decisions=1,
    )


def test_two_decision_tool_loop_has_exact_steps_and_finish_reasons(
    instrumented_deerflow,
    span_exporter,
):
    model = _ScriptedChatModel(
        [
            _tool_call_message(
                "lookup",
                args={"query": "deerflow"},
            ),
            _final_message("done"),
        ]
    )
    graph = _new_graph(model, [lookup])

    graph.invoke({"messages": [{"role": "user", "content": "research"}]})

    spans = span_exporter.get_finished_spans()
    _assert_direct_sdk_tree(spans, model_decisions=2)
    steps = _spans_of_kind(spans, "STEP")
    tools = _spans_of_kind(spans, "TOOL")
    assert [
        step.attributes.get("gen_ai.react.finish_reason") for step in steps
    ] == ["tool_calls", "stop"]
    assert len(tools) == 1
    assert tools[0].attributes["gen_ai.tool.name"] == "lookup"


def test_model_exception_closes_step_and_agent_as_error(
    instrumented_deerflow,
    span_exporter,
):
    graph = _new_graph(_ScriptedChatModel([RuntimeError("model failed")]))

    with pytest.raises(RuntimeError, match="model failed"):
        graph.invoke({"messages": [{"role": "user", "content": "fail"}]})

    spans = span_exporter.get_finished_spans()
    _assert_direct_sdk_tree(spans, model_decisions=1)
    assert _spans_of_kind(spans, "LLM")[0].status.status_code == (
        StatusCode.ERROR
    )
    assert _spans_of_kind(spans, "STEP")[0].status.status_code == (
        StatusCode.ERROR
    )
    assert _spans_of_kind(spans, "AGENT")[0].status.status_code == (
        StatusCode.ERROR
    )


def test_tool_exception_closes_tool_step_and_agent_as_error(
    instrumented_deerflow,
    span_exporter,
):
    graph = _new_graph(
        _ScriptedChatModel([_tool_call_message("fail_tool")]),
        [fail_tool],
    )

    with pytest.raises(ValueError, match="tool failed"):
        graph.invoke({"messages": [{"role": "user", "content": "fail"}]})

    spans = span_exporter.get_finished_spans()
    _assert_direct_sdk_tree(spans, model_decisions=1)
    assert _spans_of_kind(spans, "TOOL")[0].status.status_code == (
        StatusCode.ERROR
    )
    assert _spans_of_kind(spans, "STEP")[0].status.status_code == (
        StatusCode.ERROR
    )
    assert _spans_of_kind(spans, "AGENT")[0].status.status_code == (
        StatusCode.ERROR
    )


def test_recursion_limit_closes_last_step_and_agent_as_error(
    instrumented_deerflow,
    span_exporter,
):
    graph = _new_graph(
        _ScriptedChatModel(repeat_tool_name="ping"),
        [ping],
    )

    with pytest.raises(GraphRecursionError):
        graph.invoke(
            {"messages": [{"role": "user", "content": "loop"}]},
            config={"recursion_limit": 3},
        )

    spans = span_exporter.get_finished_spans()
    _assert_direct_sdk_tree(spans, model_decisions=2)
    steps = _spans_of_kind(spans, "STEP")
    assert steps[0].attributes["gen_ai.react.finish_reason"] == "tool_calls"
    assert steps[-1].status.status_code == StatusCode.ERROR
    assert _spans_of_kind(spans, "AGENT")[0].status.status_code == (
        StatusCode.ERROR
    )
