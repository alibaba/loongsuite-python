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

"""Real DeerFlow ``task`` and subagent-executor trace regressions."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from pydantic import PrivateAttr

from opentelemetry.instrumentation.deerflow import DeerFlowInstrumentor

deerflow = pytest.importorskip("deerflow")

from deerflow.agents.factory import create_deerflow_agent  # noqa: E402
from deerflow.subagents.config import SubagentConfig  # noqa: E402
from deerflow.tools.builtins.task_tool import task_tool  # noqa: E402
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import (  # noqa: E402
    BaseChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


def _final_message(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        response_metadata={"finish_reason": "stop"},
    )


def _task_calls_message(*subagent_names: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {
                    "description": f"delegate {name}",
                    "prompt": f"research with {name}",
                    "subagent_type": name,
                },
                "id": f"call-{name}",
                "type": "tool_call",
            }
            for name in subagent_names
        ],
        response_metadata={"finish_reason": "tool_calls"},
    )


class _ScriptedChatModel(BaseChatModel):
    """Deterministic model used by the real lead and subagent graphs."""

    _responses: list[AIMessage] = PrivateAttr()

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        super().__init__()
        self._responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "fake-deerflow-subagent"

    @property
    def _identifying_params(self) -> dict:
        return {}

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        message = self._responses.pop(0)
        finish_reason = message.response_metadata["finish_reason"]
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={"finish_reason": finish_reason},
                )
            ],
            llm_output={"model_name": "fake-deerflow-subagent"},
        )


class _FastAsyncio:
    """Proxy only DeerFlow task polling sleep; keep global asyncio untouched."""

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(_delay: float) -> None:
        # The real task tool polls every five seconds.  Keep a small real wait
        # so the official scheduler thread and isolated event loop get CPU,
        # while making the regression deterministic and fast.
        await asyncio.sleep(0.05)


@pytest.fixture
def instrumented_deerflow(tracer_provider):
    instrumentor = DeerFlowInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        yield
    finally:
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.fixture
def real_subagent_runtime(monkeypatch):
    """Keep official task/executor concurrency, replacing only external IO."""
    task_module = importlib.import_module("deerflow.tools.builtins.task_tool")
    executor_module = importlib.import_module("deerflow.subagents.executor")
    tools_module = importlib.import_module("deerflow.tools")
    middleware_module = importlib.import_module(
        "deerflow.agents.middlewares.tool_error_handling_middleware"
    )

    app_config = SimpleNamespace(
        tool_search=SimpleNamespace(
            enabled=False,
            auto_promote_top_k=0,
        )
    )
    models: dict[str, BaseChatModel] = {}

    def get_subagent_config(
        name: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> SubagentConfig:
        return SubagentConfig(
            name=name,
            description=f"test subagent {name}",
            model=name,
            skills=[],
            max_turns=10,
            timeout_seconds=10,
        )

    def create_chat_model(name: str, **_kwargs: Any) -> BaseChatModel:
        return models[name]

    monkeypatch.setattr(task_module, "asyncio", _FastAsyncio())
    monkeypatch.setattr(
        task_module,
        "get_available_subagent_names",
        lambda *_args, **_kwargs: list(models),
    )
    monkeypatch.setattr(
        task_module,
        "get_subagent_config",
        get_subagent_config,
    )
    monkeypatch.setattr(
        task_module,
        "resolve_subagent_model_name",
        lambda config, _parent, **_kwargs: config.model,
    )
    monkeypatch.setattr(
        task_module,
        "resolve_runtime_user_id",
        lambda runtime: (runtime.context or {}).get("user_id"),
    )
    monkeypatch.setattr(
        task_module,
        "_token_usage_cache_enabled",
        lambda _config: False,
    )
    monkeypatch.setattr(task_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(
        tools_module,
        "get_available_tools",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        executor_module,
        "get_app_config",
        lambda: app_config,
    )
    monkeypatch.setattr(
        executor_module,
        "create_chat_model",
        create_chat_model,
    )
    monkeypatch.setattr(
        executor_module,
        "build_tracing_callbacks",
        lambda: [],
    )
    monkeypatch.setattr(
        middleware_module,
        "build_subagent_runtime_middlewares",
        lambda **_kwargs: [],
    )

    yield models

    # A failed assertion must not leave official background-task state behind.
    background_tasks = getattr(executor_module, "_background_tasks", {})
    background_tasks.clear()


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


def _assert_subagent_tree(spans, *subagent_names: str) -> None:
    assert _spans_of_kind(spans, "ENTRY") == []

    agents = _spans_of_kind(spans, "AGENT")
    tools = _spans_of_kind(spans, "TOOL")
    steps = _spans_of_kind(spans, "STEP")
    llms = _spans_of_kind(spans, "LLM")

    expected_agent_names = {
        "lead-agent",
        *(f"subagent:{name}" for name in subagent_names),
    }
    assert {span.attributes["gen_ai.agent.name"] for span in agents} == (
        expected_agent_names
    )
    assert len(agents) == len(expected_agent_names)
    assert len(tools) == len(subagent_names)
    assert all(tool.attributes["gen_ai.tool.name"] == "task" for tool in tools)
    assert len(steps) == 2 + len(subagent_names)
    assert len(llms) == 2 + len(subagent_names)

    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1

    lead = next(
        span
        for span in agents
        if span.attributes["gen_ai.agent.name"] == "lead-agent"
    )
    assert lead.parent is None

    subagents = [span for span in agents if span is not lead]
    tool_ids = {span.context.span_id for span in tools}
    assert {span.parent.span_id for span in subagents} == tool_ids
    assert all(_is_descendant(tool, lead, spans) for tool in tools)

    for subagent in subagents:
        child_steps = [
            step
            for step in steps
            if step.parent.span_id == subagent.context.span_id
        ]
        assert len(child_steps) == 1
        assert (
            sum(_is_descendant(llm, child_steps[0], spans) for llm in llms)
            == 1
        )


@pytest.mark.asyncio
async def test_official_task_runs_one_real_subagent_without_extra_entry(
    instrumented_deerflow,
    real_subagent_runtime,
    span_exporter,
):
    real_subagent_runtime["research"] = _ScriptedChatModel(
        [_final_message("subagent result")]
    )
    lead_model = _ScriptedChatModel(
        [
            _task_calls_message("research"),
            _final_message("lead result"),
        ]
    )
    graph = create_deerflow_agent(
        model=lead_model,
        tools=[task_tool],
        middleware=[],
        name="lead-agent",
    )

    await graph.ainvoke(
        {"messages": [{"role": "user", "content": "delegate"}]},
        config={"metadata": {"model_name": "lead-model"}},
        context={"thread_id": "thread-one", "user_id": "user-one"},
    )

    _assert_subagent_tree(
        span_exporter.get_finished_spans(),
        "research",
    )


@pytest.mark.asyncio
async def test_official_task_keeps_two_concurrent_subagents_on_distinct_parents(
    instrumented_deerflow,
    real_subagent_runtime,
    span_exporter,
):
    real_subagent_runtime.update(
        {
            "research-one": _ScriptedChatModel(
                [_final_message("first result")]
            ),
            "research-two": _ScriptedChatModel(
                [_final_message("second result")]
            ),
        }
    )
    lead_model = _ScriptedChatModel(
        [
            _task_calls_message("research-one", "research-two"),
            _final_message("combined result"),
        ]
    )
    graph = create_deerflow_agent(
        model=lead_model,
        tools=[task_tool],
        middleware=[],
        name="lead-agent",
    )

    await graph.ainvoke(
        {"messages": [{"role": "user", "content": "parallel delegate"}]},
        config={"metadata": {"model_name": "lead-model"}},
        context={"thread_id": "thread-two", "user_id": "user-two"},
    )

    _assert_subagent_tree(
        span_exporter.get_finished_spans(),
        "research-one",
        "research-two",
    )
