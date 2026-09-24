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

"""Integration tests for DeepAgents AGENT and ReAct STEP spans."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool


class _SequenceToolCallingModel(BaseChatModel):
    def __init__(self, responses: list[AIMessage]):
        super().__init__()
        self._responses = list(responses)
        self._index = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-deepagents"

    @property
    def _identifying_params(self) -> dict:
        return {}

    def bind_tools(self, tools, **kwargs):
        return self

    def _next_message(self) -> AIMessage:
        if self._index >= len(self._responses):
            return AIMessage(content="Deep agent fallback answer.")
        message = self._responses[self._index]
        self._index += 1
        return message

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        message = self._next_message()
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={"finish_reason": "stop"},
                )
            ],
            llm_output={"model_name": "fake-deepagents"},
        )


@tool
def lookup_city(query: str) -> str:
    """Look up a city name for tests."""
    del query
    return "Hangzhou"


def _build_agent(responses: list[AIMessage], *, name: str, **kwargs: Any):
    from deepagents import create_deep_agent  # noqa: PLC0415

    return create_deep_agent(
        model=_SequenceToolCallingModel(responses),
        tools=[lookup_city],
        system_prompt="Answer briefly.",
        name=name,
        **kwargs,
    )


def _root_spans(spans):
    return [span for span in spans if span.parent is None]


def _spans_by_kind(spans, kind: str):
    return [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == kind
    ]


def _collect_descendants(spans, parent_span_id: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for span in spans:
        if span.parent is not None:
            children.setdefault(span.parent.span_id, []).append(
                span.context.span_id
            )

    result: set[int] = set()
    queue = list(children.get(parent_span_id, []))
    while queue:
        span_id = queue.pop()
        result.add(span_id)
        queue.extend(children.get(span_id, []))
    return result


def _assert_single_agent(spans, name: str):
    root_spans = _root_spans(spans)
    root_kinds = {
        span.name: span.attributes.get("gen_ai.span.kind")
        for span in root_spans
    }
    assert root_kinds == {f"invoke_agent {name}": "AGENT"}
    assert not any(
        span.name == f"chain {name}"
        and span.attributes.get("gen_ai.span.kind") == "CHAIN"
        for span in root_spans
    )
    return root_spans[0]


def _assert_step_is_under_agent(agent_span, step_span) -> None:
    assert step_span.parent is not None
    assert step_span.parent.span_id == agent_span.context.span_id


def _assert_kind_under_step(
    spans, step_span, kind: str, name: str | None = None
):
    descendants = _collect_descendants(spans, step_span.context.span_id)
    matches = [
        span
        for span in spans
        if span.context.span_id in descendants
        and span.attributes.get("gen_ai.span.kind") == kind
        and (name is None or span.name == name)
    ]
    assert matches, f"Expected {kind} span under {step_span.name}"
    return matches


def test_deepagents_root_span_is_agent_and_single_step(
    instrument, span_exporter
):
    agent = _build_agent(
        [AIMessage(content="Deep agent final answer.")],
        name="deep_test_agent",
    )

    assert getattr(agent, "_loongsuite_react_agent") is True
    assert getattr(agent, "_loongsuite_deepagents_agent") is True

    result = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})

    assert result["messages"][-1].content == "Deep agent final answer."

    spans = span_exporter.get_finished_spans()
    agent_span = _assert_single_agent(spans, "deep_test_agent")
    input_raw = agent_span.attributes.get("gen_ai.input.messages")
    assert input_raw, "AGENT span missing gen_ai.input.messages"
    input_messages = json.loads(input_raw)
    assert input_messages == [
        {
            "role": "user",
            "parts": [{"content": "hello", "type": "text"}],
        }
    ]

    step_spans = _spans_by_kind(spans, "STEP")
    assert len(step_spans) == 1
    step = step_spans[0]
    _assert_step_is_under_agent(agent_span, step)
    assert step.name == "react step"
    assert step.attributes.get("gen_ai.operation.name") == "react"
    assert step.attributes.get("gen_ai.react.round") == 1
    assert step.attributes.get("gen_ai.react.finish_reason") == "stop"
    _assert_kind_under_step(spans, step, "CHAIN", "chain model")
    _assert_kind_under_step(spans, step, "LLM")


def test_deepagents_tool_call_creates_two_react_steps(
    instrument, span_exporter
):
    agent = _build_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_city",
                        "args": {"query": "city"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Final answer: Hangzhou."),
        ],
        name="deep_tool_agent",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Where is it?"}]}
    )

    assert result["messages"][-1].content == "Final answer: Hangzhou."

    spans = span_exporter.get_finished_spans()
    agent_span = _assert_single_agent(spans, "deep_tool_agent")

    step_spans = sorted(
        _spans_by_kind(spans, "STEP"),
        key=lambda span: span.attributes.get("gen_ai.react.round", 0),
    )
    assert len(step_spans) == 2

    assert step_spans[0].attributes.get("gen_ai.react.round") == 1
    assert (
        step_spans[0].attributes.get("gen_ai.react.finish_reason")
        == "tool_calls"
    )
    assert step_spans[1].attributes.get("gen_ai.react.round") == 2
    assert step_spans[1].attributes.get("gen_ai.react.finish_reason") == "stop"

    for step in step_spans:
        _assert_step_is_under_agent(agent_span, step)
        _assert_kind_under_step(spans, step, "CHAIN", "chain model")
        _assert_kind_under_step(spans, step, "LLM")

    _assert_kind_under_step(
        spans, step_spans[0], "TOOL", "execute_tool lookup_city"
    )


def test_deepagents_skill_load_tool_span_captures_skill_attributes(
    instrument, span_exporter, tmp_path
):
    skill_dir = tmp_path / "skills" / "probe-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: probe-skill",
                "description: Use this skill for telemetry validation.",
                "metadata:",
                "  version: 1.2.3",
                "---",
                "",
                "# Probe Skill",
                "Return PROBE_SKILL_LOADED when active.",
            ]
        ),
        encoding="utf-8",
    )

    agent = _build_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {
                            "file_path": "/skills/probe-skill/SKILL.md",
                            "limit": 1000,
                        },
                        "id": "call_skill_1",
                    }
                ],
            ),
            AIMessage(content="Loaded probe skill."),
        ],
        name="deep_skill_agent",
        skills=["/skills"],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "load probe skill"}]}
    )

    assert result["messages"][-1].content == "Loaded probe skill."

    spans = span_exporter.get_finished_spans()
    read_file_spans = [
        span
        for span in spans
        if span.name == "execute_tool read_file"
        and span.attributes.get("gen_ai.tool.name") == "read_file"
    ]
    assert len(read_file_spans) == 1
    attrs = dict(read_file_spans[0].attributes or {})
    assert attrs["gen_ai.skill.name"] == "probe-skill"
    assert attrs["gen_ai.skill.id"] == "probe-skill"
    assert (
        attrs["gen_ai.skill.description"]
        == "Use this skill for telemetry validation."
    )
    assert attrs["gen_ai.skill.version"] == "1.2.3"
    assert "/skills/probe-skill/SKILL.md" in str(
        attrs["gen_ai.tool.call.arguments"]
    )


def test_deepagents_skill_helper_file_is_not_skill_load(
    instrument, span_exporter, tmp_path
):
    skill_dir = tmp_path / "skills" / "probe-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: probe-skill",
                "description: Use this skill for telemetry validation.",
                "---",
                "",
                "# Probe Skill",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "helper.py").write_text("print('helper')\n", encoding="utf-8")

    agent = _build_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {
                            "file_path": "/skills/probe-skill/helper.py",
                            "limit": 1000,
                        },
                        "id": "call_helper_1",
                    }
                ],
            ),
            AIMessage(content="Read helper."),
        ],
        name="deep_skill_helper_agent",
        skills=["/skills"],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "read helper"}]}
    )

    assert result["messages"][-1].content == "Read helper."

    spans = span_exporter.get_finished_spans()
    read_file_spans = [
        span
        for span in spans
        if span.name == "execute_tool read_file"
        and span.attributes.get("gen_ai.tool.name") == "read_file"
    ]
    assert len(read_file_spans) == 1
    attrs = dict(read_file_spans[0].attributes or {})
    assert "gen_ai.skill.name" not in attrs
    assert "/skills/probe-skill/helper.py" in str(
        attrs["gen_ai.tool.call.arguments"]
    )


def test_deepagents_stream_creates_react_step(instrument, span_exporter):
    agent = _build_agent(
        [AIMessage(content="Stream final answer.")],
        name="deep_stream_agent",
    )

    chunks = list(
        agent.stream({"messages": [{"role": "user", "content": "hello"}]})
    )

    assert chunks

    spans = span_exporter.get_finished_spans()
    agent_span = _assert_single_agent(spans, "deep_stream_agent")
    step_spans = _spans_by_kind(spans, "STEP")
    assert len(step_spans) == 1
    _assert_step_is_under_agent(agent_span, step_spans[0])
    assert step_spans[0].attributes.get("gen_ai.react.round") == 1
    assert step_spans[0].attributes.get("gen_ai.react.finish_reason") == "stop"
    _assert_kind_under_step(spans, step_spans[0], "LLM")


@pytest.mark.asyncio
async def test_deepagents_async_invoke_creates_react_step(
    instrument, span_exporter
):
    agent = _build_agent(
        [AIMessage(content="Async final answer.")],
        name="deep_async_agent",
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "hello"}]}
    )

    assert result["messages"][-1].content == "Async final answer."

    spans = span_exporter.get_finished_spans()
    agent_span = _assert_single_agent(spans, "deep_async_agent")
    step_spans = _spans_by_kind(spans, "STEP")
    assert len(step_spans) == 1
    _assert_step_is_under_agent(agent_span, step_spans[0])
    _assert_kind_under_step(spans, step_spans[0], "LLM")


def test_deepagents_parallel_invocations_keep_step_parents(
    instrument, span_exporter
):
    def run_agent(index: int) -> str:
        agent = _build_agent(
            [AIMessage(content=f"Parallel final {index}.")],
            name=f"deep_parallel_agent_{index}",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"hello {index}"}]}
        )
        return result["messages"][-1].content

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_agent, [1, 2]))

    assert sorted(results) == ["Parallel final 1.", "Parallel final 2."]

    spans = span_exporter.get_finished_spans()
    agent_spans = {
        span.name: span
        for span in _spans_by_kind(spans, "AGENT")
        if span.name.startswith("invoke_agent deep_parallel_agent_")
    }
    assert set(agent_spans) == {
        "invoke_agent deep_parallel_agent_1",
        "invoke_agent deep_parallel_agent_2",
    }

    step_spans = _spans_by_kind(spans, "STEP")
    assert len(step_spans) == 2
    parent_ids = {
        step.parent.span_id for step in step_spans if step.parent is not None
    }
    assert parent_ids == {
        span.context.span_id for span in agent_spans.values()
    }
    assert {
        step.attributes.get("gen_ai.react.round") for step in step_spans
    } == {1}
    assert {
        step.attributes.get("gen_ai.react.finish_reason")
        for step in step_spans
    } == {"stop"}


def test_top_level_create_deep_agent_export_is_wrapped(instrument):
    import deepagents  # noqa: PLC0415
    import deepagents.graph  # noqa: PLC0415

    assert deepagents.create_deep_agent is deepagents.graph.create_deep_agent
