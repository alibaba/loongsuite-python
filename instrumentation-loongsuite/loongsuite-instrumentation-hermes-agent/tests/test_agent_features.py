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

from pathlib import Path

import pytest

hermes_state = pytest.importorskip(
    "hermes_state",
    reason="Hermes runtime integration tests require the hermes-agent source.",
)
SessionDB = hermes_state.SessionDB


def _spans_by_kind(span_exporter, span_kind: str):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.span.kind") == span_kind
    ]


def _assert_parent(child, parent):
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


@pytest.mark.vcr()
def test_session_management_uses_previous_turn_context(
    require_live_hermes_env, instrument, build_agent, span_exporter
):
    agent = build_agent(enabled_toolsets=[], max_iterations=1)
    agent._disable_streaming = True

    first_turn = agent.run_conversation(
        "请记住：我的名字是小明。请只回复：已记住。"
    )
    second_turn = agent.run_conversation(
        "我刚才叫什么名字？请只回复名字。",
        conversation_history=first_turn["messages"],
    )

    assert first_turn["final_response"] == "已记住。"
    assert "小明" in second_turn["final_response"]

    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    llm_spans = _spans_by_kind(span_exporter, "LLM")
    assert not _spans_by_kind(span_exporter, "ENTRY")
    assert len(agent_spans) == 2
    assert len(llm_spans) == 2
    assert "小明" in llm_spans[1].attributes["gen_ai.input.messages"]
    assert agent_spans[0].attributes["gen_ai.conversation.id"]


@pytest.mark.vcr()
def test_memory_management_example(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = build_agent(
        enabled_toolsets=["memory"],
        max_iterations=4,
        skip_memory=False,
    )

    result = agent.run_conversation(
        "请务必调用 memory 工具，把“我的最爱颜色是蓝色”保存到 memory，然后只回复：已保存。"
    )

    assert result["final_response"] == "已保存。"
    assert agent._memory_store is not None
    assert "我的最爱颜色是蓝色" in agent._memory_store.memory_entries

    memory_file = tmp_path / "memories" / "MEMORY.md"
    assert memory_file.exists()
    assert "我的最爱颜色是蓝色" in memory_file.read_text(encoding="utf-8")

    step_spans = _spans_by_kind(span_exporter, "STEP")
    tool_spans = _spans_by_kind(span_exporter, "TOOL")
    assert tool_spans
    assert any(
        span.attributes.get("gen_ai.react.finish_reason") == "tool_calls"
        for span in step_spans
    )
    assert any(
        span.attributes.get("gen_ai.tool.name") == "memory"
        and "我的最爱颜色是蓝色"
        in span.attributes.get("gen_ai.tool.call.arguments", "")
        for span in tool_spans
    )


@pytest.mark.vcr()
def test_multi_agent_collaboration_example(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    public_fixture_paths,
    monkeypatch,
):
    path = public_fixture_paths["multi_agent"]
    workspace = public_fixture_paths["workspace"]
    monkeypatch.setenv("TERMINAL_CWD", workspace)
    monkeypatch.chdir(workspace)

    agent = build_agent(
        enabled_toolsets=["delegation", "file_tools"],
        max_iterations=4,
    )
    result = agent.run_conversation(
        f"请务必调用 delegate_task，把任务委派给子agent。"
        f"子agent需要读取文件 {path} 的内容，并返回结果。"
        "你最后只回复子agent返回的文件内容，不要解释。"
    )

    assert result["final_response"].strip()

    entry_spans = _spans_by_kind(span_exporter, "ENTRY")
    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    step_spans = _spans_by_kind(span_exporter, "STEP")
    tool_spans = _spans_by_kind(span_exporter, "TOOL")
    llm_spans = _spans_by_kind(span_exporter, "LLM")

    assert not entry_spans, (
        "Expected no ENTRY span from direct run_conversation"
    )
    assert len(agent_spans) >= 2, "Expected parent + child agent spans"
    assert len(step_spans) >= 2, "Expected parent + child step spans"
    assert len(llm_spans) >= 2, "Expected parent + child LLM spans"
    delegate_spans = [
        span
        for span in tool_spans
        if span.attributes.get("gen_ai.tool.name") == "delegate_task"
    ]
    assert delegate_spans
    delegate_span = delegate_spans[0]
    assert Path(path).name in delegate_span.attributes.get(
        "gen_ai.tool.call.arguments", ""
    )
    assert any(
        span.parent is not None
        and span.parent.span_id == delegate_span.context.span_id
        for span in agent_spans
        if span.parent is not None
    )


@pytest.mark.vcr()
def test_planning_example(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    isolated_hermes_home,
):
    planning_file = Path(
        "/tmp/loongsuite-hermes-agent-tests/planning_input.txt"
    )
    planning_file.parent.mkdir(parents=True, exist_ok=True)
    planning_file.write_text(
        "title=otelgui demo\nCHECKPOINT=planning_trace_ok\npayload=otelgui_e2e_ok\n",
        encoding="utf-8",
    )

    agent = build_agent(
        enabled_toolsets=["todo", "file"],
        max_iterations=6,
        session_db=SessionDB(isolated_hermes_home / "state.db"),
    )
    agent._disable_streaming = True

    result = agent.run_conversation(
        "这是一个必须展示规划过程的多步任务。"
        "你必须先调用 todo 工具创建 3 个任务："
        "p1=读取文件，p2=提取 CHECKPOINT 行，p3=输出最终答案。"
        "创建时只能让 p1 处于 in_progress。"
        f"随后必须调用 read_file 读取文件 {planning_file}。"
        "拿到内容后，再次调用 todo 把 p1/p2/p3 更新为 completed。"
        "最后只回复文件中以 CHECKPOINT= 开头的整行，不要解释，也不要输出 todo 内容。"
    )

    assert result["final_response"].strip() == "CHECKPOINT=planning_trace_ok"

    tool_spans = _spans_by_kind(span_exporter, "TOOL")
    step_spans = _spans_by_kind(span_exporter, "STEP")
    assert any(
        span.attributes.get("gen_ai.tool.name") == "todo"
        for span in tool_spans
    )
    assert any(
        span.attributes.get("gen_ai.tool.name") == "read_file"
        for span in tool_spans
    )
    assert any(
        span.attributes.get("gen_ai.react.finish_reason") == "tool_calls"
        for span in step_spans
    )
    assert any(
        span.attributes.get("gen_ai.react.finish_reason") == "stop"
        for span in step_spans
    )


@pytest.mark.vcr()
def test_mcp_integration_example(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    local_demo_mcp_home,
):
    agent = build_agent(
        enabled_toolsets=["demo"],
        max_iterations=4,
        session_db=SessionDB(local_demo_mcp_home / "state.db"),
        reload_mcp=True,
    )
    agent._disable_streaming = True

    result = agent.run_conversation(
        "请务必调用 mcp_demo_get_current_time 工具，timezone 传 Asia/Shanghai。"
        "拿到工具结果后，只回复返回 JSON 里的 iso 字段，不要解释，禁止自己猜时间。"
    )

    assert "T" in result["final_response"]
    tool_spans = _spans_by_kind(span_exporter, "TOOL")
    assert any(
        span.attributes.get("gen_ai.tool.name") == "mcp_demo_get_current_time"
        for span in tool_spans
    )


@pytest.mark.vcr()
def test_rag_example(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    local_demo_mcp_home,
):
    agent = build_agent(
        enabled_toolsets=["demo"],
        max_iterations=4,
        session_db=SessionDB(local_demo_mcp_home / "state.db"),
        reload_mcp=True,
    )
    agent._disable_streaming = True

    result = agent.run_conversation(
        "请务必调用 mcp_demo_search_briefing 工具，query 传 apollo telemetry。"
        "然后只回复工具结果里的 answer 字段，不要解释，也不要补充别的信息。"
    )

    assert (
        "Apollo telemetry keeps ENTRY > AGENT > STEP"
        in result["final_response"]
    )
    tool_spans = _spans_by_kind(span_exporter, "TOOL")
    assert any(
        span.attributes.get("gen_ai.tool.name") == "mcp_demo_search_briefing"
        for span in tool_spans
    )


@pytest.mark.vcr()
def test_evaluation_example(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    local_demo_mcp_home,
):
    agent = build_agent(
        enabled_toolsets=["demo"],
        max_iterations=4,
        session_db=SessionDB(local_demo_mcp_home / "state.db"),
        reload_mcp=True,
    )
    agent._disable_streaming = True

    result = agent.run_conversation(
        "请务必调用 mcp_demo_grade_candidate 工具做评估。"
        "reference 传 ENTRY > AGENT > STEP hierarchy with tools under the active step。"
        "candidate 传 The trace keeps ENTRY > AGENT > STEP hierarchy with tools under the active step.。"
        "rubric 传 exact_keyword_overlap。"
        "拿到结果后，只回复 verdict 字段，不要解释。"
    )

    assert "PASS" in result["final_response"].upper()
    tool_spans = _spans_by_kind(span_exporter, "TOOL")
    assert any(
        span.attributes.get("gen_ai.tool.name") == "mcp_demo_grade_candidate"
        for span in tool_spans
    )
