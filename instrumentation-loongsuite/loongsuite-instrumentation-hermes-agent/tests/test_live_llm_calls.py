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

import pytest
from conftest import HermesAgentInstrumentor, extract_metric_points

run_agent = pytest.importorskip(
    "run_agent",
    reason="Hermes runtime integration tests require the hermes-agent source.",
)
AIAgent = run_agent.AIAgent


def _metric_value(point):
    return getattr(point, "value", getattr(point, "sum", None))


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
def test_sync_llm_call_records_single_llm_span_and_metric(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    metric_reader,
):
    agent = build_agent(enabled_toolsets=[], max_iterations=1)
    agent._disable_streaming = True

    result = agent.run_conversation("请只回复：同步成功")

    assert result["final_response"] == "同步成功"

    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    step_spans = _spans_by_kind(span_exporter, "STEP")
    llm_spans = _spans_by_kind(span_exporter, "LLM")

    assert len(agent_spans) == 1
    assert len(step_spans) == 1
    assert len(llm_spans) == 1

    agent_span = agent_spans[0]
    step_span = step_spans[0]
    span = llm_spans[0]
    agent_output_messages = json.loads(
        agent_span.attributes["gen_ai.output.messages"]
    )
    llm_input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert agent_span.name == "invoke_agent Hermes"
    assert step_span.name == "react step"
    assert span.name == "chat qwen-turbo"
    assert agent_span.parent is None
    _assert_parent(step_span, agent_span)
    _assert_parent(span, step_span)

    assert agent_output_messages == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "同步成功"}],
            "finish_reason": "stop",
        }
    ]
    assert agent_span.attributes["gen_ai.agent.name"] == "Hermes"
    assert agent_span.attributes["gen_ai.provider.name"] == "hermes"
    assert agent_span.attributes["gen_ai.agent.system"] == "hermes"
    assert agent_span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent_span.attributes["gen_ai.conversation.id"]
    assert "gen_ai.agent.id" not in agent_span.attributes
    assert step_span.attributes["gen_ai.react.round"] == 1
    assert step_span.attributes["gen_ai.operation.name"] == "react"
    assert step_span.attributes["gen_ai.react.finish_reason"] == "stop"
    assert span.attributes["gen_ai.provider.name"] == "dashscope"
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "qwen-turbo"
    assert llm_input_messages[0]["role"] in {"system", "user"}
    assert all("parts" in message for message in llm_input_messages)
    assert span.attributes["gen_ai.response.model"]
    assert span.attributes["gen_ai.response.id"]
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert span.attributes["gen_ai.usage.output_tokens"] > 0
    assert span.attributes["gen_ai.response.finish_reasons"]

    metric_points = extract_metric_points(metric_reader, "genai_calls_count")
    assert metric_points, "Expected genai_calls_count data points"

    llm_points = [
        point
        for point in metric_points
        if point.attributes.get("spanKind") == "LLM"
        and point.attributes.get("modelName") == "qwen-turbo"
    ]
    assert llm_points, "Expected LLM metric data point for qwen-turbo"
    assert _metric_value(llm_points[0]) == 1
    assert llm_points[0].attributes["callType"] == "gen_ai"
    assert llm_points[0].attributes["callKind"] == "internal"
    assert llm_points[0].attributes["provider"] == "dashscope"

    duration_points = extract_metric_points(
        metric_reader, "genai_calls_duration_seconds"
    )
    assert duration_points, "Expected duration metric data point"

    usage_points = extract_metric_points(
        metric_reader, "genai_llm_usage_tokens"
    )
    assert any(
        point.attributes.get("tokenType") == "input" for point in usage_points
    )
    assert any(
        point.attributes.get("tokenType") == "output" for point in usage_points
    )
    assert any(
        point.attributes.get("tokenType") == "total" for point in usage_points
    )


@pytest.mark.vcr()
def test_streaming_llm_call_records_ttft(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    metric_reader,
):
    agent = build_agent(enabled_toolsets=[], max_iterations=1)
    chunks = []

    result = agent.run_conversation(
        "请数到3，每个数字之间加空格。",
        stream_callback=lambda token: chunks.append(token),
    )

    assert result["final_response"] == "1 2 3"
    assert chunks, "Expected streaming callback to receive chunks"

    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    step_spans = _spans_by_kind(span_exporter, "STEP")
    llm_spans = _spans_by_kind(span_exporter, "LLM")

    assert len(agent_spans) == 1
    assert len(step_spans) == 1
    assert len(llm_spans) == 1
    assert step_spans[0].attributes["gen_ai.react.finish_reason"] == "stop"
    assert llm_spans[0].attributes["gen_ai.response.time_to_first_token"] > 0
    assert agent_spans[0].attributes["gen_ai.response.time_to_first_token"] > 0
    assert agent_spans[0].parent is None
    _assert_parent(step_spans[0], agent_spans[0])
    _assert_parent(llm_spans[0], step_spans[0])

    metric_points = extract_metric_points(metric_reader, "genai_calls_count")
    assert sum(_metric_value(point) for point in metric_points) == 1


@pytest.mark.vcr()
def test_tool_call_creates_tool_span_and_multiple_llm_spans(
    require_live_hermes_env,
    instrument,
    build_agent,
    span_exporter,
    metric_reader,
    public_fixture_paths,
):
    path = public_fixture_paths["read_file"]

    agent = build_agent(enabled_toolsets=["file_tools"], max_iterations=4)
    result = agent.run_conversation(
        f"请务必调用 read_file 工具读取文件 {path} ，然后只回复文件内容，不要解释。"
    )

    assert "tool_ok" in result["final_response"]

    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    step_spans = _spans_by_kind(span_exporter, "STEP")
    llm_spans = _spans_by_kind(span_exporter, "LLM")
    tool_spans = _spans_by_kind(span_exporter, "TOOL")

    assert len(agent_spans) == 1
    assert len(llm_spans) >= 2
    assert len(tool_spans) >= 1
    assert len(step_spans) >= 2
    assert agent_spans[0].parent is None
    step_by_round = {
        span.attributes["gen_ai.react.round"]: span for span in step_spans
    }
    assert (
        step_by_round[1].attributes["gen_ai.react.finish_reason"]
        == "tool_calls"
    )
    assert (
        step_by_round[max(step_by_round)].attributes[
            "gen_ai.react.finish_reason"
        ]
        == "stop"
    )

    read_file_spans = [
        span
        for span in tool_spans
        if span.attributes["gen_ai.tool.name"] == "read_file"
    ]
    assert read_file_spans
    assert path in read_file_spans[0].attributes["gen_ai.tool.call.arguments"]
    _assert_parent(read_file_spans[0], step_by_round[1])

    llm_parent_ids = {
        span.parent.span_id for span in llm_spans if span.parent is not None
    }
    assert step_by_round[1].context.span_id in llm_parent_ids
    assert step_by_round[max(step_by_round)].context.span_id in llm_parent_ids

    metric_points = extract_metric_points(metric_reader, "genai_calls_count")
    assert sum(_metric_value(point) for point in metric_points) == len(
        llm_spans
    )


@pytest.mark.vcr()
def test_sync_retry_creates_two_llm_attempt_spans(
    require_live_hermes_env,
    tracer_provider,
    meter_provider,
    span_exporter,
    metric_reader,
    monkeypatch,
    build_agent,
):
    original_method = AIAgent._interruptible_api_call
    state = {"calls": 0}

    def flaky_interruptible_call(self, api_kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TimeoutError("synthetic retry")
        return original_method(self, api_kwargs)

    monkeypatch.setattr(
        AIAgent,
        "_interruptible_api_call",
        flaky_interruptible_call,
    )

    instrumentor = HermesAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    try:
        agent = build_agent(enabled_toolsets=[], max_iterations=2)
        agent._disable_streaming = True
        result = agent.run_conversation("请只回复：重试成功")
    finally:
        instrumentor.uninstrument()

    assert result["final_response"] == "重试成功"
    assert state["calls"] == 2

    step_spans = _spans_by_kind(span_exporter, "STEP")
    llm_spans = _spans_by_kind(span_exporter, "LLM")
    assert len(llm_spans) == 2
    assert len(step_spans) == 2
    assert step_spans[0].attributes["gen_ai.react.finish_reason"] == "error"

    metric_points = extract_metric_points(metric_reader, "genai_calls_count")
    assert sum(_metric_value(point) for point in metric_points) == 1
    error_points = extract_metric_points(
        metric_reader, "genai_calls_error_count"
    )
    assert sum(_metric_value(point) for point in error_points) == 1
