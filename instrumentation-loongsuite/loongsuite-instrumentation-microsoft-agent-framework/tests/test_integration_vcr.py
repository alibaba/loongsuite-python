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

"""Replay-only contracts against the real Microsoft Agent Framework package.

The cassette makes provider transport deterministic; the assertions protect
the post-processor trace contract that users see: one agent root per logical
invocation, closed MAF parentage, content-mode semantics, streaming TTFT, tool
spans, error status, and context isolation. Provider HTTP spans are deliberately
outside the MAF-tree closure rule.
"""

from __future__ import annotations

import asyncio
import os
from importlib.metadata import version

import pytest
from packaging.version import Version

from opentelemetry import trace
from opentelemetry.trace import StatusCode

AGENT = "AGENT"
LLM = "LLM"
TOOL = "TOOL"
SPAN_KIND = "gen_ai.span.kind"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TTFT = "gen_ai.response.time_to_first_token"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LOCAL_COMPATIBLE_BASE_URL = "http://127.0.0.1:18765/v1"


def _new_agent(*, name, instructions, tools=None, local=False):
    from agent_framework_openai import OpenAIChatCompletionClient

    if local:
        from openai import AsyncOpenAI

        raw_client = AsyncOpenAI(
            api_key="replay-only-invalid-key",
            base_url=LOCAL_COMPATIBLE_BASE_URL,
            max_retries=0,
        )
        client = OpenAIChatCompletionClient(
            model="qwen-plus", async_client=raw_client
        )
    else:
        client = OpenAIChatCompletionClient(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY", "replay-only-invalid-key"),
            base_url=DASHSCOPE_BASE_URL,
        )
    agent = client.as_agent(
        name=name,
        instructions=instructions,
        tools=tools,
    )
    return client, agent


async def _close_client(client):
    await client.client.close()


def _maf_spans(exporter):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get(SPAN_KIND) in {AGENT, LLM, TOOL}
    ]


def _assert_closed_maf_trace(spans, *, llm_count=1, tool_count=0):
    """Assert MAF semantic spans form exactly one closed invocation tree."""
    by_kind = {
        kind: [
            span for span in spans if span.attributes.get(SPAN_KIND) == kind
        ]
        for kind in (AGENT, LLM, TOOL)
    }
    assert len(by_kind[AGENT]) == 1
    assert len(by_kind[LLM]) == llm_count
    assert len(by_kind[TOOL]) == tool_count

    root = by_kind[AGENT][0]
    assert root.parent is None
    assert {span.context.trace_id for span in spans} == {root.context.trace_id}

    span_ids = {span.context.span_id for span in spans}
    for span in spans:
        if span is root:
            continue
        assert span.parent is not None
        assert span.parent.span_id in span_ids
    return by_kind


def _assert_stream_ttft_if_framework_exposes_pull_context(llm_span):
    """MAF added the per-pull timing hook used for TTFT in version 1.10."""
    if Version(version("agent-framework-core")) < Version("1.10.0"):
        return
    assert llm_span.attributes[TTFT] >= 0


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("maf_runtime", [False], indirect=True)
async def test_non_streaming_content_disabled(maf_runtime):
    client, agent = _new_agent(
        name="maf_vcr_agent",
        instructions="Return only the requested deterministic token.",
    )
    try:
        response = await agent.run("Reply with exactly MAF_VCR_NONSTREAM_OK")
    finally:
        await _close_client(client)

    assert str(response) == "MAF_VCR_NONSTREAM_OK"
    spans = _maf_spans(maf_runtime)
    _assert_closed_maf_trace(spans)
    for span in spans:
        assert INPUT_MESSAGES not in span.attributes
        assert OUTPUT_MESSAGES not in span.attributes
        assert SYSTEM_INSTRUCTIONS not in span.attributes


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("maf_runtime", [True], indirect=True)
async def test_non_streaming_content_enabled(maf_runtime):
    client, agent = _new_agent(
        name="maf_vcr_agent",
        instructions="Return only the requested deterministic token.",
    )
    try:
        response = await agent.run("Reply with exactly MAF_VCR_NONSTREAM_OK")
    finally:
        await _close_client(client)

    assert str(response) == "MAF_VCR_NONSTREAM_OK"
    spans = _maf_spans(maf_runtime)
    _assert_closed_maf_trace(spans)
    for span in spans:
        assert "MAF_VCR_NONSTREAM_OK" in span.attributes[OUTPUT_MESSAGES]
        assert "Reply with exactly" in span.attributes[INPUT_MESSAGES]
        assert "deterministic token" in span.attributes[SYSTEM_INSTRUCTIONS]


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_stream_fully_consumed(maf_runtime):
    client, agent = _new_agent(
        name="maf_vcr_agent",
        instructions="Return only the requested deterministic token.",
    )
    chunks = []
    try:
        async for update in agent.run(
            "Reply with exactly MAF_VCR_STREAM_OK", stream=True
        ):
            chunks.append(str(update))
    finally:
        await _close_client(client)

    assert "".join(chunks) == "MAF_VCR_STREAM_OK"
    by_kind = _assert_closed_maf_trace(_maf_spans(maf_runtime))
    _assert_stream_ttft_if_framework_exposes_pull_context(by_kind[LLM][0])


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_stream_finalizes_when_final_response_requested_early(
    maf_runtime,
):
    """Validate deferred finalization, not provider transport cancellation."""
    client, agent = _new_agent(
        name="maf_vcr_agent",
        instructions="Return only the requested deterministic token.",
    )
    try:
        stream = agent.run("Reply with exactly MAF_VCR_STREAM_OK", stream=True)
        first_update = await stream.__anext__()
        response = await stream.get_final_response()
    finally:
        await _close_client(client)

    # MAF may emit an empty metadata update before the first text delta. The
    # contract here is that one update was pulled before deferred finalization.
    assert first_update is not None
    assert str(response) == "MAF_VCR_STREAM_OK"
    by_kind = _assert_closed_maf_trace(_maf_spans(maf_runtime))
    _assert_stream_ttft_if_framework_exposes_pull_context(by_kind[LLM][0])


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_tool_round_trip_has_one_closed_trace(maf_runtime):
    def get_weather(city: str) -> str:
        """Return deterministic weather for a city."""
        return f"{city}: sunny"

    client, agent = _new_agent(
        name="maf_vcr_tool_agent",
        instructions=(
            "Always call get_weather, then return exactly MAF_VCR_TOOL_OK."
        ),
        tools=[get_weather],
        local=True,
    )
    try:
        response = await agent.run("Check weather in Hangzhou.")
    finally:
        await _close_client(client)

    assert str(response) == "MAF_VCR_TOOL_OK"
    by_kind = _assert_closed_maf_trace(
        _maf_spans(maf_runtime), llm_count=2, tool_count=1
    )
    tool = by_kind[TOOL][0]
    assert tool.attributes["gen_ai.tool.name"] == "get_weather"
    assert tool.attributes["gen_ai.tool.call.result"] == "Hangzhou: sunny"


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_provider_error_marks_agent_and_llm_error(maf_runtime):
    from agent_framework.exceptions import ChatClientException

    client, agent = _new_agent(
        name="maf_vcr_error_agent",
        instructions="Return a response.",
        local=True,
    )
    try:
        with pytest.raises(
            ChatClientException, match="synthetic provider failure"
        ):
            await agent.run("Trigger the provider failure.")
    finally:
        await _close_client(client)

    spans = _maf_spans(maf_runtime)
    by_kind = _assert_closed_maf_trace(spans)
    assert by_kind[AGENT][0].status.status_code == StatusCode.ERROR
    assert by_kind[LLM][0].status.status_code == StatusCode.ERROR
    assert all(span.attributes.get("error.type") for span in spans)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_concurrent_invocations_keep_trace_and_context_isolated(
    maf_runtime,
):
    async def invoke(index):
        client, agent = _new_agent(
            name=f"maf_vcr_concurrent_{index}",
            instructions="Return only the requested deterministic token.",
        )
        try:
            return str(
                await agent.run("Reply with exactly MAF_VCR_NONSTREAM_OK")
            )
        finally:
            await _close_client(client)

    assert await asyncio.gather(invoke(1), invoke(2)) == [
        "MAF_VCR_NONSTREAM_OK",
        "MAF_VCR_NONSTREAM_OK",
    ]

    spans = _maf_spans(maf_runtime)
    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 2
    for trace_id in trace_ids:
        _assert_closed_maf_trace(
            [span for span in spans if span.context.trace_id == trace_id]
        )
    assert not trace.get_current_span().get_span_context().is_valid
