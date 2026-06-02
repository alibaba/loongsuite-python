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

"""AgentScope v2 instrumentation tests."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
from types import SimpleNamespace

import pytest

agentscope = pytest.importorskip("agentscope")
if not importlib.metadata.version("agentscope").startswith("2."):
    pytest.skip("AgentScope v2 tests require agentscope>=2,<3", allow_module_level=True)

from agentscope.agent import Agent  # noqa: E402
from agentscope.credential import DashScopeCredential  # noqa: E402
from agentscope.message import TextBlock, UserMsg  # noqa: E402
from agentscope.model import ChatResponse, DashScopeChatModel  # noqa: E402
from agentscope.tool import ToolResponse  # noqa: E402

from opentelemetry.instrumentation.agentscope._v2_middleware import (  # noqa: E402
    AgentScopeV2Middleware,
)
from opentelemetry.instrumentation.agentscope.package import (  # noqa: E402
    get_installed_instrumentation_dependencies,
)
from opentelemetry.trace.status import StatusCode  # noqa: E402


def test_v2_dependency_detection():
    assert get_installed_instrumentation_dependencies() == (
        "agentscope >= 2.0.0, < 3.0.0",
    )


def test_instrumentor_injects_v2_middleware(instrument):
    model = _make_model(stream=False)
    agent = Agent(
        name="middleware_probe",
        system_prompt="Reply briefly.",
        model=model,
    )

    assert any(
        isinstance(middleware, AgentScopeV2Middleware)
        for middleware in agent._reply_middlewares
    )
    assert any(
        isinstance(middleware, AgentScopeV2Middleware)
        for middleware in agent._model_call_middlewares
    )
    assert any(
        isinstance(middleware, AgentScopeV2Middleware)
        for middleware in agent._acting_middlewares
    )


def test_v2_uninstrument_removes_agent_patch(instrument):
    instrument.uninstrument()

    agent = Agent(
        name="uninstrument_probe",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )

    assert not any(
        isinstance(middleware, AgentScopeV2Middleware)
        for middleware in agent._reply_middlewares
    )
    assert not any(
        isinstance(middleware, AgentScopeV2Middleware)
        for middleware in agent._model_call_middlewares
    )


async def test_v2_existing_agent_middleware_noops_after_uninstrument(
    instrument, span_exporter
):
    agent = Agent(
        name="existing_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._model_call_middlewares)
    instrument.uninstrument()

    async def model_handler(**kwargs):
        del kwargs
        return ChatResponse(content=[TextBlock(text="ok")], is_last=True)

    response = await middleware.on_model_call(
        agent,
        {
            "current_model": agent.model,
            "messages": [UserMsg(name="user", content="hello")],
        },
        model_handler,
    )

    assert response.content
    assert not span_exporter.get_finished_spans()


async def test_v2_model_call_error_path(instrument, span_exporter):
    agent = Agent(
        name="error_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._model_call_middlewares)

    async def failing_handler(**kwargs):
        del kwargs
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await middleware.on_model_call(
            agent,
            {
                "current_model": agent.model,
                "messages": [UserMsg(name="user", content="fail")],
            },
            failing_handler,
        )

    span = _spans_by_operation(span_exporter.get_finished_spans(), "chat")[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "RuntimeError"


async def test_v2_streaming_model_call_error_path(instrument, span_exporter):
    agent = Agent(
        name="stream_error_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=True),
    )
    middleware = _middleware(agent._model_call_middlewares)

    async def failing_stream():
        yield ChatResponse(content=[TextBlock(text="partial")], is_last=False)
        raise RuntimeError("stream failed")

    async def stream_handler(**kwargs):
        del kwargs
        return failing_stream()

    stream = await middleware.on_model_call(
        agent,
        {
            "current_model": agent.model,
            "messages": [UserMsg(name="user", content="fail")],
        },
        stream_handler,
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        async for _ in stream:
            pass

    span = _spans_by_operation(span_exporter.get_finished_spans(), "chat")[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "RuntimeError"


async def test_v2_tool_acting_hook(instrument, span_exporter):
    agent = Agent(
        name="tool_agent",
        system_prompt="Use tools.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._acting_middlewares)
    tool_call = SimpleNamespace(
        name="lookup_weather",
        id="tool-call-1",
        input='{"city": "Hangzhou"}',
    )

    async def tool_handler(**kwargs):
        del kwargs
        yield ToolResponse(content=[TextBlock(text="sunny")])

    results = [
        item
        async for item in middleware.on_acting(
            agent,
            {"tool_call": tool_call},
            tool_handler,
        )
    ]

    assert results
    tool_span = _spans_by_operation(
        span_exporter.get_finished_spans(), "execute_tool"
    )[0]
    assert tool_span.attributes["gen_ai.tool.name"] == "lookup_weather"


@pytest.mark.vcr()
async def test_v2_agent_non_streaming_e2e(instrument, span_exporter):
    model = _make_model(stream=False)
    agent = Agent(
        name="non_stream_agent",
        system_prompt="Reply with exactly: OK",
        model=model,
    )

    msg = await agent.reply(UserMsg(name="user", content="Say OK."))

    assert msg.get_text_content()
    _assert_agent_and_llm_spans(span_exporter.get_finished_spans())


@pytest.mark.vcr()
async def test_v2_agent_streaming_e2e(instrument, span_exporter):
    model = _make_model(stream=True)
    agent = Agent(
        name="stream_agent",
        system_prompt="Reply with a short sentence.",
        model=model,
    )

    events = [
        event
        async for event in agent.reply_stream(
            UserMsg(name="user", content="Say hello in one sentence.")
        )
    ]

    assert events
    assert any(event.__class__.__name__ == "TextBlockDeltaEvent" for event in events)
    _assert_agent_and_llm_spans(span_exporter.get_finished_spans())


@pytest.mark.vcr()
async def test_v2_agent_concurrent_e2e(instrument, span_exporter):
    async def call_agent(idx: int):
        agent = Agent(
            name=f"concurrent_agent_{idx}",
            system_prompt="Reply with exactly one short sentence.",
            model=_make_model(stream=False),
        )
        return await agent.reply(
            UserMsg(name="user", content=f"Say OK for request {idx}.")
        )

    results = await asyncio.gather(call_agent(1), call_agent(2))

    assert all(result.get_text_content() for result in results)
    spans = span_exporter.get_finished_spans()
    agent_spans = _spans_by_operation(spans, "invoke_agent")
    llm_spans = _spans_by_operation(spans, "chat")
    assert len(agent_spans) == 2
    assert len(llm_spans) == 2
    agent_span_ids = {span.context.span_id for span in agent_spans}
    assert {span.parent.span_id for span in llm_spans} == agent_span_ids


def _make_model(stream: bool):
    return DashScopeChatModel(
        credential=DashScopeCredential(api_key=os.environ["DASHSCOPE_API_KEY"]),
        model="qwen-plus",
        parameters=DashScopeChatModel.Parameters(
            max_tokens=16,
            thinking_enable=False,
        ),
        stream=stream,
        max_retries=0,
    )


def _assert_agent_and_llm_spans(spans):
    assert _spans_by_operation(spans, "invoke_agent")
    assert _spans_by_operation(spans, "chat")


def _spans_by_operation(spans, operation_name):
    return [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == operation_name
    ]


def _middleware(middlewares):
    return next(
        middleware
        for middleware in middlewares
        if isinstance(middleware, AgentScopeV2Middleware)
    )
