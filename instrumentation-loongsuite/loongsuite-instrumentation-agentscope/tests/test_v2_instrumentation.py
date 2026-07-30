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
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest

agentscope = pytest.importorskip("agentscope")
if not importlib.metadata.version("agentscope").startswith("2."):
    pytest.skip(
        "AgentScope v2 tests require agentscope>=2,<3", allow_module_level=True
    )

from agentscope.agent import Agent  # noqa: E402
from agentscope.credential import DashScopeCredential  # noqa: E402
from agentscope.message import (  # noqa: E402
    Msg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    UserMsg,
)
from agentscope.model import ChatResponse, DashScopeChatModel  # noqa: E402
from agentscope.tool import ToolResponse  # noqa: E402

from opentelemetry import trace as trace_api  # noqa: E402
from opentelemetry.instrumentation.agentscope._v2_middleware import (  # noqa: E402
    AgentScopeV2Middleware,
    _message_to_input,
)
from opentelemetry.instrumentation.agentscope.package import (  # noqa: E402
    get_installed_instrumentation_dependencies,
)
from opentelemetry.semconv._incubating.attributes import (  # noqa: E402
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace.status import StatusCode  # noqa: E402
from opentelemetry.util.genai.utils import gen_ai_json_dumps  # noqa: E402


def test_v2_dependency_detection():
    assert get_installed_instrumentation_dependencies() == (
        "agentscope >= 2.0.0, < 3.0.0",
    )


def test_v2_tool_result_message_content_is_jsonable():
    msg = Msg(
        name="assistant",
        role="assistant",
        content=[
            ToolResultBlock(
                id="tool-call-content",
                name="lookup_weather",
                output=[TextBlock(text="sunny")],
            )
        ],
    )

    input_message = _message_to_input(msg)

    assert (
        gen_ai_json_dumps([asdict(input_message)])
        == '[{"role":"assistant","parts":[{"response":[{"content":"sunny","type":"text"}],"id":"tool-call-content","type":"tool_call_response"}]}]'
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
        for middleware in agent._reasoning_middlewares
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


async def test_v2_streaming_model_call_starts_llm_span_before_model_handler(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="stream_suppression_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=True),
    )
    middleware = _middleware(agent._model_call_middlewares)
    observed_current_span_ids = []
    consumer_current_span_ids = []

    async def stream_handler(**kwargs):
        del kwargs
        observed_current_span_ids.append(
            trace_api.get_current_span().get_span_context().span_id
        )

        async def stream():
            observed_current_span_ids.append(
                trace_api.get_current_span().get_span_context().span_id
            )
            yield ChatResponse(
                content=[TextBlock(text="partial")],
                is_last=False,
            )
            observed_current_span_ids.append(
                trace_api.get_current_span().get_span_context().span_id
            )
            yield ChatResponse(
                content=[TextBlock(text="done")],
                is_last=True,
            )

        return stream()

    stream = await middleware.on_model_call(
        agent,
        {
            "current_model": agent.model,
            "messages": [UserMsg(name="user", content="hello")],
        },
        stream_handler,
    )

    async for _ in stream:
        consumer_current_span_ids.append(
            trace_api.get_current_span().get_span_context().span_id
        )

    spans = _spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert len(spans) == 1
    llm_span_id = spans[0].context.span_id
    assert observed_current_span_ids == [llm_span_id, llm_span_id, llm_span_id]
    assert consumer_current_span_ids == [0, 0]
    assert (
        trace_api.get_current_span().get_span_context().span_id != llm_span_id
    )


async def test_v2_streaming_model_aclose_is_normal_across_tasks(
    instrument,
    span_exporter,
    caplog,
):
    caplog.set_level("DEBUG", logger="opentelemetry.util.genai.handler")
    agent = Agent(
        name="stream_close_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=True),
    )
    middleware = _middleware(agent._model_call_middlewares)
    expected = ChatResponse(
        content=[TextBlock(text="partial")],
        is_last=False,
    )
    closed = 0

    async def business_stream():
        nonlocal closed
        try:
            yield expected
            yield ChatResponse(
                content=[TextBlock(text="unused")],
                is_last=True,
            )
        finally:
            closed += 1

    async def stream_handler(**kwargs):
        del kwargs
        return business_stream()

    stream = await asyncio.create_task(
        middleware.on_model_call(
            agent,
            {
                "current_model": agent.model,
                "messages": [UserMsg(name="user", content="hello")],
            },
            stream_handler,
        )
    )
    assert not trace_api.get_current_span().get_span_context().is_valid
    assert await _heartbeat_next(stream) is expected
    await asyncio.create_task(stream.aclose())

    [span] = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "chat",
    )
    assert closed == 1
    assert span.status.status_code == StatusCode.UNSET
    assert "error.type" not in span.attributes
    assert "gen_ai.response.finish_reasons" not in span.attributes
    assert not any(
        "Context detach failed" in record.getMessage()
        or "Failed to detach context" in record.getMessage()
        for record in caplog.records
    )
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_streaming_model_aclose_error_preserves_original_exception(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="stream_close_error_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=True),
    )
    middleware = _middleware(agent._model_call_middlewares)
    close_error = ValueError("business close failure")

    async def business_stream():
        try:
            yield ChatResponse(
                content=[TextBlock(text="partial")],
                is_last=False,
            )
        finally:
            raise close_error

    async def stream_handler(**kwargs):
        del kwargs
        return business_stream()

    stream = await asyncio.create_task(
        middleware.on_model_call(
            agent,
            {
                "current_model": agent.model,
                "messages": [UserMsg(name="user", content="hello")],
            },
            stream_handler,
        )
    )
    await _heartbeat_next(stream)
    try:
        await asyncio.create_task(stream.aclose())
    except ValueError as exc:
        assert exc is close_error
    else:
        pytest.fail("business close error was not raised")

    [span] = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "chat",
    )
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "ValueError"
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_streaming_model_cancellation_cleans_cross_task_context(
    instrument,
    span_exporter,
    caplog,
):
    caplog.set_level("DEBUG", logger="opentelemetry.util.genai.handler")
    agent = Agent(
        name="stream_cancel_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=True),
    )
    middleware = _middleware(agent._model_call_middlewares)
    started = asyncio.Event()
    never = asyncio.Event()

    async def business_stream():
        started.set()
        await never.wait()
        yield ChatResponse(
            content=[TextBlock(text="unreachable")],
            is_last=True,
        )

    async def stream_handler(**kwargs):
        del kwargs
        return business_stream()

    stream = await asyncio.create_task(
        middleware.on_model_call(
            agent,
            {
                "current_model": agent.model,
                "messages": [UserMsg(name="user", content="hello")],
            },
            stream_handler,
        )
    )
    task = asyncio.create_task(stream.__anext__())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    [span] = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "chat",
    )
    assert task.cancelled()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "CancelledError"
    assert not any(
        "Context detach failed" in record.getMessage()
        or "Failed to detach context" in record.getMessage()
        for record in caplog.records
    )
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_streaming_model_call_captures_input_and_output_content(
    instrument_with_content,
    span_exporter,
):
    agent = Agent(
        name="stream_content_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=True),
    )
    middleware = _middleware(agent._model_call_middlewares)

    async def stream_handler(**kwargs):
        del kwargs

        async def stream():
            yield ChatResponse(
                content=[TextBlock(text="partial")],
                is_last=False,
            )
            yield ChatResponse(
                content=[TextBlock(text="done")],
                is_last=True,
            )

        return stream()

    stream = await middleware.on_model_call(
        agent,
        {
            "current_model": agent.model,
            "messages": [UserMsg(name="user", content="hello")],
        },
        stream_handler,
    )

    async for _ in stream:
        pass

    span = _spans_by_operation(span_exporter.get_finished_spans(), "chat")[0]
    input_messages = json.loads(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    output_messages = json.loads(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == "hello"
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == "done"


async def test_v2_reply_stream_survives_cross_task_heartbeat(
    instrument,
    tracer_provider,
    span_exporter,
):
    agent = Agent(
        name="cross_task_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    child_tracer = trace_api.get_tracer(
        "agentscope-cross-task-test",
        tracer_provider=tracer_provider,
    )
    expected = Msg(
        name="assistant",
        role="assistant",
        content=[TextBlock(text="done")],
    )

    async def reply_handler(**kwargs):
        del kwargs
        with child_tracer.start_as_current_span("reply-child"):
            pass
        yield expected

    stream = middleware.on_reply(agent, {"inputs": []}, reply_handler)

    assert await _heartbeat_next(stream) is expected
    with pytest.raises(StopAsyncIteration):
        await _heartbeat_next(stream)

    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "invoke_agent",
    )
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code == StatusCode.UNSET
    [child_span] = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "reply-child"
    ]
    assert child_span.parent.span_id == agent_spans[0].context.span_id
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_reply_stream_preserves_cross_task_business_error(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="cross_task_error_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    expected_error = RuntimeError("reply failed")

    async def reply_handler(**kwargs):
        del kwargs
        yield Msg(
            name="assistant",
            role="assistant",
            content=[TextBlock(text="partial")],
        )
        raise expected_error

    stream = middleware.on_reply(agent, {"inputs": []}, reply_handler)

    await _heartbeat_next(stream)
    with pytest.raises(RuntimeError, match="reply failed") as caught:
        await _heartbeat_next(stream)

    assert caught.value is expected_error
    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "invoke_agent",
    )
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code == StatusCode.ERROR


async def test_v2_reply_start_failure_preserves_business_stream(
    instrument,
    span_exporter,
    monkeypatch,
):
    agent = Agent(
        name="start_failure_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    expected = Msg(
        name="assistant",
        role="assistant",
        content=[TextBlock(text="business result")],
    )
    handler = middleware._handler()

    def start_invoke_agent(*args, **kwargs):
        del args, kwargs
        raise ValueError("probe start failure")

    monkeypatch.setattr(handler, "start_invoke_agent", start_invoke_agent)

    async def reply_handler(**kwargs):
        del kwargs
        yield expected

    results = [
        item
        async for item in middleware.on_reply(
            agent,
            {"inputs": []},
            reply_handler,
        )
    ]

    assert results == [expected]
    assert not span_exporter.get_finished_spans()
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_react_start_failure_preserves_reasoning_stream(
    instrument,
    span_exporter,
    monkeypatch,
):
    agent = Agent(
        name="react_start_failure_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    handler = middleware._handler()
    expected = Msg(
        name="assistant",
        role="assistant",
        content=[TextBlock(text="business result")],
    )

    def start_react_step(*args, **kwargs):
        del args, kwargs
        raise ValueError("probe step start failure")

    monkeypatch.setattr(handler, "start_react_step", start_react_step)

    async def reasoning_handler(**kwargs):
        del kwargs
        yield expected

    async def reply_handler(**kwargs):
        del kwargs
        async for item in middleware.on_reasoning(
            agent,
            {},
            reasoning_handler,
        ):
            yield item

    results = [
        item
        async for item in middleware.on_reply(
            agent,
            {"inputs": []},
            reply_handler,
        )
    ]

    assert results == [expected]
    assert not _spans_by_operation(
        span_exporter.get_finished_spans(),
        "react",
    )
    assert _spans_by_operation(
        span_exporter.get_finished_spans(),
        "invoke_agent",
    )
    assert not middleware._reply_states


async def test_v2_reasoning_error_preserves_identity_and_fails_step(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="react_error_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    business_error = RuntimeError("reasoning failed")

    async def reasoning_handler(**kwargs):
        del kwargs
        yield SimpleNamespace(type="thinking_block_delta")
        raise business_error

    async def reply_handler(**kwargs):
        del kwargs
        async for item in middleware.on_reasoning(
            agent,
            {},
            reasoning_handler,
        ):
            yield item

    stream = middleware.on_reply(agent, {"inputs": []}, reply_handler)
    await _heartbeat_next(stream)
    try:
        await _heartbeat_next(stream)
    except RuntimeError as exc:
        assert exc is business_error
    else:
        pytest.fail("business error was not raised")

    [react_span] = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "react",
    )
    assert react_span.status.status_code == StatusCode.ERROR
    assert react_span.attributes["error.type"] == "RuntimeError"
    assert not middleware._reply_states


async def test_v2_reply_finish_failure_does_not_replace_business_result(
    instrument,
    span_exporter,
    monkeypatch,
):
    agent = Agent(
        name="finish_failure_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    expected = Msg(
        name="assistant",
        role="assistant",
        content=[TextBlock(text="business result")],
    )
    handler = middleware._handler()

    def stop_invoke_agent(*args, **kwargs):
        del args, kwargs
        raise ValueError("probe finish failure")

    monkeypatch.setattr(handler, "stop_invoke_agent", stop_invoke_agent)

    async def reply_handler(**kwargs):
        del kwargs
        yield expected

    results = [
        item
        async for item in middleware.on_reply(
            agent,
            {"inputs": []},
            reply_handler,
        )
    ]

    assert results == [expected]
    assert len(span_exporter.get_finished_spans()) == 1
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_reply_fail_callback_preserves_original_business_error(
    instrument,
    span_exporter,
    monkeypatch,
):
    agent = Agent(
        name="fail_callback_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    handler = middleware._handler()
    business_error = RuntimeError("business failure")

    def fail_invoke_agent(*args, **kwargs):
        del args, kwargs
        raise ValueError("probe fail callback failure")

    monkeypatch.setattr(handler, "fail_invoke_agent", fail_invoke_agent)

    async def reply_handler(**kwargs):
        del kwargs
        yield Msg(
            name="assistant",
            role="assistant",
            content=[TextBlock(text="partial")],
        )
        raise business_error

    stream = middleware.on_reply(agent, {"inputs": []}, reply_handler)
    await _heartbeat_next(stream)
    try:
        await _heartbeat_next(stream)
    except RuntimeError as exc:
        assert exc is business_error
    else:
        pytest.fail("business error was not raised")

    assert len(span_exporter.get_finished_spans()) == 1
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_reply_stream_preserves_cross_task_cancellation(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="cross_task_cancel_agent",
        system_prompt="Reply briefly.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    resumed = asyncio.Event()

    async def reply_handler(**kwargs):
        del kwargs
        yield Msg(
            name="assistant",
            role="assistant",
            content=[TextBlock(text="partial")],
        )
        resumed.set()
        await asyncio.Event().wait()

    stream = middleware.on_reply(agent, {"inputs": []}, reply_handler)
    await _heartbeat_next(stream)

    pending_next = asyncio.create_task(_heartbeat_next(stream))
    await asyncio.wait_for(resumed.wait(), timeout=1)
    pending_next.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_next

    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "invoke_agent",
    )
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code == StatusCode.ERROR
    assert agent_spans[0].attributes["error.type"] == "CancelledError"


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
    assert tool_span.attributes["gen_ai.tool.type"] == "function"


async def test_v2_acting_aclose_is_normal_across_tasks(
    instrument,
    span_exporter,
    caplog,
):
    caplog.set_level("DEBUG", logger="opentelemetry.util.genai.handler")
    agent = Agent(
        name="cross_task_close_agent",
        system_prompt="Use tools.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._acting_middlewares)
    tool_call = SimpleNamespace(
        name="successful_tool",
        id="tool-call-close",
        input="{}",
    )
    expected = ToolResponse(content=[TextBlock(text="success")])
    closed = 0

    async def tool_handler(**kwargs):
        nonlocal closed
        del kwargs
        try:
            yield expected
            yield ToolResponse(content=[TextBlock(text="unused")])
        finally:
            closed += 1

    stream = middleware.on_acting(
        agent,
        {"tool_call": tool_call},
        tool_handler,
    )
    assert await _heartbeat_next(stream) is expected
    await asyncio.create_task(stream.aclose())

    spans = span_exporter.get_finished_spans()
    [tool_span] = _spans_by_operation(spans, "execute_tool")
    [react_span] = _spans_by_operation(spans, "react")
    assert closed == 1
    assert tool_span.status.status_code == StatusCode.UNSET
    assert react_span.status.status_code == StatusCode.UNSET
    assert "error.type" not in tool_span.attributes
    assert "error.type" not in react_span.attributes
    assert react_span.attributes["gen_ai.react.finish_reason"] == "tool_calls"
    assert not any(
        "Context detach failed" in record.getMessage()
        for record in caplog.records
    )
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_acting_aclose_error_preserves_original_exception(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="cross_task_close_error_agent",
        system_prompt="Use tools.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._acting_middlewares)
    tool_call = SimpleNamespace(
        name="failing_close_tool",
        id="tool-call-close-error",
        input="{}",
    )
    close_error = ValueError("business close failure")

    async def tool_handler(**kwargs):
        del kwargs
        try:
            yield ToolResponse(content=[TextBlock(text="partial")])
        finally:
            raise close_error

    stream = middleware.on_acting(
        agent,
        {"tool_call": tool_call},
        tool_handler,
    )
    await _heartbeat_next(stream)
    try:
        await asyncio.create_task(stream.aclose())
    except ValueError as exc:
        assert exc is close_error
    else:
        pytest.fail("business close error was not raised")

    spans = span_exporter.get_finished_spans()
    [tool_span] = _spans_by_operation(spans, "execute_tool")
    [react_span] = _spans_by_operation(spans, "react")
    assert tool_span.status.status_code == StatusCode.ERROR
    assert react_span.status.status_code == StatusCode.ERROR
    assert tool_span.attributes["error.type"] == "ValueError"
    assert react_span.attributes["error.type"] == "ValueError"
    assert "gen_ai.react.finish_reason" not in react_span.attributes
    assert not trace_api.get_current_span().get_span_context().is_valid


async def test_v2_tool_result_content_capture(
    instrument_with_content,
    span_exporter,
):
    agent = Agent(
        name="tool_content_agent",
        system_prompt="Use tools.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._acting_middlewares)
    tool_call = SimpleNamespace(
        name="lookup_weather",
        id="tool-call-content",
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
    assert tool_span.attributes["gen_ai.tool.call.result"] == (
        '[{"content":"sunny","type":"text"}]'
    )


async def test_v2_react_many_tools_telemetry(instrument, span_exporter):
    agent = Agent(
        name="react_tool_agent",
        system_prompt="Use tools.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._acting_middlewares)

    for idx, name in enumerate(
        [
            "lookup_weather",
            "search_docs",
            "calculate_total",
            "write_summary",
        ],
        start=1,
    ):
        tool_call = SimpleNamespace(
            name=name,
            id=f"tool-call-{idx}",
            input=f'{{"idx": {idx}}}',
        )

        async def tool_handler(**kwargs):
            del kwargs
            yield ToolResponse(content=[TextBlock(text=f"result {idx}")])

        agent.state.cur_iter = idx - 1
        results = [
            item
            async for item in middleware.on_acting(
                agent,
                {"tool_call": tool_call},
                tool_handler,
            )
        ]
        assert results

    spans = span_exporter.get_finished_spans()
    react_spans = _spans_by_operation(spans, "react")
    tool_spans = _spans_by_operation(spans, "execute_tool")

    assert [span.attributes["gen_ai.react.round"] for span in react_spans] == [
        1,
        2,
        3,
        4,
    ]
    assert {span.attributes["gen_ai.tool.name"] for span in tool_spans} == {
        "lookup_weather",
        "search_docs",
        "calculate_total",
        "write_summary",
    }
    assert {span.attributes["gen_ai.tool.type"] for span in tool_spans} == {
        "function"
    }
    react_span_ids = {span.context.span_id for span in react_spans}
    assert {span.parent.span_id for span in tool_spans} == react_span_ids


async def test_v2_react_concurrent_tools_share_agent_iteration(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="concurrent_tool_agent",
        system_prompt="Use tools.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._acting_middlewares)
    agent.state.cur_iter = 2

    async def call_tool(idx: int):
        tool_call = SimpleNamespace(
            name=f"tool_{idx}",
            id=f"tool-call-{idx}",
            input=f'{{"idx": {idx}}}',
        )

        async def tool_handler(**kwargs):
            del kwargs
            await asyncio.sleep(0)
            yield ToolResponse(content=[TextBlock(text=f"result {idx}")])

        return [
            item
            async for item in middleware.on_acting(
                agent,
                {"tool_call": tool_call},
                tool_handler,
            )
        ]

    results = await asyncio.gather(*(call_tool(idx) for idx in range(4)))

    assert all(results)
    react_spans = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "react",
    )
    assert len(react_spans) == 4
    assert {span.attributes["gen_ai.react.round"] for span in react_spans} == {
        3
    }


async def test_v2_react_step_parents_reasoning_llm_and_concurrent_tools(
    instrument,
    span_exporter,
):
    agent = Agent(
        name="react_hierarchy_agent",
        system_prompt="Use tools.",
        model=_make_model(stream=False),
    )
    middleware = _middleware(agent._reply_middlewares)
    tool_calls = [
        ToolCallBlock(
            id=f"tool-call-{idx}",
            name=f"tool_{idx}",
            input=json.dumps({"idx": idx}),
        )
        for idx in range(2)
    ]

    async def model_handler(**kwargs):
        del kwargs
        return ChatResponse(content=tool_calls, is_last=True)

    async def reasoning_handler(**kwargs):
        del kwargs
        await middleware.on_model_call(
            agent,
            {
                "current_model": agent.model,
                "messages": [UserMsg(name="user", content="use tools")],
            },
            model_handler,
        )
        for tool_call in tool_calls:
            yield SimpleNamespace(type="TOOL_CALL_START")
            yield SimpleNamespace(type="TOOL_CALL_END")
        # Some framework versions may emit a completed message after the
        # individual tool-call events.
        yield Msg(
            name="assistant",
            role="assistant",
            content=tool_calls,
        )

    async def call_tool(tool_call):
        async def tool_handler(**kwargs):
            del kwargs
            await asyncio.sleep(0)
            yield ToolResponse(
                content=[TextBlock(text=f"result {tool_call.name}")]
            )

        return [
            item
            async for item in middleware.on_acting(
                agent,
                {"tool_call": tool_call},
                tool_handler,
            )
        ]

    expected = Msg(
        name="assistant",
        role="assistant",
        content=[TextBlock(text="done")],
    )

    async def final_model_handler(**kwargs):
        del kwargs
        return ChatResponse(
            content=[TextBlock(text="done")],
            is_last=True,
        )

    async def final_reasoning_handler(**kwargs):
        del kwargs
        await middleware.on_model_call(
            agent,
            {
                "current_model": agent.model,
                "messages": [UserMsg(name="user", content="finish")],
            },
            final_model_handler,
        )
        yield expected

    async def reply_handler(**kwargs):
        del kwargs
        async for item in middleware.on_reasoning(
            agent,
            {},
            reasoning_handler,
        ):
            yield item
        assert all(await asyncio.gather(*(call_tool(tc) for tc in tool_calls)))
        agent.state.cur_iter = 1
        async for item in middleware.on_reasoning(
            agent,
            {},
            final_reasoning_handler,
        ):
            yield item

    stream = middleware.on_reply(agent, {"inputs": []}, reply_handler)
    while True:
        try:
            await _heartbeat_next(stream)
        except StopAsyncIteration:
            break

    spans = span_exporter.get_finished_spans()
    [agent_span] = _spans_by_operation(spans, "invoke_agent")
    react_spans = _spans_by_operation(spans, "react")
    llm_spans = _spans_by_operation(spans, "chat")
    tool_spans = _spans_by_operation(spans, "execute_tool")

    assert len(react_spans) == 2
    assert len(llm_spans) == 2
    assert {span.parent.span_id for span in react_spans} == {
        agent_span.context.span_id
    }
    react_by_round = {
        span.attributes["gen_ai.react.round"]: span for span in react_spans
    }
    llm_parent_ids = {span.parent.span_id for span in llm_spans}
    assert llm_parent_ids == {span.context.span_id for span in react_spans}
    assert len(tool_spans) == 2
    assert {span.parent.span_id for span in tool_spans} == {
        react_by_round[1].context.span_id
    }
    assert (
        react_by_round[1].attributes["gen_ai.react.finish_reason"]
        == "tool_calls"
    )
    assert react_by_round[2].attributes["gen_ai.react.finish_reason"] == "stop"


async def test_v2_skill_viewer_tool_captures_skill_metadata(
    instrument,
    span_exporter,
    tmp_path,
):
    agent = Agent(
        name="skill_agent",
        system_prompt="Load a skill.",
        model=_make_model(stream=False),
    )
    skill_dir = tmp_path / "workspaces" / "demo" / "skills" / "code-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: code-review\n"
        "description: Review source code\n"
        "version: 1.2.3\n"
        "---\n"
        "Review the requested source code.\n",
        encoding="utf-8",
    )
    agent.toolkit._qp_skills = {
        "code-review": {
            "dir": str(skill_dir),
        }
    }
    middleware = _middleware(agent._acting_middlewares)
    tool_call = SimpleNamespace(
        name="Skill",
        id="skill-tool-call",
        input='{"skill": "code-review"}',
    )

    async def tool_handler(**kwargs):
        del kwargs
        yield ToolResponse(content=[TextBlock(text="skill markdown")])

    results = [
        item
        async for item in middleware.on_acting(
            agent,
            {"tool_call": tool_call},
            tool_handler,
        )
    ]

    assert results
    [tool_span] = _spans_by_operation(
        span_exporter.get_finished_spans(),
        "execute_tool",
    )
    assert tool_span.attributes["gen_ai.skill.name"] == "code-review"
    assert (
        tool_span.attributes["gen_ai.skill.id"] == "workspace:demo:code-review"
    )
    assert tool_span.attributes["gen_ai.skill.version"] == "1.2.3"


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
    assert any(
        event.__class__.__name__ == "TextBlockDeltaEvent" for event in events
    )
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
    react_spans = _spans_by_operation(spans, "react")
    assert len(agent_spans) == 2
    assert len(llm_spans) == 2
    assert len(react_spans) == 2
    agent_span_ids = {span.context.span_id for span in agent_spans}
    assert {span.parent.span_id for span in react_spans} == agent_span_ids
    react_span_ids = {span.context.span_id for span in react_spans}
    assert {span.parent.span_id for span in llm_spans} == react_span_ids


def _make_model(stream: bool):
    return DashScopeChatModel(
        credential=DashScopeCredential(
            api_key=os.environ["DASHSCOPE_API_KEY"]
        ),
        model="qwen-plus",
        parameters=DashScopeChatModel.Parameters(
            max_tokens=16,
            thinking_enable=False,
        ),
        stream=stream,
        max_retries=0,
    )


def _assert_agent_and_llm_spans(spans):
    [agent_span] = _spans_by_operation(spans, "invoke_agent")
    [react_span] = _spans_by_operation(spans, "react")
    [llm_span] = _spans_by_operation(spans, "chat")
    assert react_span.parent.span_id == agent_span.context.span_id
    assert llm_span.parent.span_id == react_span.context.span_id


def _spans_by_operation(spans, operation_name):
    return [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == operation_name
    ]


async def _heartbeat_next(stream):
    """Advance an async generator in a fresh Task, like QwenPaw heartbeat."""

    async def advance():
        return await stream.__anext__()

    return await asyncio.wait_for(asyncio.create_task(advance()), timeout=1)


def _middleware(middlewares):
    return next(
        middleware
        for middleware in middlewares
        if isinstance(middleware, AgentScopeV2Middleware)
    )
