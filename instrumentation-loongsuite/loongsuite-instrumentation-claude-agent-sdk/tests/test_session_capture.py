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

"""Session propagation tests for Claude Agent SDK instrumentation."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.instrumentation.claude_agent_sdk import (
    patch as claude_patch,
)
from opentelemetry.instrumentation.claude_agent_sdk.patch import (
    _process_agent_invocation_stream,
    wrap_claude_client_query,
    wrap_claude_client_receive_response,
    wrap_query,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)


class SystemMessage:
    def __init__(
        self,
        session_id: str,
        model: str | None = None,
        cwd: str | None = None,
    ):
        self.subtype = "init"
        self.data = {"session_id": session_id}
        if model is not None:
            self.data["model"] = model
        if cwd is not None:
            self.data["cwd"] = cwd


class StreamEvent:
    def __init__(
        self,
        session_id: str | None = None,
        event_session_id: str | None = None,
    ):
        self.session_id = session_id
        self.event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "partial"},
        }
        if event_session_id:
            self.event["session_id"] = event_session_id


class AssistantMessage:
    def __init__(self, content: list[Any], model: str = "claude-sonnet"):
        self.content = content
        self.model = model
        self.parent_tool_use_id = None
        self.error = None


class UserMessage:
    def __init__(
        self,
        content: list[Any],
        tool_use_result: dict[str, Any] | None = None,
    ):
        self.content = content
        self.tool_use_result = tool_use_result
        self.uuid = None
        self.parent_tool_use_id = None


class ResultMessage:
    def __init__(
        self,
        session_id: str | None = None,
        model_usage: dict[str, Any] | None = None,
    ):
        self.subtype = "success"
        self.duration_ms = 10
        self.duration_api_ms = 8
        self.is_error = False
        self.num_turns = 1
        self.session_id = session_id
        self.total_cost_usd = 0.01
        self.usage = {"input_tokens": 11, "output_tokens": 7}
        self.result = "done"
        self.structured_output = None
        self.model_usage = model_usage


class TextBlock:
    def __init__(self, text: str):
        self.text = text


class ToolUseBlock:
    def __init__(
        self, tool_use_id: str, name: str, tool_input: dict[str, Any]
    ):
        self.id = tool_use_id
        self.name = name
        self.input = tool_input


class ToolResultBlock:
    def __init__(self, tool_use_id: str, content: str):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = False


async def _stream(messages):
    for message in messages:
        yield message


async def _failing_stream(messages, exc):
    for message in messages:
        yield message
    raise exc


async def _cancelled_stream(session_id):
    yield SystemMessage(session_id)
    await asyncio.sleep(60)


def _spans_by_operation(spans, operation):
    return [
        span
        for span in spans
        if dict(span.attributes or {}).get(
            GenAIAttributes.GEN_AI_OPERATION_NAME
        )
        == operation
    ]


async def _run_stream(tracer_provider, messages, session_id=None):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=_stream(messages),
        handler=handler,
        model="claude-sonnet",
        prompt="inspect the project",
        session_id=session_id,
    ):
        pass


async def _read_until_result_without_exhausting(
    handler, messages, session_id=None
):
    stream = _process_agent_invocation_stream(
        wrapped_stream=_stream(messages),
        handler=handler,
        model="claude-sonnet",
        prompt="inspect the project",
        session_id=session_id,
    )
    while True:
        message = await stream.__anext__()
        if isinstance(message, ResultMessage):
            return stream


@pytest.mark.asyncio
async def test_entry_baggage_session_overrides_claude_session(
    tracer_provider, span_exporter
):
    messages = [
        SystemMessage("claude-session"),
        AssistantMessage([TextBlock("I will read a file.")]),
        AssistantMessage(
            [ToolUseBlock("toolu_1", "Read", {"file_path": "README.md"})]
        ),
        UserMessage([ToolResultBlock("toolu_1", "README content")]),
        ResultMessage("claude-session"),
    ]
    ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "entry-session")
    ctx = baggage.set_baggage(GEN_AI_USER_ID, "entry-user", ctx)
    token = otel_context.attach(ctx)
    try:
        await _run_stream(tracer_provider, messages)
    finally:
        otel_context.detach(token)

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]
    tool_span = _spans_by_operation(spans, "execute_tool")[0]

    for span in (agent_span, llm_span, tool_span):
        assert span.attributes[GEN_AI_SESSION_ID] == "entry-session"
        assert span.attributes[GEN_AI_USER_ID] == "entry-user"


@pytest.mark.asyncio
async def test_system_session_propagates_to_agent_llm_and_tool(
    tracer_provider, span_exporter
):
    messages = [
        SystemMessage("sess-system"),
        AssistantMessage([TextBlock("I will read a file.")]),
        AssistantMessage(
            [ToolUseBlock("toolu_1", "Read", {"file_path": "README.md"})]
        ),
        UserMessage([ToolResultBlock("toolu_1", "README content")]),
        ResultMessage("sess-system"),
    ]

    await _run_stream(tracer_provider, messages)

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]
    tool_span = _spans_by_operation(spans, "execute_tool")[0]

    assert agent_span.attributes[GEN_AI_SESSION_ID] == "sess-system"
    assert llm_span.attributes[GEN_AI_SESSION_ID] == "sess-system"
    assert tool_span.attributes[GEN_AI_SESSION_ID] == "sess-system"


@pytest.mark.asyncio
async def test_assistant_message_model_fills_unknown_request_model(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=_stream(
            [
                SystemMessage("sess-model"),
                AssistantMessage(
                    [TextBlock("answer")],
                    model="claude-actual-model",
                ),
                ResultMessage("sess-model"),
            ]
        ),
        handler=handler,
        model="unknown",
        prompt="inspect the model",
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]

    assert (
        agent_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-actual-model"
    )
    assert (
        llm_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-actual-model"
    )
    assert llm_span.name == "chat claude-actual-model"


@pytest.mark.asyncio
async def test_system_message_model_fills_unknown_request_model(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=_stream(
            [
                SystemMessage(
                    "sess-system-model", model="claude-system-model"
                ),
                AssistantMessage([TextBlock("answer")], model=""),
                ResultMessage("sess-system-model"),
            ]
        ),
        handler=handler,
        model="unknown",
        prompt="inspect the model",
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]

    assert (
        agent_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-system-model"
    )
    assert (
        llm_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-system-model"
    )
    assert llm_span.name == "chat claude-system-model"


@pytest.mark.asyncio
async def test_result_model_usage_fills_unknown_request_model_attribute(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=_stream(
            [
                AssistantMessage([TextBlock("answer")], model=""),
                ResultMessage(
                    "sess-result-model",
                    model_usage={"claude-result-model": {}},
                ),
            ]
        ),
        handler=handler,
        model="unknown",
        prompt="inspect the model",
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]

    assert (
        agent_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-result-model"
    )
    assert (
        llm_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "claude-result-model"
    )


@pytest.mark.asyncio
async def test_result_model_usage_with_multiple_models_keeps_unknown_model(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=_stream(
            [
                AssistantMessage([TextBlock("answer")], model=""),
                ResultMessage(
                    "sess-multi-model",
                    model_usage={"model-a": {}, "model-b": {}},
                ),
            ]
        ),
        handler=handler,
        model="unknown",
        prompt="ambiguous model usage should not be guessed",
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]

    assert (
        agent_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "unknown"
    )
    assert (
        llm_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "unknown"
    )


@pytest.mark.asyncio
async def test_tool_use_without_text_does_not_create_synthetic_llm_span(
    tracer_provider, span_exporter
):
    await _run_stream(
        tracer_provider,
        [
            SystemMessage("sess-tool-only"),
            AssistantMessage(
                [ToolUseBlock("toolu_1", "Read", {"file_path": "README.md"})]
            ),
            AssistantMessage(
                [ToolUseBlock("toolu_2", "Glob", {"pattern": "*.md"})]
            ),
            UserMessage(
                [
                    ToolResultBlock("toolu_1", "README content"),
                    ToolResultBlock("toolu_2", "README.md"),
                ]
            ),
            AssistantMessage([TextBlock("done")]),
            ResultMessage("sess-tool-only"),
        ],
    )

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_spans = sorted(
        _spans_by_operation(spans, "chat"), key=lambda span: span.start_time
    )
    tool_spans = sorted(
        _spans_by_operation(spans, "execute_tool"),
        key=lambda span: span.start_time,
    )

    assert len(llm_spans) == 1
    assert len(tool_spans) == 2
    assert all(
        tool_span.parent.span_id == agent_span.context.span_id
        for tool_span in tool_spans
    )
    assert llm_spans[0].parent.span_id == agent_span.context.span_id
    assert tool_spans[-1].start_time < llm_spans[0].start_time
    assert llm_spans[0].attributes[
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ] == ("stop",)


@pytest.mark.asyncio
async def test_result_session_sets_agent_span_when_no_init_message(
    tracer_provider, span_exporter
):
    messages = [
        AssistantMessage([TextBlock("answer")]),
        ResultMessage("sess-result"),
    ]

    await _run_stream(tracer_provider, messages)

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]
    assert agent_span.attributes[GEN_AI_SESSION_ID] == "sess-result"
    assert llm_span.attributes[GEN_AI_SESSION_ID] == "sess-result"


@pytest.mark.asyncio
async def test_stream_event_session_propagates_before_first_assistant_message(
    tracer_provider, span_exporter
):
    messages = [
        StreamEvent("sess-stream"),
        AssistantMessage([TextBlock("streamed answer")]),
        ResultMessage("sess-stream"),
    ]

    await _run_stream(tracer_provider, messages)

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]

    assert agent_span.attributes[GEN_AI_SESSION_ID] == "sess-stream"
    assert llm_span.attributes[GEN_AI_SESSION_ID] == "sess-stream"


@pytest.mark.asyncio
async def test_stream_event_dict_session_fallback(
    tracer_provider, span_exporter
):
    messages = [
        StreamEvent(event_session_id="sess-event-dict"),
        AssistantMessage([TextBlock("streamed answer")]),
        ResultMessage("sess-event-dict"),
    ]

    await _run_stream(tracer_provider, messages)

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_by_operation(spans, "invoke_agent")[0]
    llm_span = _spans_by_operation(spans, "chat")[0]

    assert agent_span.attributes[GEN_AI_SESSION_ID] == "sess-event-dict"
    assert llm_span.attributes[GEN_AI_SESSION_ID] == "sess-event-dict"


def test_stream_event_without_session_skips_baggage_lookup(monkeypatch):
    def fail_baggage_lookup():
        raise AssertionError("unexpected per-event baggage lookup")

    monkeypatch.setattr(
        claude_patch,
        "_entry_baggage_identity_attributes",
        fail_baggage_lookup,
    )
    agent_invocation = SimpleNamespace(conversation_id=None, attributes={})

    claude_patch._process_stream_event_message(StreamEvent(), agent_invocation)

    assert agent_invocation.conversation_id is None
    assert GEN_AI_SESSION_ID not in agent_invocation.attributes


@pytest.mark.asyncio
async def test_client_query_session_id_is_used_before_result_message(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    client = SimpleNamespace(
        _otel_prompt=None,
        _otel_session_id=None,
        _otel_handler=handler,
        options=SimpleNamespace(model="claude-sonnet"),
    )

    async def wrapped_query(*args, **kwargs):
        return None

    async def wrapped_receive_response():
        yield AssistantMessage([TextBlock("answer")])
        yield ResultMessage(None)

    await wrap_claude_client_query(
        wrapped_query,
        client,
        ("hello",),
        {"session_id": "client-session"},
        handler=handler,
    )

    async for _ in wrap_claude_client_receive_response(
        wrapped_receive_response,
        client,
        (),
        {},
        handler=handler,
    ):
        pass

    agent_span = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )[0]
    assert agent_span.attributes[GEN_AI_SESSION_ID] == "client-session"


@pytest.mark.asyncio
async def test_client_query_without_session_does_not_write_default_session(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    client = SimpleNamespace(
        _otel_prompt=None,
        _otel_session_id=None,
        _otel_handler=handler,
        options=SimpleNamespace(model="claude-sonnet"),
    )

    async def wrapped_query(*args, **kwargs):
        return None

    async def wrapped_receive_response():
        yield AssistantMessage([TextBlock("answer")])
        yield ResultMessage(None)

    await wrap_claude_client_query(
        wrapped_query,
        client,
        ("hello",),
        {},
        handler=handler,
    )

    async for _ in wrap_claude_client_receive_response(
        wrapped_receive_response,
        client,
        (),
        {},
        handler=handler,
    ):
        pass

    agent_span = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )[0]
    assert GEN_AI_SESSION_ID not in agent_span.attributes
    assert GenAIAttributes.GEN_AI_CONVERSATION_ID not in agent_span.attributes


@pytest.mark.asyncio
async def test_standalone_query_resume_sets_initial_session(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    options = SimpleNamespace(model="claude-sonnet", resume="resume-session")

    async def wrapped_query(*args, **kwargs):
        yield AssistantMessage([TextBlock("answer")])
        yield ResultMessage(None)

    async for _ in wrap_query(
        wrapped_query,
        None,
        ("hello",),
        {"options": options},
        handler=handler,
    ):
        pass

    agent_span = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )[0]
    llm_span = _spans_by_operation(span_exporter.get_finished_spans(), "chat")[
        0
    ]

    assert agent_span.attributes[GEN_AI_SESSION_ID] == "resume-session"
    assert llm_span.attributes[GEN_AI_SESSION_ID] == "resume-session"


@pytest.mark.asyncio
async def test_wrap_query_sequential_calls_create_independent_root_traces(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    options = SimpleNamespace(model="claude-sonnet", resume="resume-session")

    async def wrapped_query(*args, **kwargs):
        yield AssistantMessage([TextBlock("answer")])
        yield ResultMessage(None)

    for prompt in ("first", "second"):
        async for _ in wrap_query(
            wrapped_query,
            None,
            (prompt,),
            {"options": options},
            handler=handler,
        ):
            pass

    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )

    assert len(agent_spans) == 2
    assert all(span.parent is None for span in agent_spans)
    assert len({span.context.trace_id for span in agent_spans}) == 2
    assert {span.attributes[GEN_AI_SESSION_ID] for span in agent_spans} == {
        "resume-session"
    }


@pytest.mark.asyncio
async def test_wrap_query_preserves_active_parent_context(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    options = SimpleNamespace(model="claude-sonnet", resume="parent-session")
    tracer = tracer_provider.get_tracer(__name__)

    async def wrapped_query(*args, **kwargs):
        yield AssistantMessage([TextBlock("answer")])
        yield ResultMessage(None)

    with tracer.start_as_current_span("caller-operation") as parent_span:
        async for _ in wrap_query(
            wrapped_query,
            None,
            ("inside parent",),
            {"options": options},
            handler=handler,
        ):
            pass

    agent_span = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )[0]

    assert agent_span.parent is not None
    assert agent_span.parent.span_id == parent_span.context.span_id
    assert agent_span.context.trace_id == parent_span.context.trace_id
    assert agent_span.attributes[GEN_AI_SESSION_ID] == "parent-session"


@pytest.mark.asyncio
async def test_result_message_closes_agent_before_iterator_is_exhausted(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    stream = await _read_until_result_without_exhausting(
        handler,
        [
            SystemMessage("sess-terminal"),
            AssistantMessage([TextBlock("terminal answer")]),
            ResultMessage("sess-terminal"),
        ],
    )
    try:
        agent_spans = _spans_by_operation(
            span_exporter.get_finished_spans(), "invoke_agent"
        )
        assert len(agent_spans) == 1
        assert agent_spans[0].attributes[GEN_AI_SESSION_ID] == (
            "sess-terminal"
        )
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_consecutive_result_breaks_keep_outer_parent_context(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)
    streams = []

    try:
        with tracer.start_as_current_span("caller-operation") as parent_span:
            streams.append(
                await _read_until_result_without_exhausting(
                    handler,
                    [
                        SystemMessage("sess-first"),
                        AssistantMessage([TextBlock("first answer")]),
                        ResultMessage("sess-first"),
                    ],
                )
            )
            streams.append(
                await _read_until_result_without_exhausting(
                    handler,
                    [
                        SystemMessage("sess-second"),
                        AssistantMessage([TextBlock("second answer")]),
                        ResultMessage("sess-second"),
                    ],
                )
            )
    finally:
        for stream in streams:
            await stream.aclose()

    agent_spans = {
        span.attributes[GEN_AI_SESSION_ID]: span
        for span in _spans_by_operation(
            span_exporter.get_finished_spans(), "invoke_agent"
        )
    }

    first_span = agent_spans["sess-first"]
    second_span = agent_spans["sess-second"]

    for span in (first_span, second_span):
        assert span.parent is not None
        assert span.parent.span_id == parent_span.context.span_id
        assert span.context.trace_id == parent_span.context.trace_id

    assert second_span.parent.span_id != first_span.context.span_id


@pytest.mark.asyncio
async def test_task_subagent_inherits_session_id(
    tracer_provider, span_exporter
):
    messages = [
        SystemMessage("sess-task"),
        AssistantMessage([TextBlock("I will delegate this.")]),
        AssistantMessage(
            [
                ToolUseBlock(
                    "toolu_task",
                    "Task",
                    {
                        "subagent_type": "code-reviewer",
                        "description": "review session handling",
                        "prompt": "check session propagation",
                    },
                )
            ]
        ),
        UserMessage(
            [ToolResultBlock("toolu_task", "task result")],
            tool_use_result={
                "agentId": "subagent-1",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        ),
        ResultMessage("sess-task"),
    ]

    await _run_stream(tracer_provider, messages)

    invoke_agent_sessions = [
        span.attributes[GEN_AI_SESSION_ID]
        for span in _spans_by_operation(
            span_exporter.get_finished_spans(), "invoke_agent"
        )
    ]
    tool_span = _spans_by_operation(
        span_exporter.get_finished_spans(), "execute_tool"
    )[0]

    assert invoke_agent_sessions.count("sess-task") == 2
    assert tool_span.attributes[GEN_AI_SESSION_ID] == "sess-task"


@pytest.mark.asyncio
async def test_sequential_streams_create_independent_root_traces(
    tracer_provider, span_exporter
):
    await _run_stream(
        tracer_provider,
        [
            SystemMessage("sess-first"),
            AssistantMessage([TextBlock("first answer")]),
            ResultMessage("sess-first"),
        ],
    )
    await _run_stream(
        tracer_provider,
        [
            SystemMessage("sess-second"),
            AssistantMessage([TextBlock("second answer")]),
            ResultMessage("sess-second"),
        ],
    )

    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    root_agent_spans = [span for span in agent_spans if span.parent is None]

    assert len(root_agent_spans) == 2
    assert {
        span.attributes[GEN_AI_SESSION_ID] for span in root_agent_spans
    } == {"sess-first", "sess-second"}
    assert len({span.context.trace_id for span in root_agent_spans}) == 2


@pytest.mark.asyncio
async def test_cancelled_stream_detaches_agent_context(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    async def _consume_cancelled_stream():
        async for _ in _process_agent_invocation_stream(
            wrapped_stream=_cancelled_stream("sess-cancelled"),
            handler=handler,
            model="claude-sonnet",
            prompt="this stream will be cancelled",
        ):
            pass

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(_consume_cancelled_stream(), timeout=0.01)

    await _run_stream(
        tracer_provider,
        [
            SystemMessage("sess-after-cancel"),
            AssistantMessage([TextBlock("answer after cancellation")]),
            ResultMessage("sess-after-cancel"),
        ],
    )

    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    cancelled_span = [
        span
        for span in agent_spans
        if span.attributes.get(GEN_AI_SESSION_ID) == "sess-cancelled"
    ][0]
    after_span = [
        span
        for span in agent_spans
        if span.attributes.get(GEN_AI_SESSION_ID) == "sess-after-cancel"
    ][0]

    assert cancelled_span.attributes["error.type"] == "CancelledError"
    assert after_span.parent is None
    assert after_span.context.trace_id != cancelled_span.context.trace_id


@pytest.mark.asyncio
async def test_stream_exception_after_llm_detaches_agent_context(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    with pytest.raises(RuntimeError, match="stream exploded"):
        async for _ in _process_agent_invocation_stream(
            wrapped_stream=_failing_stream(
                [
                    SystemMessage("sess-error"),
                    AssistantMessage([TextBlock("partial answer")]),
                ],
                RuntimeError("stream exploded"),
            ),
            handler=handler,
            model="claude-sonnet",
            prompt="this stream will fail after an LLM turn starts",
        ):
            pass

    await _run_stream(
        tracer_provider,
        [
            SystemMessage("sess-after-error"),
            AssistantMessage([TextBlock("answer after error")]),
            ResultMessage("sess-after-error"),
        ],
    )

    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    failed_span = [
        span
        for span in agent_spans
        if span.attributes.get(GEN_AI_SESSION_ID) == "sess-error"
    ][0]
    after_span = [
        span
        for span in agent_spans
        if span.attributes.get(GEN_AI_SESSION_ID) == "sess-after-error"
    ][0]

    assert failed_span.attributes["error.type"] == "RuntimeError"
    assert after_span.parent is None
    assert after_span.context.trace_id != failed_span.context.trace_id


@pytest.mark.asyncio
async def test_stream_exception_restores_active_parent_context(
    tracer_provider, span_exporter
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("caller-operation") as parent_span:
        with pytest.raises(RuntimeError, match="stream exploded"):
            async for _ in _process_agent_invocation_stream(
                wrapped_stream=_failing_stream(
                    [
                        SystemMessage("sess-parent-error"),
                        AssistantMessage([TextBlock("partial answer")]),
                    ],
                    RuntimeError("stream exploded"),
                ),
                handler=handler,
                model="claude-sonnet",
                prompt="this parented stream will fail",
            ):
                pass

        await _run_stream(
            tracer_provider,
            [
                SystemMessage("sess-parent-after-error"),
                AssistantMessage([TextBlock("answer after parented error")]),
                ResultMessage("sess-parent-after-error"),
            ],
        )

    agent_spans = {
        span.attributes[GEN_AI_SESSION_ID]: span
        for span in _spans_by_operation(
            span_exporter.get_finished_spans(), "invoke_agent"
        )
    }
    failed_span = agent_spans["sess-parent-error"]
    after_span = agent_spans["sess-parent-after-error"]

    for span in (failed_span, after_span):
        assert span.parent is not None
        assert span.parent.span_id == parent_span.context.span_id
        assert span.context.trace_id == parent_span.context.trace_id

    assert after_span.parent.span_id != failed_span.context.span_id


@pytest.mark.asyncio
async def test_parallel_streams_keep_session_ids_isolated(
    tracer_provider, span_exporter
):
    async def run(session_id):
        await _run_stream(
            tracer_provider,
            [
                SystemMessage(session_id),
                AssistantMessage([TextBlock(f"answer for {session_id}")]),
                AssistantMessage(
                    [
                        ToolUseBlock(
                            "toolu_shared",
                            "Read",
                            {"file_path": f"{session_id}.md"},
                        )
                    ]
                ),
                UserMessage(
                    [ToolResultBlock("toolu_shared", f"{session_id} content")]
                ),
                ResultMessage(session_id),
            ],
        )

    await asyncio.gather(run("sess-a"), run("sess-b"))

    agent_spans = _spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    sessions = {span.attributes[GEN_AI_SESSION_ID] for span in agent_spans}
    assert sessions == {"sess-a", "sess-b"}
    assert len({span.context.trace_id for span in agent_spans}) == 2

    tool_sessions = {
        span.attributes[GEN_AI_SESSION_ID]
        for span in _spans_by_operation(
            span_exporter.get_finished_spans(), "execute_tool"
        )
    }
    assert tool_sessions == {"sess-a", "sess-b"}
