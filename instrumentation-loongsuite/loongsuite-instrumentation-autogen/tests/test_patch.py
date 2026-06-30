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

import pytest

from opentelemetry.instrumentation.autogen import patch
from opentelemetry.instrumentation.autogen.semantic_conventions import (
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    GenAIOperation,
    GenAISpanKind,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class Handler:
    def __init__(self):
        self.calls = []
        self.invocations = []

    def start_invoke_agent(self, invocation):
        self.calls.append(("start_agent", invocation.agent_name))
        self.invocations.append(invocation)

    def stop_invoke_agent(self, invocation):
        self.calls.append(("stop_agent", invocation.agent_name))

    def fail_invoke_agent(self, invocation, error):
        self.calls.append(
            ("fail_agent", invocation.agent_name, error.type.__name__)
        )

    def start_llm(self, invocation):
        self.calls.append(("start_llm", invocation.request_model))
        self.invocations.append(invocation)

    def stop_llm(self, invocation):
        self.calls.append(("stop_llm", invocation.request_model))

    def fail_llm(self, invocation, error):
        self.calls.append(
            ("fail_llm", invocation.request_model, error.type.__name__)
        )

    def start_execute_tool(self, invocation):
        self.calls.append(("start_tool", invocation.tool_name))
        self.invocations.append(invocation)

    def stop_execute_tool(self, invocation):
        self.calls.append(("stop_tool", invocation.tool_name))

    def fail_execute_tool(self, invocation, error):
        self.calls.append(
            ("fail_tool", invocation.tool_name, error.type.__name__)
        )


class Agent:
    _name = "assistant"
    _description = "answers"
    _model_client = None
    _system_messages = []
    _tools = []


class ModelClient:
    _create_args = {"model": "qwen-plus"}


class SystemMessage:
    content = "system"


class UserMessage:
    content = "hello"


class Usage:
    prompt_tokens = 1
    completion_tokens = 2


class CreateResult:
    content = "done"
    finish_reason = "stop"
    usage = Usage()


class ChatMessage:
    content = "agent done"
    models_usage = Usage()


class Response:
    chat_message = ChatMessage()


class ToolCall:
    id = "call-1"
    name = "add_numbers"
    arguments = {"a": 2, "b": 3}


class FunctionExecutionResult:
    content = "5"
    call_id = "call-1"
    name = "add_numbers"


class MemoryQueryEvent:
    content = ["remembered preference"]
    source = "assistant"


class HandoffMessage:
    content = "handoff to worker"
    source = "planner"
    target = "worker"
    context = [UserMessage()]


class CodeResult:
    exit_code = 0
    output = "ok"


class CodeExecutionEvent:
    retry_attempt = 1
    result = CodeResult()
    source = "coder"


class TaskResult:
    messages = [ChatMessage()]
    stop_reason = "max turns reached"


class Team:
    _name = "research_team"
    _description = "coordinates agents"
    _participant_names = ["planner", "worker"]


class ModelContext:
    async def get_messages(self):
        return []


def _set_handler(handler: Handler):
    patch._get_handler.handler = handler  # type: ignore[attr-defined]


def test_direct_model_patch_list_excludes_wrappers_and_replay_clients():
    targets = {target for _, target in patch._DIRECT_MODEL_CLIENT_PATCHES}

    assert "ChatCompletionCache" not in targets
    assert "ReplayChatCompletionClient" not in targets


@pytest.mark.asyncio
async def test_agent_wrapper_starts_and_stops_invocation():
    handler = Handler()
    _set_handler(handler)

    async def wrapped():
        yield "item"

    items = [
        item
        async for item in patch._on_messages_stream_wrapper(
            wrapped, Agent(), (), {}
        )
    ]

    assert items == ["item"]
    assert handler.calls == [
        ("start_agent", "assistant"),
        ("stop_agent", "assistant"),
    ]


@pytest.mark.asyncio
async def test_agent_wrapper_enriches_native_autogen_agent_span(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    handler = Handler()
    _set_handler(handler)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    class NativeAgent(Agent):
        _model_client = ModelClient()
        _system_messages = [SystemMessage()]

    async def wrapped(*args, **kwargs):
        yield Response()

    with tracer.start_as_current_span(
        "invoke_agent assistant",
        attributes={
            GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
            GEN_AI_PROVIDER_NAME: AUTOGEN_PROVIDER_NAME,
            GEN_AI_SPAN_KIND: GenAISpanKind.AGENT,
        },
    ):
        items = [
            item
            async for item in patch._on_messages_stream_wrapper(
                wrapped, NativeAgent(), ([UserMessage()], None), {}
            )
        ]

    [span] = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})

    assert len(items) == 1
    assert handler.calls == []
    assert attributes["gen_ai.request.model"] == "qwen-plus"
    assert "gen_ai.input.messages" in attributes
    assert "system" in attributes["gen_ai.input.messages"]
    assert "hello" in attributes["gen_ai.input.messages"]
    assert "gen_ai.output.messages" in attributes
    assert "agent done" in attributes["gen_ai.output.messages"]


@pytest.mark.asyncio
async def test_agent_wrapper_fails_invocation_on_exception():
    handler = Handler()
    _set_handler(handler)

    async def wrapped():
        yield "item"
        raise ValueError("boom")

    with pytest.raises(ValueError):
        async for _ in patch._on_messages_stream_wrapper(
            wrapped, Agent(), (), {}
        ):
            pass

    assert handler.calls == [
        ("start_agent", "assistant"),
        ("fail_agent", "assistant", "ValueError"),
    ]


@pytest.mark.asyncio
async def test_agent_wrapper_resets_direct_model_context_when_closed_early():
    handler = Handler()
    _set_handler(handler)

    async def wrapped():
        assert patch._direct_model_agent_name.get() == "assistant"
        yield "item"

    generator = patch._on_messages_stream_wrapper(wrapped, Agent(), (), {})

    assert await generator.__anext__() == "item"
    assert patch._direct_model_agent_name.get() is None
    await generator.aclose()
    assert patch._direct_model_agent_name.get() is None

    assert handler.calls == [
        ("start_agent", "assistant"),
        ("fail_agent", "assistant", "GeneratorExit"),
    ]


@pytest.mark.asyncio
async def test_agent_wrapper_skips_when_native_autogen_agent_span_is_active():
    handler = Handler()
    _set_handler(handler)
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    async def wrapped():
        yield "native"

    with tracer.start_as_current_span(
        "invoke_agent assistant",
        attributes={
            GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
            GEN_AI_PROVIDER_NAME: AUTOGEN_PROVIDER_NAME,
            GEN_AI_SPAN_KIND: GenAISpanKind.AGENT,
        },
    ):
        items = [
            item
            async for item in patch._on_messages_stream_wrapper(
                wrapped, Agent(), (), {}
            )
        ]

    assert items == ["native"]
    assert handler.calls == []


@pytest.mark.asyncio
async def test_llm_wrapper_starts_and_stops_invocation():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        yield CreateResult()

    items = [
        item
        async for item in patch._call_llm_wrapper(
            wrapped,
            None,
            (),
            {
                "model_client": ModelClient(),
                "system_messages": [],
                "model_context": ModelContext(),
                "workbench": [],
                "handoff_tools": [],
                "agent_name": "assistant",
            },
        )
    ]

    assert len(items) == 1
    assert handler.calls == [
        ("start_llm", "qwen-plus"),
        ("stop_llm", "qwen-plus"),
    ]


@pytest.mark.asyncio
async def test_llm_wrapper_fails_invocation_when_closed_early():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        yield "chunk"
        yield CreateResult()

    generator = patch._call_llm_wrapper(
        wrapped,
        None,
        (),
        {
            "model_client": ModelClient(),
            "system_messages": [],
            "model_context": ModelContext(),
            "workbench": [],
            "handoff_tools": [],
            "agent_name": "assistant",
        },
    )

    assert await generator.__anext__() == "chunk"
    await generator.aclose()

    assert handler.calls == [
        ("start_llm", "qwen-plus"),
        ("fail_llm", "qwen-plus", "GeneratorExit"),
    ]


@pytest.mark.asyncio
async def test_direct_model_create_wrapper_records_fallback_llm_span():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        return CreateResult()

    token = patch._direct_model_agent_name.set("selector")
    try:
        result = await patch._direct_model_create_wrapper(
            wrapped,
            ModelClient(),
            ([UserMessage()],),
            {"tools": [], "json_output": True},
        )
    finally:
        patch._direct_model_agent_name.reset(token)

    assert isinstance(result, CreateResult)
    assert handler.calls == [
        ("start_llm", "qwen-plus"),
        ("stop_llm", "qwen-plus"),
    ]
    invocation = handler.invocations[0]
    assert invocation.attributes["gen_ai.agent.name"] == "selector"
    assert invocation.output_type == "json"
    assert invocation.input_tokens == 1
    assert invocation.output_tokens == 2
    assert invocation.finish_reasons == ["stop"]


@pytest.mark.asyncio
async def test_direct_model_stream_wrapper_records_first_token():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        yield "chunk"
        yield CreateResult()

    items = [
        item
        async for item in patch._direct_model_create_stream_wrapper(
            wrapped,
            ModelClient(),
            ([UserMessage()],),
            {"tools": []},
        )
    ]

    assert items[0] == "chunk"
    assert isinstance(items[1], CreateResult)
    assert handler.calls == [
        ("start_llm", "qwen-plus"),
        ("stop_llm", "qwen-plus"),
    ]
    assert handler.invocations[0].monotonic_first_token_s is not None


@pytest.mark.asyncio
async def test_direct_model_wrapper_skips_when_suppressed():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        return CreateResult()

    token = patch._suppress_direct_model_span.set(True)
    try:
        result = await patch._direct_model_create_wrapper(
            wrapped, ModelClient(), ([UserMessage()],), {}
        )
    finally:
        patch._suppress_direct_model_span.reset(token)

    assert isinstance(result, CreateResult)
    assert handler.calls == []


@pytest.mark.asyncio
async def test_selector_wrapper_sets_direct_model_agent_name():
    class Selector:
        _name = "selector_manager"

    async def wrapped(*args, **kwargs):
        assert patch._direct_model_agent_name.get() == "selector_manager"
        return "writer"

    result = await patch._selector_select_speaker_wrapper(
        wrapped, Selector(), (), {}
    )

    assert result == "writer"
    assert patch._direct_model_agent_name.get() is None


@pytest.mark.asyncio
async def test_team_wrapper_records_team_and_framework_events():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        yield MemoryQueryEvent()
        yield HandoffMessage()
        yield CodeExecutionEvent()
        yield TaskResult()

    items = [
        item
        async for item in patch._team_run_stream_wrapper(
            wrapped, Team(), (), {"task": "coordinate this task"}
        )
    ]

    assert len(items) == 4
    assert handler.calls == [
        ("start_agent", "research_team"),
        ("stop_agent", "research_team"),
    ]
    invocation = handler.invocations[0]
    assert invocation.attributes["autogen.team.type"] == "Team"
    assert invocation.attributes["autogen.team.participants"] == [
        "planner",
        "worker",
    ]
    assert invocation.attributes["autogen.memory.result_count"] == 1
    assert invocation.attributes["autogen.handoff.source"] == "planner"
    assert invocation.attributes["autogen.handoff.target"] == "worker"
    assert invocation.attributes["autogen.handoff.context_count"] == 1
    assert invocation.attributes["autogen.code.exit_code"] == 0
    assert invocation.attributes["autogen.code.retry_attempt"] == 1
    assert invocation.attributes["autogen.team.message_count"] == 1
    assert invocation.attributes["autogen.team.stop_reason"] == (
        "max turns reached"
    )
    assert invocation.input_messages[0].parts[0].content == (
        "coordinate this task"
    )


@pytest.mark.asyncio
async def test_execute_tool_wrapper_records_arguments_and_result():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        assert patch._suppress_native_tool_span.get()
        return ToolCall(), FunctionExecutionResult()

    result = await patch._execute_tool_call_wrapper(
        wrapped, None, (ToolCall(), [], [], "assistant", None, None), {}
    )

    assert isinstance(result, tuple)
    assert handler.calls == [
        ("start_tool", "add_numbers"),
        ("stop_tool", "add_numbers"),
    ]
    invocation = handler.invocations[0]
    assert invocation.tool_call_id == "call-1"
    assert invocation.tool_call_arguments == {"a": 2, "b": 3}
    assert invocation.tool_call_result == "5"
