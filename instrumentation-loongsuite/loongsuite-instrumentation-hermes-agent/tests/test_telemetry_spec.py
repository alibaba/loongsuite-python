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

import importlib
import json
import threading
from types import SimpleNamespace

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import LLMInvocation


def _spans_by_kind(span_exporter, span_kind: str):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.span.kind") == span_kind
    ]


def _assert_parent(child, parent):
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


def _messages(attr: str):
    return json.loads(attr)


def _assert_standard_entry_span(
    span,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    capture_content: bool = True,
    has_ttft: bool = False,
):
    attributes = span.attributes

    assert span.name == "enter_ai_application_system"
    assert attributes["gen_ai.span.kind"] == "ENTRY"
    assert attributes["gen_ai.operation.name"] == "enter"
    if session_id is not None:
        assert attributes["gen_ai.session.id"] == session_id
    if user_id is not None:
        assert attributes["gen_ai.user.id"] == user_id

    if capture_content:
        input_messages = _messages(attributes["gen_ai.input.messages"])
        output_messages = _messages(attributes["gen_ai.output.messages"])
        if input_text is not None:
            assert input_messages == [
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": input_text}],
                }
            ]
        if output_text is not None:
            assert output_messages == [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": output_text}],
                    "finish_reason": "stop",
                }
            ]
    else:
        assert "gen_ai.input.messages" not in attributes
        assert "gen_ai.output.messages" not in attributes

    if has_ttft:
        assert attributes["gen_ai.response.time_to_first_token"] > 0
    else:
        assert "gen_ai.response.time_to_first_token" not in attributes

    assert "gen_ai.provider.name" not in attributes
    assert "gen_ai.agent.system" not in attributes
    assert "gen_ai.request.model" not in attributes
    assert not any(key.startswith("hermes.") for key in attributes)


def _assert_no_captured_content(span):
    attributes = span.attributes
    for key in (
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    ):
        assert key not in attributes


def test_instrumentation_skips_missing_hermes_targets(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    monkeypatch,
):
    instrumentor_module = importlib.import_module(
        "opentelemetry.instrumentation.hermes_agent.instrumentor"
    )
    wrapped_targets = []

    def fake_wrap_function_wrapper(module_name, name, wrapper):
        if module_name == "tools.session_search_tool":
            raise ModuleNotFoundError(
                "No module named 'tools.session_search_tool'",
                name="tools.session_search_tool",
            )
        if module_name == "tools.delegate_tool":
            raise AttributeError("delegate_task")
        wrapped_targets.append((module_name, name))

    monkeypatch.setattr(
        instrumentor_module,
        "wrap_function_wrapper",
        fake_wrap_function_wrapper,
    )

    instrumentation_module.HermesAgentInstrumentor()._instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    assert ("run_agent", "AIAgent.run_conversation") in wrapped_targets
    assert ("tools.memory_tool", "memory_tool") in wrapped_targets
    assert not any(
        module_name == "tools.session_search_tool"
        for module_name, _ in wrapped_targets
    )


def test_uninstrumentation_skips_missing_hermes_targets(
    instrumentation_module,
    monkeypatch,
):
    instrumentor_module = importlib.import_module(
        "opentelemetry.instrumentation.hermes_agent.instrumentor"
    )
    ai_agent = type("AIAgent", (), {})
    modules = {
        "model_tools": SimpleNamespace(),
        "run_agent": SimpleNamespace(AIAgent=ai_agent),
        "tools.delegate_tool": SimpleNamespace(),
        "tools.memory_tool": SimpleNamespace(),
        "tools.todo_tool": SimpleNamespace(),
    }
    unwrapped_targets = []

    def fake_import_module(module_name):
        if module_name == "tools.session_search_tool":
            raise ModuleNotFoundError(
                "No module named 'tools.session_search_tool'",
                name="tools.session_search_tool",
            )
        return modules[module_name]

    def fake_unwrap(parent, attribute):
        if parent is modules["tools.delegate_tool"]:
            raise AttributeError(attribute)
        unwrapped_targets.append((parent, attribute))

    monkeypatch.setattr(
        instrumentor_module, "import_module", fake_import_module
    )
    monkeypatch.setattr(instrumentor_module, "unwrap", fake_unwrap)

    instrumentation_module.HermesAgentInstrumentor()._uninstrument()

    assert (ai_agent, "run_conversation") in unwrapped_targets
    assert (modules["tools.memory_tool"], "memory_tool") in unwrapped_targets
    assert not any(
        parent is modules["tools.delegate_tool"]
        for parent, _ in unwrapped_targets
    )


class _FakeAgent:
    def __init__(
        self,
        *,
        session_id: str,
        model: str = "qwen-turbo",
        provider: str = "dashscope",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        tools=None,
        platform: str | None = None,
        user_id: str | None = None,
    ):
        self.session_id = session_id
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.tools = tools or []
        self.platform = platform
        self._user_id = user_id


def _tool_call(
    *,
    call_id: str = "call-1",
    name: str = "read_file",
    arguments: str = '{"path": "/tmp/demo.txt"}',
):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    *,
    content=None,
    finish_reason: str = "stop",
    tool_calls=None,
    model: str = "qwen-turbo",
    response_id: str = "resp-1",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        model=model,
        id=response_id,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _response_with_cached_prompt_tokens(
    *,
    content: str = "最终答案",
    finish_reason: str = "stop",
    model: str = "qwen-turbo",
    response_id: str = "resp-cache-1",
    prompt_tokens: int = 14,
    completion_tokens: int = 5,
    cached_tokens: int = 4,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        model=model,
        id=response_id,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def _codex_response(
    *,
    content: str = "最终答案",
    model: str = "gpt-5.4",
    response_id: str = "resp-codex-1",
    status: str = "completed",
    incomplete_reason: str | None = None,
    input_tokens: int = 12,
    output_tokens: int = 7,
    cached_tokens: int = 3,
    tool_calls=None,
):
    output = []
    if tool_calls:
        for tool_call in tool_calls:
            output.append(
                SimpleNamespace(
                    type="function_call",
                    id=tool_call.id,
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            )
    elif content is not None:
        output.append(
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=content)],
            )
        )

    return SimpleNamespace(
        output=output,
        output_text=content or "",
        model=model,
        id=response_id,
        status=status,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason)
            if incomplete_reason is not None
            else None
        ),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def _codex_input_items(user_message: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": user_message,
        }
    ]


def _runtime(instrumentation_module, tracer_provider, meter_provider):
    tracer = trace_api.get_tracer(
        "hermes-agent-spec-tests",
        tracer_provider=tracer_provider,
    )
    handler = ExtendedTelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    metrics = instrumentation_module._HermesMetrics(
        meter_provider=meter_provider,
    )
    return SimpleNamespace(
        tracer=tracer,
        handler=handler,
        run_wrapper=instrumentation_module._RunConversationWrapper(handler),
        llm_wrapper=instrumentation_module._LLMCallWrapper(
            handler,
            metrics,
            streaming=False,
        ),
        streaming_llm_wrapper=instrumentation_module._LLMCallWrapper(
            handler,
            metrics,
            streaming=True,
        ),
        tool_wrapper=instrumentation_module._ToolCallWrapper(handler),
        tool_dispatch_wrapper=instrumentation_module._ToolDispatchWrapper(
            handler
        ),
        tool_batch_wrapper=instrumentation_module._ToolBatchWrapper(handler),
        delegate_tool_wrapper=instrumentation_module._ToolExecutionWrapper(
            handler,
            "delegate_task",
        ),
    )


def test_agent_layer_reuses_existing_entry_instead_of_creating_a_new_one(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-parent")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="完成", finish_reason="stop"),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "完成"}

    with runtime.tracer.start_as_current_span(
        "gateway request",
        attributes={
            "gen_ai.operation.name": "enter",
            "gen_ai.span.kind": "ENTRY",
        },
    ) as ingress_span:
        runtime.run_wrapper(wrapped_run, agent, ("请回复：完成",), {})

    entry_spans = _spans_by_kind(span_exporter, "ENTRY")
    agent_spans = _spans_by_kind(span_exporter, "AGENT")

    assert len(entry_spans) == 1
    assert len(agent_spans) == 1
    assert agent_spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"
    _assert_parent(agent_spans[0], ingress_span)


def test_agent_layer_reuses_active_llm_span_without_creating_entry(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="session-active-llm",
        platform="dingtalk",
        user_id="ding-user",
    )
    outer_llm = LLMInvocation(
        request_model="qwen-turbo",
        provider="dashscope",
    )

    runtime.handler.start_llm(outer_llm)
    try:
        runtime.run_wrapper(
            lambda user_message: {"final_response": "nested ok"},
            agent,
            ("nested call",),
            {},
        )
    finally:
        runtime.handler.stop_llm(outer_llm)

    entry_spans = _spans_by_kind(span_exporter, "ENTRY")
    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    llm_spans = _spans_by_kind(span_exporter, "LLM")

    assert entry_spans == []
    assert len(agent_spans) == 1
    assert len(llm_spans) == 1
    _assert_parent(agent_spans[0], llm_spans[0])


def test_api_server_agent_creates_entry_parent_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="api-session",
        platform="api_server",
        user_id="api-user",
    )

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="PONG", finish_reason="stop"),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "PONG"}

    runtime.run_wrapper(wrapped_run, agent, ("只回复 PONG",), {})

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]

    _assert_standard_entry_span(
        entry_span,
        session_id="api-session",
        user_id="api-user",
        input_text="只回复 PONG",
        output_text="PONG",
    )
    _assert_parent(agent_span, entry_span)


def test_gateway_platform_agent_creates_entry_parent_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="dingtalk-session",
        platform="dingtalk",
        user_id="ding-user",
    )

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="收到", finish_reason="stop"),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "收到"}

    runtime.run_wrapper(wrapped_run, agent, ("帮我看下",), {})

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]

    _assert_standard_entry_span(
        entry_span,
        session_id="dingtalk-session",
        user_id="ding-user",
        input_text="帮我看下",
        output_text="收到",
    )
    _assert_parent(agent_span, entry_span)


def test_entry_span_fails_when_agent_fail_cleanup_raises(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
    monkeypatch,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="entry-fail-session",
        platform="dingtalk",
        user_id="entry-fail-user",
    )

    class UserError(RuntimeError):
        pass

    original_fail_invoke_agent = runtime.handler.fail_invoke_agent

    def fail_invoke_agent_then_raise(invocation, error):
        original_fail_invoke_agent(invocation, error)
        raise RuntimeError("telemetry cleanup failed")

    monkeypatch.setattr(
        runtime.handler,
        "fail_invoke_agent",
        fail_invoke_agent_then_raise,
    )

    def wrapped_run(_user_message):
        raise UserError("boom")

    with pytest.raises(UserError):
        runtime.run_wrapper(
            wrapped_run,
            agent,
            ("please fail",),
            {},
        )

    entry_spans = _spans_by_kind(span_exporter, "ENTRY")
    agent_spans = _spans_by_kind(span_exporter, "AGENT")

    assert len(entry_spans) == 1
    assert len(agent_spans) == 1
    assert entry_spans[0].status.status_code == StatusCode.ERROR
    assert agent_spans[0].status.status_code == StatusCode.ERROR
    assert entry_spans[0].attributes["error.type"] == UserError.__qualname__
    assert agent_spans[0].attributes["error.type"] == UserError.__qualname__


def test_entry_span_omits_messages_when_content_capture_disabled(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
    monkeypatch,
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT"
    )
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="no-content-session",
        platform="dingtalk",
        user_id="no-content-user",
    )
    first_tool_call = _tool_call(call_id="call-no-content")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: "tool_ok",
                agent,
                (
                    "read_file",
                    {"path": "/tmp/demo.txt"},
                    "task-1",
                    first_tool_call.id,
                ),
                {},
            ),
            agent,
            (),
            {},
        )
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="完成",
                finish_reason="stop",
                response_id="resp-final-no-content",
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {
                            "role": "tool",
                            "tool_call_id": first_tool_call.id,
                            "content": "tool_ok",
                        },
                    ],
                },
            ),
            {},
        )
        return {"final_response": "完成"}

    runtime.run_wrapper(
        wrapped_run,
        agent,
        ("请回复：完成",),
        {},
    )

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    _assert_standard_entry_span(
        entry_span,
        session_id="no-content-session",
        user_id="no-content-user",
        capture_content=False,
    )
    for span in (
        _spans_by_kind(span_exporter, "AGENT")
        + _spans_by_kind(span_exporter, "LLM")
        + _spans_by_kind(span_exporter, "TOOL")
    ):
        _assert_no_captured_content(span)


def test_cli_platform_agent_creates_entry_parent_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="cli-session", platform="cli")

    runtime.run_wrapper(
        lambda user_message: {"final_response": "完成"},
        agent,
        ("请回复：完成",),
        {},
    )

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    _assert_standard_entry_span(
        entry_span,
        session_id="cli-session",
        input_text="请回复：完成",
        output_text="完成",
    )
    _assert_parent(agent_span, entry_span)


def test_tui_platform_agent_creates_entry_parent_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="tui-session", platform="tui")

    runtime.run_wrapper(
        lambda user_message: {"final_response": "完成"},
        agent,
        ("请回复：完成",),
        {},
    )

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    _assert_standard_entry_span(
        entry_span,
        session_id="tui-session",
        input_text="请回复：完成",
        output_text="完成",
    )
    _assert_parent(agent_span, entry_span)


def test_entry_platform_uses_env_session_source_when_agent_platform_missing(
    monkeypatch,
):
    helpers = importlib.import_module(
        "opentelemetry.instrumentation.hermes_agent.helpers"
    )
    monkeypatch.setenv("HERMES_SESSION_SOURCE", " Web ")
    agent = _FakeAgent(session_id="env-session")

    assert helpers.resolve_entry_platform(agent) == "web"


def test_entry_platform_prefers_agent_platform_over_env_session_source(
    monkeypatch,
):
    helpers = importlib.import_module(
        "opentelemetry.instrumentation.hermes_agent.helpers"
    )
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "web")
    agent = _FakeAgent(session_id="dingtalk-session", platform="dingtalk")

    assert helpers.resolve_entry_platform(agent) == "dingtalk"


def test_entry_platform_empty_env_uses_cli_default(monkeypatch):
    helpers = importlib.import_module(
        "opentelemetry.instrumentation.hermes_agent.helpers"
    )
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "   ")
    agent = _FakeAgent(session_id="default-session")

    assert helpers.resolve_entry_platform(agent) == "cli"


def test_extract_server_info_does_not_default_port_for_schemeless_url():
    helpers = importlib.import_module(
        "opentelemetry.instrumentation.hermes_agent.helpers"
    )
    agent = _FakeAgent(session_id="server-info", base_url="api.example.com")

    assert helpers._extract_server_info(agent) == ("api.example.com", None)


def test_extract_server_info_defaults_port_for_explicit_scheme():
    helpers = importlib.import_module(
        "opentelemetry.instrumentation.hermes_agent.helpers"
    )
    agent = _FakeAgent(
        session_id="server-info",
        base_url="https://api.example.com/compatible-mode/v1",
    )

    assert helpers._extract_server_info(agent) == ("api.example.com", 443)


def test_agent_without_platform_uses_env_session_source_for_entry(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "web")
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="env-entry-session")

    runtime.run_wrapper(
        lambda user_message: {"final_response": "完成"},
        agent,
        ("请回复：完成",),
        {},
    )

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    _assert_standard_entry_span(
        entry_span,
        session_id="env-entry-session",
        input_text="请回复：完成",
        output_text="完成",
    )
    _assert_parent(agent_span, entry_span)


def test_agent_without_platform_or_env_uses_cli_default_entry_source(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
    monkeypatch,
):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="library-session")

    runtime.run_wrapper(
        lambda user_message: {"final_response": "完成"},
        agent,
        ("请回复：完成",),
        {},
    )

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    _assert_standard_entry_span(
        entry_span,
        session_id="library-session",
        input_text="请回复：完成",
        output_text="完成",
    )
    _assert_parent(agent_span, entry_span)


def test_agent_span_does_not_backfill_agent_id_from_session_id(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-agent-id")

    runtime.run_wrapper(
        lambda user_message: {"final_response": "完成"},
        agent,
        ("请回复：完成",),
        {},
    )

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    assert (
        agent_span.attributes["gen_ai.conversation.id"] == "session-agent-id"
    )
    assert "gen_ai.agent.id" not in agent_span.attributes


def test_agent_provider_and_system_are_normalized_for_hermes(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="session-provider",
        provider="alibaba",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="完成", finish_reason="stop"),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "完成"}

    runtime.run_wrapper(wrapped_run, agent, ("请回复：完成",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]

    assert agent_span.attributes["gen_ai.provider.name"] == "hermes"
    assert agent_span.attributes["gen_ai.agent.system"] == "hermes"
    assert llm_span.attributes["gen_ai.provider.name"] == "dashscope"


def test_final_text_response_uses_stop_as_step_finish_reason(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-stop")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="最终答案", finish_reason="stop"
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请回复最终答案",), {})

    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    assert step_span.name == "react step"
    assert step_span.attributes["gen_ai.operation.name"] == "react"
    assert step_span.attributes["gen_ai.react.finish_reason"] == "stop"


def test_agent_span_uses_last_step_finish_reason(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-length")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="部分答案", finish_reason="length"
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "部分答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请回复一个长答案",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    step_span = _spans_by_kind(span_exporter, "STEP")[0]

    assert step_span.attributes["gen_ai.react.finish_reason"] == "length"
    assert list(agent_span.attributes["gen_ai.response.finish_reasons"]) == [
        "length"
    ]


def test_react_steps_restore_agent_context_between_rounds(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-react")
    first_tool_call = _tool_call()

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
                response_id="resp-tool",
                prompt_tokens=11,
                completion_tokens=3,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: "tool_ok",
                agent,
                (
                    "read_file",
                    {"path": "/tmp/demo.txt"},
                    "task-1",
                    first_tool_call.id,
                ),
                {},
            ),
            agent,
            (),
            {},
        )
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="tool_ok",
                finish_reason="stop",
                response_id="resp-final",
                prompt_tokens=13,
                completion_tokens=2,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "tool", "content": "tool_ok"},
                    ],
                },
            ),
            {},
        )
        return {"final_response": "tool_ok"}

    runtime.run_wrapper(wrapped_run, agent, ("读取文件后回复内容",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    step_spans = sorted(
        _spans_by_kind(span_exporter, "STEP"),
        key=lambda span: span.attributes["gen_ai.react.round"],
    )
    llm_spans = _spans_by_kind(span_exporter, "LLM")
    tool_spans = _spans_by_kind(span_exporter, "TOOL")

    assert len(step_spans) == 2
    _assert_parent(step_spans[0], agent_span)
    _assert_parent(step_spans[1], agent_span)
    _assert_parent(llm_spans[0], step_spans[0])
    _assert_parent(tool_spans[0], step_spans[0])
    _assert_parent(llm_spans[1], step_spans[1])
    assert step_spans[1].parent.span_id != step_spans[0].context.span_id


def test_tool_span_captures_call_id_arguments_and_result(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-tool")
    first_tool_call = _tool_call(call_id="call-read-file")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: "tool_ok",
                agent,
                (
                    "read_file",
                    {"path": "/tmp/demo.txt"},
                    "task-1",
                    first_tool_call.id,
                ),
                {},
            ),
            agent,
            (),
            {},
        )
        return {"final_response": "tool_ok"}

    runtime.run_wrapper(wrapped_run, agent, ("请调用 read_file",), {})

    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    tool_span = _spans_by_kind(span_exporter, "TOOL")[0]

    _assert_parent(tool_span, step_span)
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "read_file"
    assert tool_span.attributes["gen_ai.tool.call.id"] == "call-read-file"
    assert (
        "/tmp/demo.txt" in tool_span.attributes["gen_ai.tool.call.arguments"]
    )
    assert tool_span.attributes["gen_ai.tool.call.result"] == "tool_ok"


def test_tool_dispatch_span_captures_positional_call_id(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-dispatch")
    received = {}

    def wrapped_dispatch(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return "dispatch_ok"

    runtime.tool_dispatch_wrapper(
        wrapped_dispatch,
        agent,
        ("read_file", {"path": "/tmp/demo.txt"}, "task-1", "call-dispatch"),
        {},
    )

    tool_span = _spans_by_kind(span_exporter, "TOOL")[0]

    assert received["args"] == (
        "read_file",
        {"path": "/tmp/demo.txt"},
        "task-1",
        "call-dispatch",
    )
    assert received["kwargs"] == {}
    assert tool_span.attributes["gen_ai.tool.name"] == "read_file"
    assert tool_span.attributes["gen_ai.tool.call.id"] == "call-dispatch"


def test_delegate_task_tool_execution_captures_positional_arguments(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-delegate-positional")

    runtime.delegate_tool_wrapper(
        lambda *args, **kwargs: "delegate_ok",
        agent,
        (
            "修复测试",
            "失败日志在 pytest.log",
            ["file_tools"],
            [{"goal": "定位失败"}],
            3,
        ),
        {},
    )

    tool_span = _spans_by_kind(span_exporter, "TOOL")[0]
    arguments = json.loads(tool_span.attributes["gen_ai.tool.call.arguments"])

    assert tool_span.attributes["gen_ai.tool.name"] == "delegate_task"
    assert arguments == {
        "goal": "修复测试",
        "context": "失败日志在 pytest.log",
        "toolsets": ["file_tools"],
        "tasks": [{"goal": "定位失败"}],
        "max_iterations": 3,
    }


def test_skill_tool_span_captures_skill_semantic_attributes(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-skill")

    runtime.tool_wrapper(
        lambda *args, **kwargs: json.dumps(
            {
                "success": True,
                "name": "loongsuite-pr-review",
                "description": "Review LoongSuite PR readiness.",
                "metadata": {"version": "1.2.3"},
            }
        ),
        agent,
        (
            "skill_view",
            {"name": "loongsuite-pr-review"},
            "task-1",
            "call-skill-view",
        ),
        {},
    )

    tool_span = _spans_by_kind(span_exporter, "TOOL")[0]

    assert tool_span.attributes["gen_ai.tool.name"] == "skill_view"
    assert tool_span.attributes["gen_ai.skill.name"] == (
        "loongsuite-pr-review"
    )
    assert tool_span.attributes["gen_ai.skill.id"] == "loongsuite-pr-review"
    assert tool_span.attributes["gen_ai.skill.description"] == (
        "Review LoongSuite PR readiness."
    )
    assert tool_span.attributes["gen_ai.skill.version"] == "1.2.3"


def test_skill_manage_tool_span_captures_skill_semantic_attributes(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-skill-manage")

    runtime.tool_wrapper(
        lambda *args, **kwargs: json.dumps(
            {
                "success": True,
                "name": "loongsuite-ci-review",
                "description": "Review LoongSuite CI readiness.",
                "metadata": {"id": "skill-ci", "version": "2.0.0"},
            }
        ),
        agent,
        (
            "skill_manage",
            {"action": "create", "name": "loongsuite-ci-review"},
            "task-1",
            "call-skill-manage",
        ),
        {},
    )

    tool_span = _spans_by_kind(span_exporter, "TOOL")[0]

    assert tool_span.attributes["gen_ai.tool.name"] == "skill_manage"
    assert tool_span.attributes["gen_ai.tool.call.id"] == ("call-skill-manage")
    assert tool_span.attributes["gen_ai.skill.name"] == (
        "loongsuite-ci-review"
    )
    assert tool_span.attributes["gen_ai.skill.id"] == "skill-ci"
    assert tool_span.attributes["gen_ai.skill.description"] == (
        "Review LoongSuite CI readiness."
    )
    assert tool_span.attributes["gen_ai.skill.version"] == "2.0.0"


def test_nested_tool_dispatch_reuses_outer_tool_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    dispatch_wrapper = instrumentation_module._ToolDispatchWrapper(
        runtime.handler
    )
    agent = _FakeAgent(session_id="session-tool-nested")
    first_tool_call = _tool_call(call_id="call-read-file")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: dispatch_wrapper(
                    lambda *inner_args, **inner_kwargs: "tool_ok",
                    None,
                    ("read_file", {"path": "/tmp/demo.txt"}),
                    {},
                ),
                agent,
                (
                    "read_file",
                    {"path": "/tmp/demo.txt"},
                    "task-1",
                    first_tool_call.id,
                ),
                {},
            ),
            agent,
            (),
            {},
        )
        return {"final_response": "tool_ok"}

    runtime.run_wrapper(wrapped_run, agent, ("请调用 read_file",), {})

    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    tool_spans = _spans_by_kind(span_exporter, "TOOL")

    assert len(tool_spans) == 1
    _assert_parent(tool_spans[0], step_span)
    assert tool_spans[0].attributes["gen_ai.tool.call.id"] == "call-read-file"


def test_agent_rolls_up_last_response_metadata_and_usage(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-rollup")
    first_tool_call = _tool_call()

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
                response_id="resp-1",
                prompt_tokens=11,
                completion_tokens=3,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(lambda: "tool_ok", agent, (), {})
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="最终答案",
                finish_reason="stop",
                response_id="resp-2",
                prompt_tokens=13,
                completion_tokens=5,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "tool", "content": "tool_ok"},
                    ],
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(
        wrapped_run, agent, ("请在调用工具后给出最终答案",), {}
    )

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    assert agent_span.attributes["gen_ai.response.model"] == "qwen-turbo"
    assert agent_span.attributes["gen_ai.response.id"] == "resp-2"
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 24
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 8
    assert agent_span.attributes["gen_ai.usage.total_tokens"] == 32


def test_streaming_ttft_rolls_up_to_agent_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="session-stream",
        platform="api_server",
        user_id="stream-user",
    )

    def wrapped_run(user_message):
        runtime.streaming_llm_wrapper(
            _streaming_api_response,
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "1 2 3"}

    runtime.run_wrapper(wrapped_run, agent, ("请数到 3",), {})

    entry_span = _spans_by_kind(span_exporter, "ENTRY")[0]
    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]

    assert llm_span.attributes["gen_ai.operation.name"] == "chat"
    assert llm_span.attributes["gen_ai.response.time_to_first_token"] > 0
    assert agent_span.attributes["gen_ai.response.time_to_first_token"] > 0
    _assert_standard_entry_span(
        entry_span,
        session_id="session-stream",
        user_id="stream-user",
        input_text="请数到 3",
        output_text="1 2 3",
        has_ttft=True,
    )
    _assert_parent(agent_span, entry_span)


def test_codex_responses_usage_and_finish_reason_are_normalized(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="session-codex",
        model="gpt-5.4",
        provider="openai-codex",
    )
    agent.api_mode = "codex_responses"

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _codex_response(
                content="最终答案",
                input_tokens=12,
                output_tokens=7,
                cached_tokens=3,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "instructions": "你是一个严谨的助手。",
                    "input": _codex_input_items(user_message),
                    "max_output_tokens": 256,
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请直接回答",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]
    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    input_messages = _messages(llm_span.attributes["gen_ai.input.messages"])
    output_messages = _messages(llm_span.attributes["gen_ai.output.messages"])

    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 12
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 7
    assert llm_span.attributes["gen_ai.usage.total_tokens"] == 19
    assert llm_span.attributes["gen_ai.response.finish_reason"] == "stop"
    assert llm_span.attributes["gen_ai.response.finish_reasons"] == '["stop"]'
    assert input_messages == [
        {
            "role": "system",
            "parts": [{"type": "text", "content": "你是一个严谨的助手。"}],
        },
        {
            "role": "user",
            "parts": [{"type": "text", "content": "请直接回答"}],
        },
    ]
    assert output_messages == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "最终答案"}],
            "finish_reason": "stop",
        }
    ]
    assert step_span.attributes["gen_ai.react.finish_reason"] == "stop"
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 12
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 7
    assert agent_span.attributes["gen_ai.usage.total_tokens"] == 19


def test_chat_completions_total_tokens_match_hermes_prompt_total(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-cache-chat")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response_with_cached_prompt_tokens(
                prompt_tokens=14,
                completion_tokens=5,
                cached_tokens=4,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请直接回答",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]

    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 14
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 5
    assert llm_span.attributes["gen_ai.usage.total_tokens"] == 19
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 14
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 5
    assert agent_span.attributes["gen_ai.usage.total_tokens"] == 19


def test_streaming_wrapper_suppresses_nested_llm_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-nested-stream")

    runtime.streaming_llm_wrapper(
        lambda api_kwargs, **_: runtime.llm_wrapper(
            lambda inner_api_kwargs: _response(
                content="完成", finish_reason="stop"
            ),
            agent,
            (api_kwargs,),
            {},
        ),
        agent,
        (
            {
                "model": agent.model,
                "messages": [{"role": "user", "content": "请回复：完成"}],
            },
        ),
        {},
    )

    llm_spans = _spans_by_kind(span_exporter, "LLM")
    step_spans = _spans_by_kind(span_exporter, "STEP")

    assert len(llm_spans) == 1
    assert len(step_spans) == 1


def test_child_agent_invocation_reuses_ingress_without_creating_child_entry(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    parent_agent = _FakeAgent(session_id="session-parent")
    child_agent = _FakeAgent(session_id="session-child", platform="dingtalk")
    delegate_call = _tool_call(call_id="call-delegate", name="delegate_task")

    def child_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="child_result", finish_reason="stop"
            ),
            child_agent,
            (
                {
                    "model": child_agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "child_result"}

    def parent_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[delegate_call],
            ),
            parent_agent,
            (
                {
                    "model": parent_agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: runtime.run_wrapper(
                    child_run,
                    child_agent,
                    ("请读取文件并返回结果",),
                    {},
                ),
                parent_agent,
                (
                    "delegate_task",
                    {"goal": "请读取文件并返回结果"},
                    "task-1",
                    delegate_call.id,
                ),
                {},
            ),
            parent_agent,
            (),
            {},
        )
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="child_result", finish_reason="stop"
            ),
            parent_agent,
            (
                {
                    "model": parent_agent.model,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "tool", "content": "child_result"},
                    ],
                },
            ),
            {},
        )
        return {"final_response": "child_result"}

    with runtime.tracer.start_as_current_span(
        "gateway request",
        attributes={
            "gen_ai.operation.name": "enter",
            "gen_ai.span.kind": "ENTRY",
        },
    ) as ingress_span:
        runtime.run_wrapper(
            parent_run, parent_agent, ("委派子 agent 完成任务",), {}
        )

    entry_spans = _spans_by_kind(span_exporter, "ENTRY")
    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    tool_spans = [
        span
        for span in _spans_by_kind(span_exporter, "TOOL")
        if span.attributes["gen_ai.tool.name"] == "delegate_task"
    ]
    parent_agent_span = next(
        span
        for span in agent_spans
        if span.attributes["gen_ai.conversation.id"] == "session-parent"
    )
    child_agent_span = next(
        span
        for span in agent_spans
        if span.attributes["gen_ai.conversation.id"] == "session-child"
    )

    assert len(entry_spans) == 1
    assert len(agent_spans) == 2
    _assert_parent(parent_agent_span, ingress_span)
    _assert_parent(child_agent_span, tool_spans[0])


def test_nested_agent_response_id_does_not_leak_to_parent(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    """Parent agent span must record its own last response_id, not the child's.

    Regression test for the state corruption where child agent's LLM writes
    last_response_id into the shared ContextVar dict, overwriting the parent's
    value. Fixed by push_state/reset_state giving each invocation its own frame.
    """
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    parent_agent = _FakeAgent(session_id="session-parent-rid")
    child_agent = _FakeAgent(session_id="session-child-rid")
    delegate_call = _tool_call(
        call_id="call-delegate-rid", name="delegate_task"
    )

    def child_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="child answer",
                finish_reason="stop",
                response_id="child-resp-1",
                prompt_tokens=5,
                completion_tokens=2,
            ),
            child_agent,
            ({"model": child_agent.model, "messages": []},),
            {},
        )
        return {"final_response": "child answer"}

    def parent_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[delegate_call],
                response_id="parent-resp-1",
                prompt_tokens=10,
                completion_tokens=3,
            ),
            parent_agent,
            ({"model": parent_agent.model, "messages": []},),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: runtime.run_wrapper(
                    child_run,
                    child_agent,
                    ("子任务",),
                    {},
                ),
                parent_agent,
                (
                    "delegate_task",
                    {"goal": "子任务"},
                    "task-rid",
                    delegate_call.id,
                ),
                {},
            ),
            parent_agent,
            (),
            {},
        )
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="parent final",
                finish_reason="stop",
                response_id="parent-resp-2",
                prompt_tokens=12,
                completion_tokens=4,
            ),
            parent_agent,
            ({"model": parent_agent.model, "messages": []},),
            {},
        )
        return {"final_response": "parent final"}

    runtime.run_wrapper(
        parent_run, parent_agent, ("请委派子 agent 后给出最终答案",), {}
    )

    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    parent_span = next(
        s
        for s in agent_spans
        if s.attributes.get("gen_ai.conversation.id") == "session-parent-rid"
    )
    child_span = next(
        s
        for s in agent_spans
        if s.attributes.get("gen_ai.conversation.id") == "session-child-rid"
    )

    assert parent_span.attributes["gen_ai.response.id"] == "parent-resp-2", (
        "parent agent span recorded child's response_id — "
        "push_state/reset_state not isolating nested call state"
    )
    assert child_span.attributes["gen_ai.response.id"] == "child-resp-1"


def _streaming_api_response(api_kwargs, *, on_first_delta=None):
    if on_first_delta is not None:
        on_first_delta()
    return _response(content="1 2 3", finish_reason="stop")


def test_state_is_isolated_across_concurrent_invocations(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    """Concurrent calls on the same AIAgent must produce isolated react-step
    and token state.
    """
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    shared_agent = _FakeAgent(session_id="session-shared")
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run_one_conversation(thread_id: int) -> None:
        try:
            tool_call = _tool_call(call_id=f"call-t{thread_id}")

            def wrapped_run(user_message):
                # First LLM round: returns a tool_call -> opens a step span
                # that must be closed by the tool batch wrapper.
                runtime.llm_wrapper(
                    lambda api_kwargs: _response(
                        content=None,
                        finish_reason="tool_calls",
                        tool_calls=[tool_call],
                        response_id=f"resp-t{thread_id}-1",
                        prompt_tokens=11,
                        completion_tokens=3,
                    ),
                    shared_agent,
                    (
                        {
                            "model": shared_agent.model,
                            "messages": [
                                {"role": "user", "content": user_message}
                            ],
                        },
                    ),
                    {},
                )
                # Synchronise both threads after they have opened their
                # first step span. With the buggy implementation, the second
                # thread's ``state(instance)`` call would clobber the first
                # thread's ``current_step_invocation`` here, leaking a span.
                barrier.wait(timeout=5)
                runtime.tool_batch_wrapper(
                    lambda: runtime.tool_wrapper(
                        lambda *args, **kwargs: f"tool_ok_t{thread_id}",
                        shared_agent,
                        (
                            "read_file",
                            {"path": f"/tmp/t{thread_id}.txt"},
                            f"task-t{thread_id}",
                            tool_call.id,
                        ),
                        {},
                    ),
                    shared_agent,
                    (),
                    {},
                )
                # Second LLM round: closes the second step with finish=stop.
                runtime.llm_wrapper(
                    lambda api_kwargs: _response(
                        content=f"final-t{thread_id}",
                        finish_reason="stop",
                        response_id=f"resp-t{thread_id}-2",
                        prompt_tokens=13,
                        completion_tokens=2,
                    ),
                    shared_agent,
                    (
                        {
                            "model": shared_agent.model,
                            "messages": [
                                {"role": "user", "content": user_message},
                                {
                                    "role": "tool",
                                    "content": f"tool_ok_t{thread_id}",
                                },
                            ],
                        },
                    ),
                    {},
                )
                return {"final_response": f"final-t{thread_id}"}

            runtime.run_wrapper(
                wrapped_run, shared_agent, (f"hello-from-t{thread_id}",), {}
            )
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    threads = [
        threading.Thread(target=run_one_conversation, args=(i,))
        for i in (1, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, f"worker thread raised: {errors}"

    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    step_spans = _spans_by_kind(span_exporter, "STEP")
    llm_spans = _spans_by_kind(span_exporter, "LLM")
    tool_spans = _spans_by_kind(span_exporter, "TOOL")

    # Two concurrent agent invocations must produce two complete trees.
    assert len(agent_spans) == 2
    # Without the fix, one of the four step spans is leaked / never closed
    # because state["current_step_invocation"] is overwritten by the sibling
    # thread before ``finish_step`` gets to close it.
    assert len(step_spans) == 4, (
        "expected 4 react step spans (2 per concurrent call), got "
        f"{len(step_spans)} - state(instance) is not concurrency-safe"
    )
    assert len(llm_spans) == 4
    assert len(tool_spans) == 2

    # Each agent span must root its own complete sub-tree: 2 steps,
    # 2 LLM spans, 1 tool span. Without the fix, spans from sibling
    # invocations are mixed under the wrong agent root.
    for agent_span in agent_spans:
        descendant_steps = [
            span
            for span in step_spans
            if span.parent
            and span.parent.span_id == agent_span.context.span_id
        ]
        assert len(descendant_steps) == 2, (
            "each agent span should own exactly 2 react step spans, "
            f"got {len(descendant_steps)}"
        )
        descendant_step_ids = {
            span.context.span_id for span in descendant_steps
        }
        descendant_llms = [
            span
            for span in llm_spans
            if span.parent and span.parent.span_id in descendant_step_ids
        ]
        descendant_tools = [
            span
            for span in tool_spans
            if span.parent and span.parent.span_id in descendant_step_ids
        ]
        assert len(descendant_llms) == 2
        assert len(descendant_tools) == 1
