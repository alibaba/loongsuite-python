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

"""Tests for A2AInstrumentor.

The span lifecycle is owned by the shared
``opentelemetry-util-genai`` ``ExtendedTelemetryHandler``: this
instrumentation only builds an ``InvokeAgentInvocation`` at the real
``a2a-sdk`` server-side executor boundary and drives it through
``start_invoke_agent`` / ``stop_invoke_agent`` /
``fail_invoke_agent``. Every test asserts on spans exported by the
*real* ``AgentExecutor`` ABC (never a stand-in for the framework).

Every GREEN assertion is paired with a RED baseline proving the same
executor call produces no telemetry span when the instrumentor is
inactive.
"""

from __future__ import annotations

import pytest
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import (
    ROLE_USER,
    Message,
    Part,
    SendMessageRequest,
    Task,
    TaskStatus,
)

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.a2a import (
    _A2A_TASK_ID,
    _A2A_TASK_STATE,
    _FRAMEWORK,
    A2AInstrumentor,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)

_AGENT = GenAiSpanKindValues.AGENT.value


# ---------------------------------------------------------------------------
# Real a2a-sdk primitives
# ---------------------------------------------------------------------------


def _real_request_context(
    *,
    user_input: str = "what is 2+2?",
    context_id: str = "ctx-1",
    task_id: str = "task-1",
    state: int = 1,
):
    """Build a *real* ``a2a-sdk`` ``RequestContext``."""
    message = Message(
        role=ROLE_USER,
        task_id=task_id,
        context_id=context_id,
        parts=[Part(text=user_input)],
    )
    task = Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=state),
    )
    return RequestContext(
        call_context=ServerCallContext(),
        request=SendMessageRequest(message=message),
        task_id=task_id,
        context_id=context_id,
        task=task,
    )


def _make_executor_cls(inner_tracer_provider=None, *, result="ok"):
    """Build a fresh real ``AgentExecutor`` subclass each call."""

    class _Exec(AgentExecutor):
        def __init__(self):
            self.inner_tracer = trace_api.get_tracer(
                "test.inner", tracer_provider=inner_tracer_provider
            )

        async def execute(self, context, event_queue):
            with self.inner_tracer.start_as_current_span("agent-inner-work"):
                pass
            return result

        async def cancel(self, context, event_queue):
            return None

    return _Exec


async def _run(executor, context=None):
    return await executor.execute(context or _real_request_context(), object())


def _agent_spans(spans):
    return [s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == _AGENT]


# ---------------------------------------------------------------------------
# RED: no telemetry without instrumentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_red_no_spans_without_instrumentation(span_exporter):
    ExecCls = _make_executor_cls()
    result = await _run(ExecCls())
    assert result == "ok"

    names = [s.name for s in span_exporter.get_finished_spans()]
    assert not any(n.startswith("invoke_agent") for n in names), names
    assert "a2a.execute" not in names, names


# ---------------------------------------------------------------------------
# GREEN: handler-owned invoke_agent AGENT span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_green_handler_owns_agent_span(instrument, span_exporter):
    ExecCls = _make_executor_cls()
    result = await _run(ExecCls())
    assert result == "ok"

    spans = span_exporter.get_finished_spans()
    agent_spans = _agent_spans(spans)
    assert len(agent_spans) == 1, [
        (s.name, dict(s.attributes or {})) for s in spans
    ]
    span = agent_spans[0]
    attrs = span.attributes

    assert span.name.startswith("invoke_agent ")
    assert span.name != "invoke_agent a2a"
    # The handler creates the AGENT execution span itself; it is INTERNAL
    # (server-side, in-process execution -- no client/protocol span).
    assert span.kind == SpanKind.INTERNAL
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    assert attrs.get(GenAI.GEN_AI_PROVIDER_NAME) == _FRAMEWORK
    assert attrs.get(GenAI.GEN_AI_AGENT_NAME) == "_Exec"
    assert span.status.status_code == StatusCode.UNSET


@pytest.mark.asyncio
async def test_green_context_id_maps_to_conversation_id(
    instrument, span_exporter
):
    """A2A ``contextId`` -> ``gen_ai.conversation.id`` (semconv #195)."""
    ExecCls = _make_executor_cls()
    context = _real_request_context(context_id="conversation-777")
    await _run(ExecCls(), context=context)

    agent = _agent_spans(span_exporter.get_finished_spans())[0]
    attrs = agent.attributes
    assert attrs.get(GenAI.GEN_AI_CONVERSATION_ID) == "conversation-777"
    # The legacy a2a.* context key must not be re-introduced.
    assert "a2a.context.id" not in attrs
    # Task context is still attached to the handler-owned span.
    assert attrs.get(_A2A_TASK_ID) == "task-1"
    state_name = attrs.get(_A2A_TASK_STATE)
    assert state_name in {
        "TASK_STATE_SUBMITTED",
        "SUBMITTED",
    }, state_name


@pytest.mark.asyncio
async def test_green_inner_work_nests_under_agent(
    instrument, tracer_provider, span_exporter
):
    # Route the executor's own child span to the same exporter so the
    # nesting relationship is observable.
    ExecCls = _make_executor_cls(inner_tracer_provider=tracer_provider)
    await _run(ExecCls())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2, [s.name for s in spans]

    agent = _agent_spans(spans)[0]
    inner = next(s for s in spans if s.name == "agent-inner-work")

    # The handler owns the whole boundary now, so inner executor work nests
    # directly under the single invoke_agent AGENT span (no structural span).
    assert inner.parent.span_id == agent.context.span_id
    assert inner.context.trace_id == agent.context.trace_id
    assert agent.kind == SpanKind.INTERNAL


# ---------------------------------------------------------------------------
# Content capture is entirely decided by the shared util
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_content_captured_by_default(
    instrument, span_exporter, monkeypatch
):
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        raising=False,
    )
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", raising=False)
    ExecCls = _make_executor_cls()
    await _run(ExecCls())

    agent = _agent_spans(span_exporter.get_finished_spans())[0]
    assert GenAI.GEN_AI_INPUT_MESSAGES not in agent.attributes


@pytest.mark.asyncio
async def test_span_only_captures_input_on_span(
    instrument, span_exporter, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    ExecCls = _make_executor_cls()
    await _run(
        ExecCls(),
        context=_real_request_context(user_input="secret-free prompt"),
    )

    agent = _agent_spans(span_exporter.get_finished_spans())[0]
    messages = agent.attributes.get(GenAI.GEN_AI_INPUT_MESSAGES)
    assert messages is not None
    assert "secret-free prompt" in messages


@pytest.mark.asyncio
async def test_event_only_emits_log_not_span_attribute(
    instrument, span_exporter, log_exporter, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "EVENT_ONLY"
    )
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", "true")
    ExecCls = _make_executor_cls()
    await _run(
        ExecCls(),
        context=_real_request_context(user_input="event-only prompt"),
    )

    agent = _agent_spans(span_exporter.get_finished_spans())[0]
    # Under EVENT_ONLY the span carries no message content ...
    assert GenAI.GEN_AI_INPUT_MESSAGES not in agent.attributes
    # ... but the shared util emits the details event with content.
    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    record = logs[0].log_record
    assert record.event_name == (
        "gen_ai.client.agent.invoke.operation.details"
    )
    assert "event-only prompt" in str(
        record.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    )


# ---------------------------------------------------------------------------
# Failure: handler records ERROR; caller sees the ORIGINAL exception
# ---------------------------------------------------------------------------


class _BoomError(RuntimeError):
    pass


def _boom_executor_cls():
    class _BoomExec(AgentExecutor):
        async def execute(self, context, event_queue):
            raise _BoomError("business boom")

        async def cancel(self, context, event_queue):
            return None

    return _BoomExec


@pytest.mark.asyncio
async def test_green_failure_records_error_and_reraises_original(
    instrument, span_exporter
):
    ExecCls = _boom_executor_cls()

    with pytest.raises(_BoomError) as excinfo:
        await ExecCls().execute(_real_request_context(), object())

    # The exact original exception type/message reaches the caller; the
    # telemetry layer never substitutes its own exception.
    assert type(excinfo.value) is _BoomError
    assert str(excinfo.value) == "business boom"

    agent = _agent_spans(span_exporter.get_finished_spans())[0]
    assert agent.status.status_code == StatusCode.ERROR
    assert agent.attributes.get("error.type") == "_BoomError"


@pytest.mark.asyncio
async def test_red_failure_reraises_without_instrumentation():
    ExecCls = _boom_executor_cls()
    with pytest.raises(_BoomError, match="business boom"):
        await ExecCls().execute(_real_request_context(), object())


# ---------------------------------------------------------------------------
# Fail-safe: telemetry faults never alter business behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fault_start_does_not_block_execution(
    instrument, handler, monkeypatch
):
    def _boom(*args, **kwargs):
        raise RuntimeError("telemetry start exploded")

    monkeypatch.setattr(handler, "start_invoke_agent", _boom)

    ExecCls = _make_executor_cls()
    assert await _run(ExecCls()) == "ok"


@pytest.mark.asyncio
async def test_fault_stop_does_not_block_execution(
    instrument, handler, monkeypatch
):
    def _boom(*args, **kwargs):
        raise RuntimeError("telemetry stop exploded")

    monkeypatch.setattr(handler, "stop_invoke_agent", _boom)

    ExecCls = _make_executor_cls()
    assert await _run(ExecCls()) == "ok"


@pytest.mark.asyncio
async def test_fault_set_attribute_does_not_block_execution(
    raising_tracer_provider, span_exporter
):
    """``span.set_attribute`` raising must not alter the business result."""
    # The raising provider serves both the boundary tracer and the shared
    # util handler, so handler-side attribute recording faults too.
    from opentelemetry.util.genai.extended_handler import (
        get_extended_telemetry_handler,
    )

    provider = raising_tracer_provider(setattr_raises=True)
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")
    get_extended_telemetry_handler(tracer_provider=provider)

    instrumentor = A2AInstrumentor()
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    try:
        ExecCls = _make_executor_cls()
        assert await _run(ExecCls()) == "ok"
        # Proof the fault really fired inside the handler: the AGENT
        # span never reached end() after set_attributes raised, so no
        # AGENT span was exported (only the business result matters).
        exported = span_exporter.get_finished_spans()
        assert not _agent_spans(exported), [s.name for s in exported]
    finally:
        instrumentor.uninstrument()
        if hasattr(get_extended_telemetry_handler, "_default_handler"):
            delattr(get_extended_telemetry_handler, "_default_handler")


@pytest.mark.asyncio
async def test_fault_error_recording_keeps_original_exception(
    raising_tracer_provider,
):
    """Error recording raising must not replace the business exception."""
    from opentelemetry.util.genai.extended_handler import (
        get_extended_telemetry_handler,
    )

    provider = raising_tracer_provider(error_raises=True)
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")
    get_extended_telemetry_handler(tracer_provider=provider)

    instrumentor = A2AInstrumentor()
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    try:
        ExecCls = _boom_executor_cls()
        with pytest.raises(_BoomError, match="business boom") as excinfo:
            await ExecCls().execute(_real_request_context(), object())
        assert type(excinfo.value) is _BoomError
    finally:
        instrumentor.uninstrument()
        if hasattr(get_extended_telemetry_handler, "_default_handler"):
            delattr(get_extended_telemetry_handler, "_default_handler")


@pytest.mark.asyncio
async def test_fault_fail_invoke_agent_keeps_original_exception(
    instrument, handler, monkeypatch
):
    """A raising ``fail_invoke_agent`` must not replace the error."""

    def _boom(invocation, error):
        raise RuntimeError("telemetry error recording exploded")

    monkeypatch.setattr(handler, "fail_invoke_agent", _boom)

    ExecCls = _boom_executor_cls()
    with pytest.raises(_BoomError, match="business boom") as excinfo:
        await ExecCls().execute(_real_request_context(), object())
    assert type(excinfo.value) is _BoomError


@pytest.mark.asyncio
async def test_fault_invocation_build_keeps_business_intact(instrument):
    """Context access exploding while building the invocation is fail-safe."""

    class _HostileContext:
        @property
        def context_id(self):
            raise RuntimeError("context exploded")

        @property
        def task_id(self):
            raise RuntimeError("context exploded")

        @property
        def current_task(self):
            raise RuntimeError("context exploded")

        def get_user_input(self):
            raise RuntimeError("context exploded")

    ExecCls = _make_executor_cls()
    assert await ExecCls().execute(_HostileContext(), object()) == "ok"


# ---------------------------------------------------------------------------
# No client/protocol span; late subclass hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_client_or_protocol_spans(instrument, span_exporter):
    ExecCls = _make_executor_cls()
    await _run(ExecCls())

    spans = span_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert len(names) == 1 and names[0].startswith("invoke_agent"), names
    for span in spans:
        assert span.kind != SpanKind.CLIENT
        assert span.kind != SpanKind.SERVER


@pytest.mark.asyncio
async def test_green_agent_span_late_subclass(instrument, span_exporter):
    ExecCls = _make_executor_cls()
    await _run(ExecCls())
    assert _agent_spans(span_exporter.get_finished_spans())


# ---------------------------------------------------------------------------
# uninstrument restores wrappers (drain-safe)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_red_uninstrument_restores_executor(
    tracer_provider, span_exporter
):
    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    ExecCls = _make_executor_cls()
    await _run(ExecCls())
    assert _agent_spans(span_exporter.get_finished_spans())

    instrumentor.uninstrument()
    span_exporter.clear()

    ExecCls2 = _make_executor_cls()
    await _run(ExecCls2())
    names = [s.name for s in span_exporter.get_finished_spans()]
    assert not any(n.startswith("invoke_agent") for n in names), names
    assert "a2a.execute" not in names


@pytest.mark.asyncio
async def test_double_uninstrument_safe(tracer_provider):
    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    instrumentor.uninstrument()
    instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_uninstrument_restores_init_subclass(tracer_provider):
    before = AgentExecutor.__dict__.get("__init_subclass__")

    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    assert AgentExecutor.__dict__.get("__init_subclass__") is not before

    instrumentor.uninstrument()
    after = AgentExecutor.__dict__.get("__init_subclass__")
    assert after is before

    class _Late(AgentExecutor):
        async def execute(self, context, event_queue):
            return None

        async def cancel(self, context, event_queue):
            return None

    assert _Late is not None
