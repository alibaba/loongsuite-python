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

"""Tests for LlamaIndexInstrumentor.

These tests drive the *real* ``llama-index-core`` dispatcher with real
``MockLLM`` / ``MockEmbedding`` stubs and a real workflow ``ReActAgent``
(scripted streaming model + a real ``FunctionTool``); only the model output
is scripted. They assert that:

  * the shared ``ExtendedTelemetryHandler`` owns one ``invoke_agent`` AGENT
    span per real agent run and one ``execute_tool`` TOOL span per tool
    execution, with correct nesting;
  * agent-internal setup/parse/step/call_tool orchestration produces no
    AGENT (and no extra) spans; a non-agent LLM call is an LLM span;
  * all four content-capture modes behave through the shared handler,
    including EVENT_ONLY (log event, no span content);
  * telemetry faults (start/stop/set_attribute raising) never block the
    business run, change its result, or replace a business exception;
  * uninstrument drains open spans and stops span production.
"""

from __future__ import annotations

from importlib.metadata import requires

import pytest

from opentelemetry.instrumentation.llama_index import (
    _GEN_AI_FRAMEWORK,
    _GEN_AI_OPERATION_NAME,
    _GEN_AI_SPAN_KIND,
    _SPAN_KIND_AGENT,
    _SPAN_KIND_EMBEDDING,
    _SPAN_KIND_LLM,
    _SPAN_KIND_TOOL,
    LlamaIndexInstrumentor,
    _classify,
    _span_id_prefix,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

CAPTURE_ENVVAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
INPUT_MESSAGES_KEY = GenAI.GEN_AI_INPUT_MESSAGES
OUTPUT_MESSAGES_KEY = GenAI.GEN_AI_OUTPUT_MESSAGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm():
    from llama_index.core.llms import MockLLM

    return MockLLM(max_tokens=8)


def _chat_once(llm):
    from llama_index.core.llms import ChatMessage

    return llm.chat([ChatMessage(role="user", content="hi there")])


def _spans_by_kind(span_exporter):
    spans = span_exporter.get_finished_spans()
    grouped: dict[str, list] = {}
    for span in spans:
        kind = span.attributes.get(_GEN_AI_SPAN_KIND)
        grouped.setdefault(kind, []).append(span)
    return grouped


def _make_react_llm(boom: bool = False):
    """A streaming MockLLM that drives one tool call then a final answer.

    Workflow agents consume ``astream_chat`` streams; only the model output
    is scripted -- every framework event/span is emitted for real.
    """
    from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM

    class ScriptedReAct(MockLLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            object.__setattr__(self, "_calls", 0)

        def _next(self):
            calls = self._calls + 1
            object.__setattr__(self, "_calls", calls)
            if boom:
                raise ValueError("business boom")
            if calls == 1:
                text = (
                    "Thought: I need the weather\n"
                    "Action: get_weather\n"
                    'Action Input: {"city": "SF"}'
                )
            else:
                text = "Thought: done\nAnswer: sunny in SF"
            return ChatResponse(
                message=ChatMessage(role="assistant", content=text)
            )

        async def astream_chat(self, messages, **kwargs):
            response = self._next()

            async def gen():
                yield response

            return gen()

    return ScriptedReAct()


async def _run_react_agent():
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool

    def get_weather(city: str) -> str:
        """Useful for getting the weather for a city."""
        return f"sunny in {city}"

    agent = ReActAgent(
        tools=[FunctionTool.from_defaults(fn=get_weather)],
        llm=_make_react_llm(),
    )
    return await agent.run("what is the weather in SF?")


# ---------------------------------------------------------------------------
# Pure classification unit tests
# ---------------------------------------------------------------------------


def test_span_id_prefix_strips_uuid():
    assert (
        _span_id_prefix("MockLLM.chat-8c7b6315-7a0e-401b-9185-59474d2632c0")
        == "MockLLM.chat"
    )
    assert _span_id_prefix("") == ""


@pytest.mark.parametrize(
    "prefix,expected",
    [
        # The real agent boundary.
        ("ReActAgent.run", "agent"),
        ("FunctionAgent.arun", "agent"),
        ("AgentWorkflow.run", "agent"),
        # The actual tool/function execution.
        ("FunctionTool.call", "tool"),
        ("FunctionTool.acall", "tool"),
        # Model calls -- LLM, including inside agent loops / streaming.
        ("MockLLM.chat", "llm"),
        ("OpenAI.complete", "llm"),
        ("MockLLM.astream_chat", "llm"),
        ("OpenAI.astructured_predict", "llm"),
        # Other GenAI operations.
        ("MockEmbedding.get_text_embedding", "embedding"),
        ("VectorIndexRetriever.retrieve", "retrieval"),
        ("LLMRerank.postprocess_nodes", "rerank"),
    ],
)
def test_classify_maps_real_operations(prefix, expected):
    assert _classify(prefix) == expected


@pytest.mark.parametrize(
    "prefix",
    [
        # Agent-internal machinery must not become AGENT spans.
        ("FunctionAgent.setup_agent"),
        ("BaseWorkflowAgent.init_run"),
        ("BaseWorkflowAgent.run_agent_step"),
        ("BaseWorkflowAgent.take_step"),
        ("ReActOutputParser.parse"),
        ("BaseWorkflowAgent.parse_agent_output"),
        ("BaseWorkflowAgent.aggregate_tool_results"),
        # The agent's own tool orchestration is not the execution.
        ("BaseWorkflowAgent.call_tool"),
        # The Tool.__call__ trampoline only delegates to Tool.call.
        ("FunctionTool.__call__"),
        # Generic query/chain steps are not standalone GenAI operations here.
        ("RetrieverQueryEngine.query"),
        ("CompactAndRefine.synthesize"),
        ("RetrieverQueryEngine.synthesize"),
    ],
)
def test_classify_skips_internal_and_chain_steps(prefix):
    assert _classify(prefix) is None, prefix


def test_internal_llm_call_is_llm_not_agent():
    # The model turn inside an agent loop is an LLM, not another AGENT span.
    assert _classify("MockLLM.astream_chat") == "llm"
    assert _classify("MockLLM.stream_complete") == "llm"


# ---------------------------------------------------------------------------
# RED: no spans when not instrumented
# ---------------------------------------------------------------------------


def test_red_no_spans_without_instrumentation(span_exporter):
    _chat_once(_mock_llm())
    assert span_exporter.get_finished_spans() == ()


# ---------------------------------------------------------------------------
# GREEN: LLM chat spans are handler-owned LLM spans
# ---------------------------------------------------------------------------


def test_green_chat_produces_llm_span(instrument, span_exporter):
    _chat_once(_mock_llm())

    grouped = _spans_by_kind(span_exporter)
    chat_spans = grouped.get(_SPAN_KIND_LLM, [])
    assert chat_spans, grouped.keys()
    # The outer call is chat; the inner prompt completion is text_completion.
    operations = {s.attributes.get(_GEN_AI_OPERATION_NAME) for s in chat_spans}
    assert operations == {"chat", "text_completion"}, operations
    for span in chat_spans:
        assert span.attributes.get(_GEN_AI_FRAMEWORK) == "llama_index"


def test_green_chat_complete_share_trace_and_exact_parent(
    instrument, span_exporter
):
    """MockLLM.chat internally calls MockLLM.complete (real framework flow)."""
    _chat_once(_mock_llm())

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 2

    assert {s.context.trace_id for s in spans} and len(
        {s.context.trace_id for s in spans}
    ) == 1

    def _find(operation: str):
        return next(
            s
            for s in spans
            if s.attributes.get(_GEN_AI_OPERATION_NAME) == operation
        )

    chat = _find("chat")
    complete = _find("text_completion")
    # Handler-owned spans carry the handler's naming/kind.
    assert chat.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_LLM
    assert complete.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_LLM
    # Exact parent id, not just "some exported span".
    assert complete.parent is not None
    assert complete.parent.span_id == chat.context.span_id


def test_green_embedding_span(instrument, span_exporter):
    from llama_index.core.embeddings import MockEmbedding

    MockEmbedding(embed_dim=4).get_text_embedding("hello world")

    grouped = _spans_by_kind(span_exporter)
    assert grouped.get(_SPAN_KIND_EMBEDDING), grouped.keys()


# ---------------------------------------------------------------------------
# GREEN: one AGENT + one TOOL span for a real workflow agent run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_agent_run_one_agent_one_tool_nested(
    instrument, span_exporter
):
    response = await _run_react_agent()
    assert "sunny in SF" in str(response)

    grouped = _spans_by_kind(span_exporter)
    agent_spans = grouped.get(_SPAN_KIND_AGENT, [])
    tool_spans = grouped.get(_SPAN_KIND_TOOL, [])
    llm_spans = grouped.get(_SPAN_KIND_LLM, [])

    # Exactly one agent invocation for the whole run...
    assert len(agent_spans) == 1, [s.name for s in agent_spans]
    agent_span = agent_spans[0]
    assert agent_span.name.startswith("invoke_agent"), agent_span.name
    assert agent_span.attributes.get(_GEN_AI_OPERATION_NAME) == "invoke_agent"
    assert agent_span.parent is None

    # ...one tool execution, handler-named and nested directly under the
    # agent span even though skipped orchestration spans sit between them...
    assert len(tool_spans) == 1, [s.name for s in tool_spans]
    tool_span = tool_spans[0]
    assert tool_span.name.startswith("execute_tool"), tool_span.name
    assert "get_weather" in tool_span.name
    assert tool_span.attributes.get(_GEN_AI_OPERATION_NAME) == "execute_tool"
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id

    # ...and the in-loop model turns are LLM spans, also under the agent.
    assert len(llm_spans) >= 1, grouped.keys()
    for llm_span in llm_spans:
        assert llm_span.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_LLM
        assert llm_span.parent is not None
        assert llm_span.parent.span_id == agent_span.context.span_id

    # All spans share one trace.
    all_spans = span_exporter.get_finished_spans()
    assert len({s.context.trace_id for s in all_spans}) == 1


@pytest.mark.asyncio
async def test_standalone_llm_call_emits_no_agent_span(
    instrument, span_exporter
):
    # A plain model call (indexing/retrieval style) must not be an agent.
    _chat_once(_mock_llm())
    grouped = _spans_by_kind(span_exporter)
    assert grouped.get(_SPAN_KIND_AGENT, []) == []
    assert grouped.get(_SPAN_KIND_LLM)


# ---------------------------------------------------------------------------
# Uninstrument: stops spans and drains stranded ones
# ---------------------------------------------------------------------------


def test_red_uninstrument_stops_spans(span_exporter, tracer_provider):
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    _chat_once(_mock_llm())
    assert span_exporter.get_finished_spans()

    instrumentor.uninstrument()
    span_exporter.clear()

    _chat_once(_mock_llm())
    assert span_exporter.get_finished_spans() == ()


def test_instrument_is_idempotent_on_uninstrument():
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(skip_dep_check=True)
    instrumentor.uninstrument()
    instrumentor.uninstrument()  # second uninstrument must not raise


def test_uninstrument_drains_open_spans(span_exporter, tracer_provider):
    from llama_index.core.instrumentation import get_dispatcher

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    dispatcher = get_dispatcher()
    # Open a real agent span through the dispatcher and never close it.
    dispatcher.span_enter(
        id_="ReActAgent.run-drain-probe", bound_args=None, instance=None
    )
    assert span_exporter.get_finished_spans() == ()

    instrumentor.uninstrument()
    ended = span_exporter.get_finished_spans()
    agent_spans = [
        s
        for s in ended
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    ]
    assert len(agent_spans) == 1, [s.name for s in ended]
    assert agent_spans[0].name.startswith("invoke_agent")


# ---------------------------------------------------------------------------
# Content capture modes (all owned by the shared handler)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    ["NO_CONTENT", "SPAN_ONLY", "EVENT_ONLY", "SPAN_AND_EVENT"],
)
def test_capture_modes_for_llm_chat(
    instrument, span_exporter, log_exporter, monkeypatch, mode
):
    monkeypatch.setenv(CAPTURE_ENVVAR, mode)
    _chat_once(_mock_llm())

    llm_span = next(
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_LLM
        and s.attributes.get(_GEN_AI_OPERATION_NAME) == "chat"
        and s.parent is None
    )
    events = [
        record
        for record in log_exporter.get_finished_logs()
        if record.log_record.event_name
        == "gen_ai.client.inference.operation.details"
    ]

    content_on_span = INPUT_MESSAGES_KEY in llm_span.attributes
    content_on_event = any(
        INPUT_MESSAGES_KEY in (record.log_record.attributes or {})
        for record in events
    )

    if mode == "NO_CONTENT":
        assert not content_on_span
        assert not events
    elif mode == "SPAN_ONLY":
        assert content_on_span
        assert not content_on_event
    elif mode == "EVENT_ONLY":
        assert not content_on_span
        assert content_on_event
    else:  # SPAN_AND_EVENT
        assert content_on_span
        assert content_on_event

    # Structural span exists under every mode.
    assert llm_span.attributes.get(_GEN_AI_FRAMEWORK) == "llama_index"


@pytest.mark.asyncio
async def test_event_only_agent_run_emits_agent_event_without_span_content(
    instrument, span_exporter, log_exporter, monkeypatch
):
    monkeypatch.setenv(CAPTURE_ENVVAR, "EVENT_ONLY")
    await _run_react_agent()

    agent_span = next(
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    )
    assert INPUT_MESSAGES_KEY not in agent_span.attributes
    assert OUTPUT_MESSAGES_KEY not in agent_span.attributes

    agent_events = [
        record
        for record in log_exporter.get_finished_logs()
        if record.log_record.event_name
        == "gen_ai.client.agent.invoke.operation.details"
    ]
    assert agent_events, [
        r.log_record.event_name for r in log_exporter.get_finished_logs()
    ]


# ---------------------------------------------------------------------------
# Fail-safe: telemetry faults never break the business flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_failure_does_not_block_agent_run(
    instrument, span_exporter, monkeypatch
):
    handler = instrument._genai_handler

    def boom_start(invocation, context=None):
        raise RuntimeError("telemetry start exploded")

    monkeypatch.setattr(handler, "start_invoke_agent", boom_start)

    response = await _run_react_agent()
    # Business result unchanged.
    assert "sunny in SF" in str(response)
    # No agent span could start; the run still completed.
    grouped = _spans_by_kind(span_exporter)
    assert grouped.get(_SPAN_KIND_AGENT, []) == []


@pytest.mark.asyncio
async def test_start_failure_after_span_created_closes_span_and_restores_context(
    instrument, span_exporter, monkeypatch
):
    """A ``start_*`` that raises *after* creating the span and attaching its
    context must not strand either.

    ``test_start_failure_does_not_block_agent_run`` covers a ``start_*`` that
    raises before it does anything. Here the real ``start_llm`` runs -- it
    creates the span and attaches the invocation's context -- and only then
    raises, so the handler is left holding a partially initialized invocation.
    The ambient context must be restored and no span may stay recording.
    """
    from opentelemetry import context as context_api
    from opentelemetry import trace as trace_api

    handler = instrument._genai_handler
    real_start = handler.start_llm
    started = []

    def start_then_boom(invocation, context=None):
        real_start(invocation, context=context)
        started.append(invocation)
        raise RuntimeError("telemetry start exploded after span creation")

    monkeypatch.setattr(handler, "start_llm", start_then_boom)

    outer_context = context_api.get_current()
    outer_span = trace_api.get_current_span()

    _chat_once(_mock_llm())

    # The instrumented chat call must have reached the failing start_llm.
    assert started, "start_llm was never reached"
    for invocation in started:
        # start_llm created the span and attached its context; neither may
        # outlive a start that raised before the record was published.
        assert invocation.context_token is None
        assert invocation.span is None or not invocation.span.is_recording()
    # The ambient context is restored to exactly what it was before the call.
    assert context_api.get_current() is outer_context
    assert trace_api.get_current_span() is outer_span


@pytest.mark.asyncio
async def test_stop_failure_does_not_block_agent_run_or_strand_span(
    instrument, span_exporter, monkeypatch
):
    handler = instrument._genai_handler

    def boom_stop(invocation):
        # Simulate a failure deep in attribute serialization.
        invocation.span.set_attribute("gen_ai.boom", "x")
        raise RuntimeError("telemetry stop exploded")

    monkeypatch.setattr(handler, "stop_invoke_agent", boom_stop)

    response = await _run_react_agent()
    assert "sunny in SF" in str(response)

    # The fail-safe fallback still detaches and ends the agent span even
    # though stop_invoke_agent raised. Under a set_attribute fault the handler
    # could not write gen_ai.span.kind, so identify the span by its name.
    ended = span_exporter.get_finished_spans()
    agent_spans = [s for s in ended if s.name.startswith("invoke_agent")]
    assert len(agent_spans) == 1, [s.name for s in ended]


@pytest.mark.asyncio
async def test_set_attribute_failure_never_blocks_agent_run(
    instrument, span_exporter, monkeypatch
):
    from opentelemetry.sdk.trace import _Span as SdkSpan

    real_set_attribute = SdkSpan.set_attribute

    def raising_set_attribute(self, key, value):  # noqa: ANN001
        if str(key).startswith("gen_ai."):
            raise RuntimeError("set_attribute exploded")
        return real_set_attribute(self, key, value)

    monkeypatch.setattr(SdkSpan, "set_attribute", raising_set_attribute)

    response = await _run_react_agent()
    assert "sunny in SF" in str(response)
    # The run produced its full span tree despite every gen_ai setattr
    # failing (spans are identified by name because the fault prevents the
    # handler from writing gen_ai.span.kind).
    ended = span_exporter.get_finished_spans()
    assert len([s for s in ended if s.name.startswith("invoke_agent")]) == 1
    assert len([s for s in ended if s.name.startswith("execute_tool")]) == 1


@pytest.mark.asyncio
async def test_business_exception_propagates_unchanged_through_failing_telemetry(
    instrument, span_exporter, monkeypatch
):
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool

    handler = instrument._genai_handler

    def boom_fail(invocation, error):  # noqa: ANN001
        raise RuntimeError("telemetry fail exploded")

    monkeypatch.setattr(handler, "fail_invoke_agent", boom_fail)

    def get_weather(city: str) -> str:
        """Useful for getting the weather for a city."""
        return f"sunny in {city}"

    agent = ReActAgent(
        tools=[FunctionTool.from_defaults(fn=get_weather)],
        llm=_make_react_llm(boom=True),
    )

    # The *business* ValueError/message must be the one raised, never replaced
    # by the telemetry RuntimeError.
    with pytest.raises(ValueError, match="business boom"):
        await agent.run("weather?")


# ---------------------------------------------------------------------------
# Runtime dependency declaration
# ---------------------------------------------------------------------------


def test_chat_end_without_response_does_not_report_request_as_output():
    """A LLMChatEndEvent whose optional response is absent must leave output
    unset; the event's ``messages`` are request messages, not generated output.
    Predict/structured-predict end events instead carry the result in ``output``.
    """
    from llama_index.core.instrumentation.events.llm import (
        LLMChatEndEvent,
        LLMPredictEndEvent,
    )
    from llama_index.core.llms import ChatMessage

    from opentelemetry.instrumentation.llama_index import (
        _build_event_handler,
        _build_span_handler,
    )
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
    )
    from opentelemetry.util.genai.types import LLMInvocation

    span_handler = _build_span_handler(ExtendedTelemetryHandler())
    handler = _build_event_handler(span_handler)

    class _Rec:
        def __init__(self, invocation):
            self.kind = "llm"
            self.invocation = invocation

    # 1) chat end, no response: request messages must not become output.
    chat_inv = LLMInvocation(provider="llama_index")
    span_handler._ls_records["chat"] = _Rec(chat_inv)
    # LLMChatEndEvent requires the response field; None means the optional
    # generated response is absent (messages here are still request messages).
    handler._handle_llm(
        chat_inv,
        "LLMChatEndEvent",
        LLMChatEndEvent(
            messages=[ChatMessage(role="user", content="the prompt")],
            response=None,
        ),
    )
    assert chat_inv.output_messages == []

    # 2) predict end: the event carries the generated string in ``output``;
    # the handler routes it through _enrich_llm_response which turns a bare
    # string response into an output message.
    pred_inv = LLMInvocation(provider="llama_index")
    span_handler._ls_records["pred"] = _Rec(pred_inv)
    handler._handle_llm(
        pred_inv,
        "LLMPredictEndEvent",
        LLMPredictEndEvent(output="predicted"),
    )
    flat = [
        part.content
        for msg in pred_inv.output_messages
        for part in getattr(msg, "parts", [])
    ]
    assert flat == ["predicted"], flat


def test_opentelemetry_util_genai_is_runtime_dependency():
    requirements = requires("loongsuite-instrumentation-llama-index")
    assert requirements is not None
    assert any(
        req.split()[0] == "opentelemetry-util-genai" for req in requirements
    ), requirements
