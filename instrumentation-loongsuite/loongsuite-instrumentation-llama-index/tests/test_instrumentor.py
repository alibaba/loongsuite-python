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

Every span-producing assertion has a paired RED check: without the
instrumentor active (before ``instrument`` / after ``uninstrument``), the
same LlamaIndex call produces **zero** OTel spans. This guards against the
test passing for reasons unrelated to the instrumentation.
"""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.llama_index import (
    _GEN_AI_FRAMEWORK,
    _GEN_AI_OPERATION_NAME,
    _GEN_AI_SPAN_KIND,
    _SPAN_KIND_AGENT,
    _SPAN_KIND_CHAIN,
    _SPAN_KIND_EMBEDDING,
    _SPAN_KIND_LLM,
    _SPAN_KIND_TOOL,
    LlamaIndexInstrumentor,
    _classify,
    _span_id_prefix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm():
    from llama_index.core.llms import MockLLM

    return MockLLM(max_tokens=8)


def _chat_once(llm):
    from llama_index.core.llms import ChatMessage

    return llm.chat([ChatMessage(role="user", content="hi there")])


# ---------------------------------------------------------------------------
# Pure classification unit tests (no dispatcher needed)
# ---------------------------------------------------------------------------


def test_span_id_prefix_strips_uuid():
    assert (
        _span_id_prefix("MockLLM.chat-8c7b6315-7a0e-401b-9185-59474d2632c0")
        == "MockLLM.chat"
    )
    assert _span_id_prefix("") == ""


@pytest.mark.parametrize(
    "prefix,expected_kind",
    [
        ("MockLLM.chat", _SPAN_KIND_LLM),
        ("OpenAI.complete", _SPAN_KIND_LLM),
        ("MockEmbedding.get_text_embedding", _SPAN_KIND_EMBEDDING),
        ("VectorIndexRetriever.retrieve", "RETRIEVER"),
        ("LLMRerank.postprocess_nodes", "RERANKER"),
        ("CompactAndRefine.synthesize", "TASK"),
        ("RetrieverQueryEngine.query", "CHAIN"),
        ("ReActAgent.run", "AGENT"),
        # #273: agent-internal machinery must NOT inherit AGENT from the
        # class name -- only a genuine agent invocation (run/chat) is AGENT.
        ("FunctionAgent.call_tool", _SPAN_KIND_TOOL),
        ("ReActAgent.call_tool", _SPAN_KIND_TOOL),
        ("FunctionAgent.take_step", _SPAN_KIND_CHAIN),
        ("FunctionAgent.setup_agent", _SPAN_KIND_CHAIN),
        ("FunctionAgent.finalize", _SPAN_KIND_CHAIN),
        ("ReActAgent.handle_tool_call_results", _SPAN_KIND_CHAIN),
        # Copilot: async/streaming structured prediction are LLM calls.
        ("OpenAI.astructured_predict", _SPAN_KIND_LLM),
        ("OpenAI.stream_structured_predict", _SPAN_KIND_LLM),
        ("OpenAI.astream_structured_predict", _SPAN_KIND_LLM),
    ],
)
def test_classify(prefix, expected_kind):
    kind, _op = _classify(prefix)
    assert kind == expected_kind


def test_agent_internal_methods_are_not_agent_spans():
    # Direct guard for the #273 review: a class named *Agent* must not turn
    # setup/parse/call_tool into AGENT spans. call_tool is TOOL; the rest are
    # internal steps (CHAIN), and only run/chat is the AGENT invocation.
    assert _classify("FunctionAgent.run")[0] == _SPAN_KIND_AGENT
    assert _classify("FunctionAgent.call_tool")[0] == _SPAN_KIND_TOOL
    for internal in ("take_step", "setup_agent", "init_run", "finalize"):
        kind, _op = _classify(f"FunctionAgent.{internal}")
        assert kind != _SPAN_KIND_AGENT, internal
        assert kind == _SPAN_KIND_CHAIN, internal


# ---------------------------------------------------------------------------
# RED: no spans when not instrumented
# ---------------------------------------------------------------------------


def test_red_no_spans_without_instrumentation(span_exporter, tracer_provider):
    """Baseline: driving an LLM chat with NO instrumentor active must not
    produce any OTel spans on our exporter."""
    llm = _mock_llm()
    _chat_once(llm)
    assert span_exporter.get_finished_spans() == ()


# ---------------------------------------------------------------------------
# GREEN: spans appear and nest correctly when instrumented
# ---------------------------------------------------------------------------


def test_green_chat_produces_llm_span(instrument, span_exporter):
    llm = _mock_llm()
    _chat_once(llm)

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    chat_spans = [
        s
        for s in spans
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_LLM
    ]
    assert chat_spans, f"no LLM span among {[s.name for s in spans]}"
    for s in chat_spans:
        assert s.attributes.get(_GEN_AI_FRAMEWORK) == "llama_index"
        assert s.attributes.get(_GEN_AI_OPERATION_NAME) == "chat"


def test_green_chat_complete_share_trace_and_nest(instrument, span_exporter):
    """MockLLM.chat internally calls MockLLM.complete. The two spans must
    share one trace_id and the complete span must be a child of the chat
    span — proving parent_span_id is faithfully mapped."""
    llm = _mock_llm()
    _chat_once(llm)

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 2, [s.name for s in spans]

    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, f"spans split across traces: {trace_ids}"

    chat = next(s for s in spans if s.name.endswith(".chat"))
    complete = next(s for s in spans if s.name.endswith(".complete"))

    # Assert the EXACT parent id, not just 'some exported span': a broken
    # parent_span_id mapping must not be able to satisfy this test.
    assert complete.parent is not None
    assert complete.parent.span_id == chat.context.span_id
    assert complete.context.trace_id == chat.context.trace_id


def test_green_embedding_span(instrument, span_exporter):
    from llama_index.core.embeddings import MockEmbedding

    emb = MockEmbedding(embed_dim=4)
    emb.get_text_embedding("hello world")

    spans = span_exporter.get_finished_spans()
    emb_spans = [
        s
        for s in spans
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_EMBEDDING
    ]
    assert emb_spans, f"no EMBEDDING span among {[s.name for s in spans]}"


# ---------------------------------------------------------------------------
# RED after uninstrument: teardown must stop span production
# ---------------------------------------------------------------------------


def test_red_uninstrument_stops_spans(span_exporter, tracer_provider):
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    llm = _mock_llm()
    _chat_once(llm)
    assert len(span_exporter.get_finished_spans()) >= 1

    instrumentor.uninstrument()
    span_exporter.clear()

    _chat_once(_mock_llm())
    assert span_exporter.get_finished_spans() == (), (
        "spans still produced after uninstrument"
    )


# ---------------------------------------------------------------------------
# Lifecycle: double instrument / uninstrument is safe
# ---------------------------------------------------------------------------


def test_instrument_is_idempotent_on_uninstrument(
    span_exporter, tracer_provider
):
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    instrumentor.uninstrument()
    # second uninstrument must not raise
    instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Content capture is governed by the shared GenAI util's switch (#273)
# ---------------------------------------------------------------------------


def _chat_span(span_exporter):
    return next(
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_LLM
    )


def test_content_captured_when_shared_util_enables_span_content(
    instrument, span_exporter, monkeypatch
):
    # SPAN_ONLY via the standard shared-util env => input messages on the span.
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    _chat_once(_mock_llm())
    span = _chat_span(span_exporter)
    assert "gen_ai.input.messages" in span.attributes


def test_content_suppressed_when_shared_util_disables_content(
    instrument, span_exporter, monkeypatch
):
    # NO_CONTENT (the shared-util default) => structural span but no messages.
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT"
    )
    _chat_once(_mock_llm())
    span = _chat_span(span_exporter)
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
    # the structural span itself is still emitted
    assert span.attributes.get(_GEN_AI_FRAMEWORK) == "llama_index"


# ---------------------------------------------------------------------------
# Uninstrument must not strand spans that were open when it ran (Copilot #2)
# ---------------------------------------------------------------------------


def test_uninstrument_drains_open_spans(span_exporter, tracer_provider):
    """A span left open at uninstrument time must still be ended (exported),
    not stranded because the handler was detached before it closed."""
    from llama_index.core.instrumentation import get_dispatcher

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )

    dispatcher = get_dispatcher()
    # Manually open a span through the dispatcher and DO NOT close it.
    dispatcher.span_enter(
        id_="ManualThing.run-abc", bound_args=None, instance=None
    )
    assert span_exporter.get_finished_spans() == (), "span ended too early"

    # Uninstrument while that span is still open: it must be drained (ended).
    instrumentor.uninstrument()
    ended = span_exporter.get_finished_spans()
    assert any(s.name == "ManualThing.run" for s in ended), (
        f"open span was stranded, not drained: {[s.name for s in ended]}"
    )
