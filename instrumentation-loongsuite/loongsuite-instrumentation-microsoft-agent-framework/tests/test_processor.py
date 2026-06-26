"""Unit tests for MAFSemanticProcessor span enrichment.

Tests verify that the processor correctly:
- Injects ``gen_ai.span.kind`` for each MAF span type.
- Renames MAF private-prefix attributes (``workflow.id`` → ``gen_ai.workflow.id`` …).
- Reclassifies ``executor.process`` spans by ``executor.type``.
- Normalizes ``gen_ai.provider.name``.
- Backfills ``gen_ai.response.time_to_first_token`` from streaming events.
- Leaves successful spans with the SDK's default status, preserves ``ERROR`` on failure.
- Aggregates metrics counters.
"""

from __future__ import annotations

import time

from opentelemetry.instrumentation.microsoft_agent_framework.semantic_conventions import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_TTFT,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAIOperation,
    GenAISpanKind,
)
from opentelemetry.instrumentation.microsoft_agent_framework.span_processor import (
    MAFSemanticProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, Status, StatusCode


def _setup():
    """Return ``(tracer_provider, tracer, exporter)`` with the MAF processor."""
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = MAFSemanticProcessor(
        meter_provider=None,
        metrics_enabled=False,
        capture_sensitive_data=False,
    )
    tp.add_span_processor(processor)
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("test")
    return tp, tracer, exporter, processor


def _flush(exporter):
    # Force spans to be exported to the in-memory exporter.
    return exporter.get_finished_spans()


def test_llm_span_gets_llm_kind_and_chat_operation():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4o")
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, 10)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, 20)
    spans = _flush(exporter)
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.LLM
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.CHAT
    assert s.kind == SpanKind.CLIENT
    assert s.status.status_code == StatusCode.UNSET


def test_tool_span_gets_tool_kind():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("execute_tool get_weather") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.EXECUTE_TOOL)
        span.set_attribute("gen_ai.tool.name", "get_weather")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.TOOL


def test_embedding_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "embeddings text-embedding-3-small"
    ) as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.EMBEDDINGS)
        span.set_attribute("gen_ai.request.model", "text-embedding-3-small")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.EMBEDDING


def test_agent_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent my-agent") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute("gen_ai.agent.name", "my-agent")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.AGENT


def test_workflow_run_span_gets_chain_kind_and_workflow_op():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("workflow.run abc-123") as span:
        span.set_attribute("workflow.id", "abc-123")
        span.set_attribute("workflow.name", "MyWorkflow")
        span.set_attribute("workflow.description", "d")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.CHAIN
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.WORKFLOW
    # MAF private-prefix renamed to gen_ai.* canonical keys
    assert s.attributes.get("gen_ai.workflow.id") == "abc-123"
    assert s.attributes.get("gen_ai.workflow.name") == "MyWorkflow"
    # The original MAF private key should be removed (best-effort).
    assert "workflow.id" not in s.attributes


def test_executor_process_function_executor_becomes_task():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("executor.process fid-1") as span:
        # MAF does NOT set gen_ai.operation.name for executor.process spans.
        span.set_attribute("executor.id", "fid-1")
        span.set_attribute("executor.type", "FunctionExecutor")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.TASK
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.TASK
    assert s.attributes.get("gen_ai.task.name") == "fid-1"
    assert s.attributes.get("gen_ai.task.type") == "FunctionExecutor"


def test_executor_process_agent_executor_becomes_agent():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("executor.process aid-1") as span:
        span.set_attribute("executor.id", "aid-1")
        span.set_attribute("executor.type", "AgentExecutor")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.AGENT
    assert (
        s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.INVOKE_AGENT
    )


def test_executor_process_unknown_executor_stays_chain():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("executor.process xid") as span:
        span.set_attribute("executor.id", "xid")
        span.set_attribute("executor.type", "SomeOtherExecutor")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.CHAIN
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.WORKFLOW


def test_message_send_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("message.send m-1") as span:
        span.set_attribute("message.source_id", "src")
        span.set_attribute("message.target_id", "tgt")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.CHAIN
    assert s.attributes.get("gen_ai.message.source_id") == "src"
    assert s.attributes.get("gen_ai.message.target_id") == "tgt"


def test_provider_normalization_azure_openai_to_openai():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "azure_openai")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "openai"


def test_ttft_backfill_from_first_event():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "openai")
        # Emit a streaming-chunk event a bit after start.
        # The SDK uses wall-clock ns for event timestamps.
        # We just need the event to be present; the processor will compute the
        # delta from start_time.
        time.sleep(0.01)
        span.add_event("streaming.chunk")
    spans = _flush(exporter)
    s = spans[0]
    ttft = s.attributes.get(GEN_AI_RESPONSE_TTFT)
    assert ttft is not None and ttft > 0


def test_finish_reasons_json_string_normalized_to_array():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, '["stop"]')
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)


def _setup_with_metrics():
    """Like ``_setup`` but with metrics aggregation enabled."""
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = MAFSemanticProcessor(
        meter_provider=None, metrics_enabled=True, capture_sensitive_data=False
    )
    tp.add_span_processor(processor)
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("test")
    return tp, tracer, exporter, processor


def test_error_span_keeps_error_status_and_increments_error_counter():
    tp, tracer, exporter, processor = _setup_with_metrics()
    try:
        with tracer.start_as_current_span("chat gpt-4o") as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
            span.set_attribute(GEN_AI_PROVIDER_NAME, "openai")
            span.set_status(Status(StatusCode.ERROR, "boom"))
            span.set_attribute("error.type", "RateLimitError")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    spans = _flush(exporter)
    assert spans[0].status.status_code == StatusCode.ERROR
    # Metric error counter incremented (model not set -> "unknown").
    found = False
    for (m, k), count in processor._counters.calls_error_count.items():
        if k == GenAISpanKind.LLM and count == 1:
            found = True
            break
    assert found, "error counter not incremented"


def test_metrics_counters_incremented_on_llm_span():
    tp, tracer, exporter, processor = _setup_with_metrics()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute("gen_ai.request.model", "gpt-4o")
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, 5)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, 7)
    _ = _flush(exporter)
    assert processor._counters.calls_count[("gpt-4o", GenAISpanKind.LLM)] == 1
    assert (
        processor._counters.llm_usage_input_tokens[
            ("gpt-4o", GenAISpanKind.LLM)
        ]
        == 5
    )
    assert (
        processor._counters.llm_usage_output_tokens[
            ("gpt-4o", GenAISpanKind.LLM)
        ]
        == 7
    )


def test_react_step_span_classification():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("react step"):
        # When emitted by our react_step_patch, the handler sets
        # gen_ai.operation.name=react and gen_ai.span.kind=STEP itself. But
        # if the processor sees a "react step" name without op set, it should
        # classify it as STEP/react as a fallback.
        pass
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.STEP
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.REACT


def test_uninstrument_releases_processor():
    from opentelemetry.instrumentation.microsoft_agent_framework import (
        MicrosoftAgentFrameworkInstrumentor,
    )

    inst = MicrosoftAgentFrameworkInstrumentor()
    inst._uninstrument()
    assert inst._processor is None
    assert inst._react_applied is False


def test_uninstrument_removes_registered_processor_from_provider():
    from opentelemetry.instrumentation.microsoft_agent_framework import (
        MicrosoftAgentFrameworkInstrumentor,
    )

    tp = TracerProvider()
    inst = MicrosoftAgentFrameworkInstrumentor()
    inst._instrument(tracer_provider=tp, react_step_enabled=False)
    processor = inst._processor
    assert processor is not None
    inst._uninstrument()
    asp = getattr(tp, "_active_span_processor", None)
    procs = (
        getattr(asp, "_span_processors", None)
        if asp is not None
        else getattr(tp, "_span_processors", None)
    )
    assert procs is not None and processor not in procs


def test_non_maf_span_is_left_untouched():
    tp, tracer, exporter, processor = _setup_with_metrics()
    with tracer.start_as_current_span("http request") as span:
        span.set_attribute("http.method", "GET")
    spans = _flush(exporter)
    s = spans[0]
    assert GEN_AI_SPAN_KIND not in s.attributes
    assert GEN_AI_OPERATION_NAME not in s.attributes
    assert not processor._counters.calls_count


def test_maf_dict_attribute_is_serialized_via_gen_ai_json_dumps():
    """Dict/list attribute values written into ``_attributes`` by MAF under
    private prefixes must be JSON-serialized via
    ``opentelemetry.util.genai.utils.gen_ai_json_dumps`` when renamed to
    ``gen_ai.*`` keys, because OTel SDKs reject arbitrary dict attribute
    values. Hard constraint #2.

    We exercise this directly through ``_set_attr`` because the SDK drops
    dict values at ``set_attribute`` time (logged as a warning), which is
    exactly the scenario our coercion defends against when MAF mutates
    ``_attributes`` directly itself (it does so in several internal paths).
    """
    from opentelemetry.instrumentation.microsoft_agent_framework import (
        span_processor as sp,
    )

    tp, tracer, exporter, _ = _setup()
    workflow_def = {"nodes": ["a", "b"], "edges": [{"from": "a", "to": "b"}]}

    # Drive the rename path through _set_attr (the same path on_end uses after
    # the SDK has stopped accepting set_attribute calls).
    with tracer.start_as_current_span("workflow.run xyz") as span:
        sp._set_attr(span, "gen_ai.workflow.definition", workflow_def)
    spans = _flush(exporter)
    s = spans[0]
    val = s.attributes.get("gen_ai.workflow.definition")
    assert isinstance(val, str)
    assert "nodes" in val and "edges" in val


def test_safe_dumps_uses_gen_ai_json_dumps():
    """Unit test the helper directly."""
    from opentelemetry.instrumentation.microsoft_agent_framework import (
        span_processor as sp,
    )

    # gen_ai_json_dumps round-trips standard JSON; our wrapper must preserve it.
    out = sp._safe_dumps({"a": 1, "b": [1, 2]})
    assert isinstance(out, str)
    assert "a" in out and "b" in out


def test_safe_dumps_truncates_at_4kb():
    """_safe_dumps must cap output at 4096 chars (execute.md single-field cap)."""
    from opentelemetry.instrumentation.microsoft_agent_framework import (
        span_processor as sp,
    )

    big = {"k": "x" * 10_000}
    out = sp._safe_dumps(big)
    assert isinstance(out, str)
    assert len(out) <= 4096


def test_mcp_span_classified_as_client():
    """MCP spans emitted by MAF's ``create_mcp_client_span`` carry no
    ``gen_ai.operation.name``; their name is ``{mcp.method.name} {target}``
    (unbounded), so they must be detected via the ``mcp.method.name``
    attribute and classified as ``(CLIENT, mcp)``. Regression for [M1].
    """
    from opentelemetry.trace import SpanKind

    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "tools/call get_weather", kind=SpanKind.CLIENT
    ) as span:
        # MAF writes mcp.method.name (no gen_ai.operation.name).
        span.set_attribute("mcp.method.name", "tools/call")
        span.set_attribute("mcp.session.id", "sess-1")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.CLIENT
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.MCP


def test_mcp_span_via_client_kind_and_mcp_attr_fallback():
    """Fallback path: a CLIENT span with any ``mcp.*`` attribute (but missing
    ``mcp.method.name``) is still classified as MCP."""
    from opentelemetry.trace import SpanKind

    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "initialize", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("mcp.protocol.version", "2024-11-05")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.CLIENT
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.MCP


def test_non_mcp_client_span_is_not_misclassified_as_mcp():
    """A CLIENT span without any ``mcp.*`` attribute must NOT be classified as
    MCP — guards against false positives on unrelated client spans."""
    from opentelemetry.trace import SpanKind

    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "http request", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("http.method", "GET")
    spans = _flush(exporter)
    s = spans[0]
    assert GEN_AI_SPAN_KIND not in s.attributes
    assert GEN_AI_OPERATION_NAME not in s.attributes


def test_mcp_span_op_name_overridden_to_mcp_when_maf_writes_execute_tool():
    """[P1] regression: MAF emits ``gen_ai.operation.name=execute_tool`` on the
    MCP ``tools/call`` inner span (its ``create_mcp_client_span`` reuses the
    tool-call op name even though it sets ``mcp.method.name``). The processor
    must override the op name to ``mcp`` so the span is not mislabeled as a
    TOOL call in the ARMS pipeline.

    Before the fix, ``on_end``'s op-name override only fired when
    ``span_kind in {TASK, AGENT}`` — CLIENT (MCP) was missing, so the inner
    span kept MAF's ``execute_tool`` value.
    """
    from opentelemetry.trace import SpanKind

    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "tools/call slow_summary", kind=SpanKind.CLIENT
    ) as span:
        # MAF writes both mcp.method.name AND gen_ai.operation.name=execute_tool.
        span.set_attribute("mcp.method.name", "tools/call")
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.EXECUTE_TOOL)
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.CLIENT
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.MCP


def test_provider_normalization_keeps_framework_provider_separate():
    """Framework-level provider names must not be collapsed to ``openai``.

    MAF can route to multiple underlying providers, so ``microsoft.agent_framework``
    is lower-cased and kept distinct instead of pretending every MAF span used
    OpenAI.
    """
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent my-agent") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "microsoft.agent_framework")
    spans = _flush(exporter)
    assert (
        spans[0].attributes.get(GEN_AI_PROVIDER_NAME)
        == "microsoft.agent_framework"
    )


def test_provider_normalization_case_insensitive_variant():
    """Unknown provider values should lower-case to avoid metric cardinality."""
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent my-agent") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "Microsoft.Agent_Framework")
    spans = _flush(exporter)
    assert (
        spans[0].attributes.get(GEN_AI_PROVIDER_NAME)
        == "microsoft.agent_framework"
    )


def test_provider_normalization_list_wrapped_value():
    """[P3] OTel attributes may be a sequence of strings. MAF occasionally
    writes ``gen_ai.provider.name`` as ``["microsoft.agent_framework"]`` on
    AGENT spans. The normalizer should unwrap the sequence and normalize the
    first element's casing."""
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent my-agent") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, ["Microsoft.Agent_Framework"])
    spans = _flush(exporter)
    assert (
        spans[0].attributes.get(GEN_AI_PROVIDER_NAME)
        == "microsoft.agent_framework"
    )


def test_force_flush_sweeps_stale_live_spans():
    class _StartedSpan:
        start_time = 0

    processor = MAFSemanticProcessor(
        meter_provider=None,
        metrics_enabled=False,
        capture_sensitive_data=False,
    )
    processor._live_spans["deadbeef"] = _StartedSpan()
    processor._span_parents["deadbeef"] = None

    assert processor.force_flush()
    assert "deadbeef" not in processor._live_spans
    assert "deadbeef" not in processor._span_parents


def test_instrument_prepends_processor_before_existing_exporters():
    """[P5] When exporters were registered before ``instrument()`` (the common
    bootstrap order: provider → exporter processor → instrument()), the MAF
    semantic processor must run FIRST in the pipeline so its ``on_end``
    enrichments (gen_ai.span.kind, operation.name, rename map,
    provider normalization) are visible to those exporters. Without the
    prepend, an exporter that captured the span before our ``on_end`` would
    ship an un-enriched span.
    """
    from opentelemetry.instrumentation.microsoft_agent_framework import (
        MicrosoftAgentFrameworkInstrumentor,
    )

    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    # Bootstrap-style order: exporter processor FIRST, then our instrumentor.
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    inst = MicrosoftAgentFrameworkInstrumentor()
    # Skip MAF enable_instrumentation (MAF not installed in this env).
    inst._instrument(
        tracer_provider=tp,
        react_step_enabled=False,
    )
    try:
        asp = getattr(tp, "_active_span_processor", None)
        procs = (
            getattr(asp, "_span_processors", None)
            if asp is not None
            else getattr(tp, "_span_processors", None)
        )
        assert procs is not None and len(procs) >= 2
        from opentelemetry.instrumentation.microsoft_agent_framework.span_processor import (
            MAFSemanticProcessor as _Proc,
        )

        assert isinstance(procs[0], _Proc), (
            "MAFSemanticProcessor must be at index 0 so it runs before "
            "exporter processors"
        )
    finally:
        inst._uninstrument()
