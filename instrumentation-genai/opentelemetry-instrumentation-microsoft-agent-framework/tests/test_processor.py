"""Unit tests for MAFSemanticProcessor span enrichment.

Tests verify that the processor correctly:
- Injects ``gen_ai.span.kind`` for each MAF span type.
- Renames MAF private-prefix attributes (``workflow.id`` → ``gen_ai.workflow.id`` …).
- Reclassifies ``executor.process`` spans by ``executor.type``.
- Normalizes ``gen_ai.provider.name``.
- Backfills ``gen_ai.response.time_to_first_token`` from streaming events.
- Sets ``StatusCode.OK`` on success, preserves ``ERROR`` on failure.
- Aggregates metrics counters.
"""

from __future__ import annotations

import time

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Status, StatusCode

from opentelemetry.instrumentation.microsoft_agent_framework.span_processor import (
    MAFSemanticProcessor,
)
from opentelemetry.instrumentation.microsoft_agent_framework.semantic_conventions import (
    GEN_AI_FRAMEWORK,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_RESPONSE_TTFT,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USER_TTFT,
    GenAIOperation,
    GenAISpanKind,
)


def _setup():
    """Return ``(tracer_provider, tracer, exporter)`` with the MAF processor."""
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = MAFSemanticProcessor(
        meter_provider=None, metrics_enabled=False, capture_sensitive_data=False
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
    assert s.attributes.get(GEN_AI_FRAMEWORK) == "microsoft-agent-framework"
    assert s.status.status_code == StatusCode.OK


def test_tool_span_gets_tool_kind():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("execute_tool get_weather") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.EXECUTE_TOOL)
        span.set_attribute("gen_ai.tool.name", "get_weather")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.TOOL


def test_embedding_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
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
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.INVOKE_AGENT


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
    user_ttft = s.attributes.get(GEN_AI_USER_TTFT)
    assert ttft is not None and ttft > 0
    assert user_ttft == ttft


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
    assert processor._counters.llm_usage_input_tokens[("gpt-4o", GenAISpanKind.LLM)] == 5
    assert processor._counters.llm_usage_output_tokens[("gpt-4o", GenAISpanKind.LLM)] == 7


def test_react_step_span_classification():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("react step") as span:
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

    # Without MAF installed we just exercise that _uninstrument does not raise
    # and clears state. We construct the instrumentor and call _uninstrument
    # without _instrument to ensure idempotent teardown.
    inst = MicrosoftAgentFrameworkInstrumentor()
    inst._uninstrument()
    assert inst._processor is None
    assert inst._react_applied is False


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
