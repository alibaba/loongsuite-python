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

"""Unit tests for MAFSemanticProcessor span enrichment.

Tests verify that the processor correctly:
- Injects ``gen_ai.span.kind`` for each MAF span type.
- Copies registry-defined MAF attributes (``workflow.name`` → ``gen_ai.workflow.name``).
- Reclassifies ``executor.process`` spans by ``executor.type``.
- Normalizes ``gen_ai.provider.name``.
- Backfills ``gen_ai.response.time_to_first_token`` from streaming events.
- Leaves successful spans with the SDK's default status, preserves ``ERROR`` on failure.
- Leaves metrics to Microsoft Agent Framework's native instruments.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.microsoft_agent_framework.semantic_conventions import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_TTFT,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    MAF_LIVE_SPAN_MARKER,
    MAF_PROVIDER_NAME,
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
from opentelemetry.trace import SpanKind, StatusCode


def _setup():
    """Return ``(tracer_provider, tracer, exporter)`` with the MAF processor."""
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = MAFSemanticProcessor(capture_sensitive_data=False)
    tp.add_span_processor(processor)
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("test")
    return tp, tracer, exporter, processor


def _setup_exporter_before_processor():
    """Simulate exporters registered before the MAF semantic processor."""
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = MAFSemanticProcessor(capture_sensitive_data=False)
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tp.add_span_processor(processor)
    tracer = tp.get_tracer("test")
    return tp, tracer, exporter, processor


def _flush(exporter):
    # Force spans to be exported to the in-memory exporter.
    return exporter.get_finished_spans()


def test_processor_does_not_register_cumulative_gauges():
    meter_provider = MagicMock()
    processor = MAFSemanticProcessor(
        meter_provider=meter_provider,
        slow_threshold_ms=250,
        metrics_enabled=True,
        capture_sensitive_data=True,
    )

    assert processor._capture_sensitive is True
    meter_provider.get_meter.assert_not_called()


def _mark_maf_span(span):
    setattr(span, MAF_LIVE_SPAN_MARKER, MAF_PROVIDER_NAME)


def test_llm_span_gets_llm_kind_and_chat_operation():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        _mark_maf_span(span)
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


def test_live_marker_set_after_start_is_visible_to_exporter_snapshot():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        # The MAF bridge marks live spans after the SDK has called
        # SpanProcessor.on_start. on_end enrichments must update the
        # ReadableSpan snapshot too, otherwise downstream exporters on newer
        # SDKs will not see the fields.
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4o")

    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.LLM
    assert spans[0].kind == SpanKind.CLIENT


def test_tool_span_gets_tool_kind():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("execute_tool get_weather") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.EXECUTE_TOOL)
        span.set_attribute("gen_ai.tool.name", "get_weather")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.TOOL


def test_embedding_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "embeddings text-embedding-3-small"
    ) as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.EMBEDDINGS)
        span.set_attribute("gen_ai.request.model", "text-embedding-3-small")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.EMBEDDING


def test_agent_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent my-agent") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute("gen_ai.agent.name", "my-agent")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.AGENT


def test_workflow_run_span_gets_workflow_kind_and_invoke_workflow_op():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("workflow.run abc-123") as span:
        span.set_attribute("workflow.id", "abc-123")
        span.set_attribute("workflow.name", "MyWorkflow")
        span.set_attribute("workflow.description", "d")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.WORKFLOW
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.WORKFLOW
    # Only registry-defined workflow attributes are copied into gen_ai.*.
    assert s.attributes.get("gen_ai.workflow.name") == "MyWorkflow"
    assert s.attributes.get("workflow.id") == "abc-123"
    assert "gen_ai.workflow.id" not in s.attributes


def test_workflow_start_attrs_are_exported_if_exporter_runs_first():
    tp, tracer, exporter, _ = _setup_exporter_before_processor()
    with tracer.start_as_current_span(
        "workflow.run abc-123",
        attributes={
            "workflow.id": "abc-123",
            "workflow.name": "MyWorkflow",
        },
    ):
        pass
    s = _flush(exporter)[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.WORKFLOW
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.WORKFLOW
    assert s.attributes.get("gen_ai.workflow.name") == "MyWorkflow"
    assert s.attributes.get("workflow.id") == "abc-123"
    assert "gen_ai.workflow.id" not in s.attributes


def test_executor_process_function_executor_stays_workflow_operation():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("executor.process fid-1") as span:
        # MAF does NOT set gen_ai.operation.name for executor.process spans.
        span.set_attribute("executor.id", "fid-1")
        span.set_attribute("executor.type", "FunctionExecutor")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.WORKFLOW
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.WORKFLOW
    assert s.attributes.get("executor.id") == "fid-1"
    assert "gen_ai.task.name" not in s.attributes


def test_executor_start_attrs_are_exported_if_exporter_runs_first():
    tp, tracer, exporter, _ = _setup_exporter_before_processor()
    with tracer.start_as_current_span(
        "executor.process fid-1",
        attributes={
            "executor.id": "fid-1",
            "executor.type": "FunctionExecutor",
        },
    ):
        pass
    s = _flush(exporter)[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.WORKFLOW
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.WORKFLOW
    assert s.attributes.get("executor.id") == "fid-1"
    assert "gen_ai.task.name" not in s.attributes


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
    assert (
        s.attributes.get(GEN_AI_PROVIDER_NAME) == "microsoft.agent_framework"
    )


def test_executor_process_unknown_executor_stays_workflow_operation():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("executor.process xid") as span:
        span.set_attribute("executor.id", "xid")
        span.set_attribute("executor.type", "SomeOtherExecutor")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.WORKFLOW
    assert s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.WORKFLOW


def test_message_send_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("message.send m-1") as span:
        span.set_attribute("message.source_id", "src")
        span.set_attribute("message.target_id", "tgt")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.WORKFLOW
    assert s.attributes.get("message.source_id") == "src"
    assert "gen_ai.message.source_id" not in s.attributes


def test_provider_normalization_azure_openai_to_openai():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "azure_openai")
    spans = _flush(exporter)
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "openai"


def test_ttft_backfill_from_first_event():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        _mark_maf_span(span)
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


def test_ttft_backfill_skips_exception_event():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "openai")
        span.add_event("exception", {"exception.type": "RuntimeError"})
    spans = _flush(exporter)
    assert GEN_AI_RESPONSE_TTFT not in spans[0].attributes


def test_finish_reasons_json_string_normalized_to_array():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, '["tool_call"]')
    spans = _flush(exporter)
    # Stable across OTel SDKs that expose array attributes as tuple or list.
    assert list(spans[0].attributes.get(GEN_AI_RESPONSE_FINISH_REASONS)) == [
        "tool_calls"
    ]


def test_output_messages_finish_reason_added_for_agent_span():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent planner") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute(
            "gen_ai.output.messages",
            json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": "done"}],
                    }
                ]
            ),
        )
    [exported] = _flush(exporter)
    messages = json.loads(exported.attributes["gen_ai.output.messages"])
    assert messages[0]["finish_reason"] == "stop"


def test_agent_input_output_boundary_filters_intermediate_messages():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent planner") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps(
                [
                    {
                        "role": "system",
                        "parts": [{"type": "text", "content": "system"}],
                    },
                    {
                        "role": "user",
                        "parts": [{"type": "text", "content": "weather?"}],
                    },
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "lookup",
                                "arguments": {"city": "Hangzhou"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "type": "tool_call_response",
                                "id": "call-1",
                                "response": "sunny",
                            }
                        ],
                    },
                ]
            ),
        )
        span.set_attribute(
            "gen_ai.output.messages",
            json.dumps(
                [
                    {
                        "role": "assistant",
                        "finish_reason": "tool_call",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "lookup",
                                "arguments": {"city": "Hangzhou"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "type": "tool_call_response",
                                "id": "call-1",
                                "response": "sunny",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "reasoning",
                                "content": "hidden scratchpad",
                            },
                            {
                                "type": "text",
                                "content": "Hangzhou is sunny.",
                            },
                        ],
                    },
                ]
            ),
        )
    [exported] = _flush(exporter)

    input_messages = json.loads(exported.attributes["gen_ai.input.messages"])
    assert [message["role"] for message in input_messages] == ["user"]
    assert input_messages[0]["parts"][0]["content"] == "weather?"

    output_messages = json.loads(exported.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"] == [
        {"type": "text", "content": "Hangzhou is sunny."}
    ]
    assert output_messages[0]["finish_reason"] == "stop"


def test_output_messages_finish_reason_tool_call_is_normalized():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        _mark_maf_span(span)
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(
            "gen_ai.output.messages",
            json.dumps(
                [
                    {
                        "role": "assistant",
                        "finish_reason": "tool_call",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "lookup",
                                "arguments": {"q": "x"},
                            }
                        ],
                    }
                ]
            ),
        )
    [exported] = _flush(exporter)
    messages = json.loads(exported.attributes["gen_ai.output.messages"])
    assert messages[0]["finish_reason"] == "tool_calls"


def test_react_step_span_classification():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("react step") as span:
        _mark_maf_span(span)
        # When emitted by our react_step_patch, the handler sets
        # gen_ai.operation.name=react and gen_ai.span.kind=STEP itself. But
        # when a marked MAF span reaches the processor without op set, it
        # should classify it as STEP/react.
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


def test_instrument_rolls_back_bridge_when_processor_registration_fails(
    monkeypatch,
):
    import opentelemetry.instrumentation.microsoft_agent_framework as maf

    tracer_provider = MagicMock()
    tracer_provider.add_span_processor.side_effect = RuntimeError(
        "registration failed"
    )
    apply_bridge = MagicMock()
    revert_bridge = MagicMock()
    monkeypatch.setattr(maf, "apply_util_genai_bridge", apply_bridge)
    monkeypatch.setattr(maf, "revert_util_genai_bridge", revert_bridge)

    inst = maf.MicrosoftAgentFrameworkInstrumentor()
    with pytest.raises(RuntimeError, match="registration failed"):
        inst._instrument(
            tracer_provider=tracer_provider,
            react_step_enabled=False,
        )

    apply_bridge.assert_called_once_with(
        tracer_provider=tracer_provider,
        meter_provider=None,
    )
    revert_bridge.assert_called_once_with()
    assert inst._processor is None


def test_non_maf_span_is_left_untouched():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("http request") as span:
        span.set_attribute("http.method", "GET")
    spans = _flush(exporter)
    s = spans[0]
    assert GEN_AI_SPAN_KIND not in s.attributes
    assert GEN_AI_OPERATION_NAME not in s.attributes


def test_autogen_span_with_overlapping_agent_operation_is_left_untouched():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("invoke_agent assistant") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "autogen")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_PROVIDER_NAME) == "autogen"
    assert GEN_AI_SPAN_KIND not in s.attributes


def test_autogen_llm_span_with_private_marker_is_left_untouched():
    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span("chat gpt-4o") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.CHAT)
        span.set_attribute(GEN_AI_PROVIDER_NAME, "openai")
        setattr(span, "_loongsuite_autogen_framework", "autogen")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_PROVIDER_NAME) == "openai"
    assert GEN_AI_SPAN_KIND not in s.attributes
    assert "_loongsuite_autogen_framework" not in s.attributes
    assert s.kind == SpanKind.INTERNAL


def test_foreign_agent_spans_are_left_untouched():
    tp, tracer, exporter, _ = _setup()
    for provider_name in ("langchain", "agentscope"):
        with tracer.start_as_current_span(
            f"invoke_agent {provider_name}-agent"
        ) as span:
            span.set_attribute(
                GEN_AI_OPERATION_NAME, GenAIOperation.INVOKE_AGENT
            )
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider_name)
            span.set_attribute("gen_ai.agent.name", f"{provider_name}-agent")

    spans = _flush(exporter)
    assert len(spans) == 2
    for span in spans:
        assert GEN_AI_SPAN_KIND not in span.attributes


def test_dict_attribute_is_serialized_via_gen_ai_json_dumps():
    """Dict/list GenAI values are JSON-serialized before SDK export."""
    from opentelemetry.instrumentation.microsoft_agent_framework import (
        span_processor as sp,
    )

    tp, tracer, exporter, _ = _setup()
    message = {"role": "user", "content": "hello"}

    # Drive the rename path through _set_attr (the same path on_end uses after
    # the SDK has stopped accepting set_attribute calls).
    with tracer.start_as_current_span("workflow.run xyz") as span:
        sp._set_attr(span, "gen_ai.input.messages", message)
    spans = _flush(exporter)
    s = spans[0]
    val = s.attributes.get("gen_ai.input.messages")
    assert isinstance(val, str)
    assert "user" in val and "hello" in val


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


def test_mcp_tool_call_span_classified_as_mcp_execute_tool():
    """MCP spans emitted by MAF's ``create_mcp_client_span`` carry no
    ``gen_ai.operation.name``; their name is ``{mcp.method.name} {target}``
    (unbounded), so they must be detected via the ``mcp.method.name``
    attribute and classified as ``(MCP, execute_tool)``. Regression for [M1].
    """
    from opentelemetry.trace import SpanKind

    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "tools/call get_weather", kind=SpanKind.CLIENT
    ) as span:
        _mark_maf_span(span)
        # MAF writes mcp.method.name (no gen_ai.operation.name).
        span.set_attribute("mcp.method.name", "tools/call")
        span.set_attribute("mcp.session.id", "sess-1")
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.MCP
    assert (
        s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.EXECUTE_TOOL
    )
    assert s.attributes.get("gen_ai.tool.name") == "get_weather"


def test_mcp_lifecycle_span_is_not_classified_as_genai():
    """MCP protocol lifecycle spans are not GenAI tool executions."""
    from opentelemetry.trace import SpanKind

    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "initialize", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("mcp.method.name", "initialize")
        span.set_attribute("mcp.protocol.version", "2024-11-05")
    spans = _flush(exporter)
    s = spans[0]
    assert GEN_AI_SPAN_KIND not in s.attributes
    assert GEN_AI_OPERATION_NAME not in s.attributes
    assert s.attributes.get("mcp.method.name") == "initialize"


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


def test_mcp_tool_call_stays_mcp_when_maf_writes_execute_tool():
    """[P1] regression: MAF emits ``gen_ai.operation.name=execute_tool`` on the
    MCP ``tools/call`` inner span (its ``create_mcp_client_span`` reuses the
    tool-call op name even though it sets ``mcp.method.name``). The processor
    must keep the logical span kind as ``MCP`` so it is consistent with the
    LoongSuite GenAI registry while still using upstream ``execute_tool``.

    Non-tool MCP lifecycle spans are intentionally left as protocol spans.
    """
    from opentelemetry.trace import SpanKind

    tp, tracer, exporter, _ = _setup()
    with tracer.start_as_current_span(
        "tools/call slow_summary", kind=SpanKind.CLIENT
    ) as span:
        _mark_maf_span(span)
        # MAF writes both mcp.method.name AND gen_ai.operation.name=execute_tool.
        span.set_attribute("mcp.method.name", "tools/call")
        span.set_attribute(GEN_AI_OPERATION_NAME, GenAIOperation.EXECUTE_TOOL)
    spans = _flush(exporter)
    s = spans[0]
    assert s.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.MCP
    assert (
        s.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.EXECUTE_TOOL
    )
    assert s.attributes.get("gen_ai.tool.name") == "slow_summary"


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

    processor = MAFSemanticProcessor(capture_sensitive_data=False)
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
