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

"""Tests for the MAF util-genai bridge.

These tests use a tiny fake ``agent_framework.observability`` module so they do
not depend on the real MAF package. The important contract is exporter-visible:
attributes must be written before ``span.end()`` snapshots the span.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import types

from opentelemetry import trace
from opentelemetry.instrumentation.microsoft_agent_framework import (
    util_genai_bridge,
)
from opentelemetry.instrumentation.microsoft_agent_framework.semantic_conventions import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_TTFT,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAIOperation,
    GenAISpanKind,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def _install_fake_observability(monkeypatch):
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("fake-maf")

    @contextlib.contextmanager
    def _get_span(attributes, span_name_attribute):
        operation = attributes.get(GEN_AI_OPERATION_NAME, "operation")
        span_name = attributes.get(span_name_attribute, "unknown")
        span = tracer.start_span(f"{operation} {span_name}")
        span.set_attributes(attributes)
        with trace.use_span(
            span,
            end_on_exit=True,
            record_exception=False,
            set_status_on_exception=False,
        ) as current_span:
            yield current_span

    def _start_streaming_span(attributes, span_name_attribute):
        operation = attributes.get(GEN_AI_OPERATION_NAME, "operation")
        span_name = attributes.get(span_name_attribute, "unknown")
        span = tracer.start_span(f"{operation} {span_name}")
        span.set_attributes(attributes)
        return span

    @contextlib.contextmanager
    def _activate_span(span):
        with trace.use_span(span, end_on_exit=False):
            yield

    @contextlib.contextmanager
    def get_function_span(attributes):
        span = tracer.start_span(
            f"{attributes[GEN_AI_OPERATION_NAME]} {attributes['gen_ai.tool.name']}"
        )
        span.set_attributes(attributes)
        with trace.use_span(
            span,
            end_on_exit=True,
            record_exception=False,
            set_status_on_exception=False,
        ) as current_span:
            yield current_span

    @contextlib.contextmanager
    def create_mcp_client_span(method_name, target=None, attributes=None):
        span_name = f"{method_name} {target}" if target else method_name
        span = tracer.start_span(span_name, kind=trace.SpanKind.CLIENT)
        span.set_attribute("mcp.method.name", method_name)
        if attributes:
            span.set_attributes(attributes)
        with trace.use_span(span, end_on_exit=True) as current_span:
            yield current_span

    obs_mod = types.ModuleType("agent_framework.observability")
    obs_mod._get_span = _get_span
    obs_mod._start_streaming_span = _start_streaming_span
    obs_mod._activate_span = _activate_span
    obs_mod.get_function_span = get_function_span
    obs_mod.create_mcp_client_span = create_mcp_client_span
    obs_mod.get_tracer = lambda: tracer

    af_mod = types.ModuleType("agent_framework")
    af_mod.observability = obs_mod
    tools_mod = types.ModuleType("agent_framework._tools")
    tools_mod.get_function_span = get_function_span
    mcp_mod = types.ModuleType("agent_framework._mcp")
    mcp_mod.create_mcp_client_span = create_mcp_client_span
    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.observability", obs_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._tools", tools_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._mcp", mcp_mod)
    util_genai_bridge.revert_util_genai_bridge()
    return obs_mod, exporter


def test_llm_get_span_is_finalized_by_util_genai_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        attributes = {
            GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
            GEN_AI_PROVIDER_NAME: "azure_openai",
            GEN_AI_REQUEST_MODEL: "qwen-plus",
            GEN_AI_USAGE_INPUT_TOKENS: 11,
            GEN_AI_USAGE_OUTPUT_TOKENS: 13,
            GEN_AI_RESPONSE_FINISH_REASONS: '["stop"]',
        }
        with obs_mod._get_span(attributes, GEN_AI_REQUEST_MODEL):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    assert GEN_AI_SPAN_KIND not in attributes
    assert attributes[GEN_AI_PROVIDER_NAME] == "azure_openai"
    span = exporter.get_finished_spans()[0]
    assert span.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.LLM
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.CHAT
    assert span.attributes.get(GEN_AI_PROVIDER_NAME) == "openai"
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 11
    assert span.attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS) == 13
    assert span.attributes.get(GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)
    assert span.kind == trace.SpanKind.CLIENT


def test_streaming_llm_end_wrapper_finalizes_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        span = obs_mod._start_streaming_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
                GEN_AI_REQUEST_MODEL: "qwen-plus",
            },
            GEN_AI_REQUEST_MODEL,
        )
        with obs_mod._activate_span(span):
            pass
        span.end()
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    exported = exporter.get_finished_spans()[0]
    assert exported.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.LLM
    assert exported.attributes.get(GEN_AI_RESPONSE_TTFT) is not None
    assert exported.kind == trace.SpanKind.CLIENT


def test_streaming_error_does_not_emit_fallback_ttft(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        span = obs_mod._start_streaming_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
                GEN_AI_REQUEST_MODEL: "qwen-not-a-real-model",
            },
            GEN_AI_REQUEST_MODEL,
        )
        with obs_mod._activate_span(span):
            pass
        span.set_status(trace.Status(trace.StatusCode.ERROR))
        span.end()
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    exported = exporter.get_finished_spans()[0]
    assert exported.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.LLM
    assert GEN_AI_RESPONSE_TTFT not in exported.attributes


def test_streaming_exception_event_does_not_emit_fallback_ttft(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        span = obs_mod._start_streaming_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
                GEN_AI_REQUEST_MODEL: "qwen-not-a-real-model",
            },
            GEN_AI_REQUEST_MODEL,
        )
        with obs_mod._activate_span(span):
            pass
        span.add_event("exception", {"exception.type": "RuntimeError"})
        span.end()
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    exported = exporter.get_finished_spans()[0]
    assert exported.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.LLM
    assert GEN_AI_RESPONSE_TTFT not in exported.attributes


def test_embedding_span_is_finalized_by_util_genai_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        with obs_mod._get_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.EMBEDDINGS,
                GEN_AI_PROVIDER_NAME: "openai",
                GEN_AI_REQUEST_MODEL: "text-embedding-v4",
                GEN_AI_USAGE_INPUT_TOKENS: 17,
            },
            GEN_AI_REQUEST_MODEL,
        ):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.EMBEDDING
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == (
        GenAIOperation.EMBEDDINGS
    )
    assert span.attributes.get(GEN_AI_REQUEST_MODEL) == "text-embedding-v4"
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 17


def test_tool_span_is_finalized_by_util_genai_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        tools_mod = sys.modules["agent_framework._tools"]
        with tools_mod.get_function_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.EXECUTE_TOOL,
                "gen_ai.tool.name": "city_score",
                "gen_ai.tool.call.id": "call-1",
                "gen_ai.tool.type": "function",
            }
        ):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.TOOL
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == (
        GenAIOperation.EXECUTE_TOOL
    )
    assert span.attributes.get("gen_ai.tool.name") == "city_score"
    assert span.attributes.get("gen_ai.tool.call.id") == "call-1"


def test_agent_span_is_finalized_by_util_genai_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        with obs_mod._get_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
                GEN_AI_PROVIDER_NAME: "microsoft.agent_framework",
                "gen_ai.agent.name": "planner",
                "gen_ai.agent.id": "agent-1",
            },
            "gen_ai.agent.name",
        ):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.AGENT
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == (
        GenAIOperation.INVOKE_AGENT
    )
    assert span.attributes.get("gen_ai.agent.name") == "planner"
    assert span.attributes.get("gen_ai.agent.id") == "agent-1"


def test_agent_span_gets_framework_provider_when_missing(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        with obs_mod._get_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
                "gen_ai.agent.name": "planner",
            },
            "gen_ai.agent.name",
        ):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    assert (
        span.attributes.get(GEN_AI_PROVIDER_NAME)
        == "microsoft.agent_framework"
    )


def test_agent_output_messages_get_finish_reason_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        with obs_mod._get_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
                "gen_ai.agent.name": "planner",
                "gen_ai.output.messages": json.dumps(
                    [
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": "done"}],
                        }
                    ]
                ),
            },
            "gen_ai.agent.name",
        ):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert messages[0]["finish_reason"] == "stop"


def test_agent_boundary_messages_filtered_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        with obs_mod._get_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
                "gen_ai.agent.name": "planner",
                "gen_ai.input.messages": json.dumps(
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
                "gen_ai.output.messages": json.dumps(
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
            },
            "gen_ai.agent.name",
        ):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert [message["role"] for message in input_messages] == ["user"]
    assert input_messages[0]["parts"][0]["content"] == "weather?"

    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert output_messages == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hangzhou is sunny."}],
            "finish_reason": "stop",
        }
    ]


def test_tool_call_finish_reason_is_normalized_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        with obs_mod._get_span(
            {
                GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
                GEN_AI_REQUEST_MODEL: "qwen-plus",
                GEN_AI_RESPONSE_FINISH_REASONS: '["tool_call"]',
                "gen_ai.output.messages": json.dumps(
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
            },
            GEN_AI_REQUEST_MODEL,
        ):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert messages[0]["finish_reason"] == "tool_calls"
    assert span.attributes.get(GEN_AI_RESPONSE_FINISH_REASONS) == (
        "tool_calls",
    )


def test_mcp_span_is_seeded_before_export(monkeypatch):
    obs_mod, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        mcp_mod = sys.modules["agent_framework._mcp"]
        with mcp_mod.create_mcp_client_span("tools/call", "city_score"):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get(GEN_AI_SPAN_KIND) == GenAISpanKind.MCP
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == (
        GenAIOperation.EXECUTE_TOOL
    )
    assert span.attributes.get("gen_ai.tool.name") == "city_score"


def test_mcp_lifecycle_span_is_not_seeded_as_genai(monkeypatch):
    _, exporter = _install_fake_observability(monkeypatch)
    util_genai_bridge.apply_util_genai_bridge()
    try:
        mcp_mod = sys.modules["agent_framework._mcp"]
        with mcp_mod.create_mcp_client_span("initialize"):
            pass
    finally:
        util_genai_bridge.revert_util_genai_bridge()
    span = exporter.get_finished_spans()[-1]
    assert span.attributes.get("mcp.method.name") == "initialize"
    assert GEN_AI_SPAN_KIND not in span.attributes
    assert GEN_AI_OPERATION_NAME not in span.attributes


def test_apply_revert_apply_keeps_single_wrapper_layer(monkeypatch):
    obs_mod, _ = _install_fake_observability(monkeypatch)
    original_get_span = obs_mod._get_span
    original_start_streaming_span = obs_mod._start_streaming_span
    tools_mod = sys.modules["agent_framework._tools"]
    original_tool_span = tools_mod.get_function_span

    util_genai_bridge.apply_util_genai_bridge()
    first_get_span = obs_mod._get_span
    first_streaming = obs_mod._start_streaming_span
    first_tool_span = tools_mod.get_function_span
    util_genai_bridge.revert_util_genai_bridge()
    assert obs_mod._get_span is original_get_span
    assert obs_mod._start_streaming_span is original_start_streaming_span
    assert tools_mod.get_function_span is original_tool_span

    util_genai_bridge.apply_util_genai_bridge()
    try:
        assert obs_mod._get_span is not original_get_span
        assert (
            obs_mod._start_streaming_span is not original_start_streaming_span
        )
        assert tools_mod.get_function_span is not original_tool_span
        assert obs_mod._get_span is not first_get_span
        assert obs_mod._start_streaming_span is not first_streaming
        assert tools_mod.get_function_span is not first_tool_span
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    assert obs_mod._get_span is original_get_span
    assert obs_mod._start_streaming_span is original_start_streaming_span
    assert tools_mod.get_function_span is original_tool_span


def test_apply_skips_when_util_genai_private_helpers_are_unavailable(
    monkeypatch,
):
    obs_mod, _ = _install_fake_observability(monkeypatch)
    original_get_span = obs_mod._get_span

    monkeypatch.setattr(
        util_genai_bridge,
        "_UTIL_GENAI_IMPORT_ERROR",
        ImportError("missing private helper"),
    )

    util_genai_bridge.apply_util_genai_bridge()

    assert obs_mod._get_span is original_get_span


def test_activate_span_wrapper_supports_async_context_manager():
    events = []

    @contextlib.asynccontextmanager
    async def _activate_span(span):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    wrapped = util_genai_bridge._wrap_activate_span(_activate_span)
    span = types.SimpleNamespace()

    async def _run():
        async with wrapped(span):
            events.append("body")

    asyncio.run(_run())

    assert events == ["enter", "body", "exit"]
    assert (
        getattr(span, util_genai_bridge._STREAM_FIRST_TOKEN_ATTR) is not None
    )


def test_legacy_streaming_agent_context_parents_inner_spans(monkeypatch):
    """MAF 1.0 created streaming agent spans after ``execute()``.

    The returned stream then produced inner chat/tool spans without the agent
    span as current context. The legacy bridge patch must create the agent span
    first and activate it while the stream resolves and pulls updates.
    """

    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("fake-legacy-maf")

    class _OtelAttr:
        OPERATION = GEN_AI_OPERATION_NAME
        AGENT_NAME = "gen_ai.agent.name"
        AGENT_ID = "gen_ai.agent.id"
        AGENT_INVOKE_OPERATION = GenAIOperation.INVOKE_AGENT

    class _Settings:
        ENABLED = True
        SENSITIVE_DATA_ENABLED = True

    class _ResponseStream:
        def __init__(self, stream, *, finalizer=None):
            self._stream_source = stream
            self._stream = None
            self._iterator = None
            self._updates = []
            self._finalizer = finalizer
            self._finalized = False
            self._final_result = None
            self._cleanup_hooks = []
            self._cleanup_run = False
            self._transform_hooks = []
            self._result_hooks = []
            self._map_update = None
            self._consumed = False
            self._wrap_inner = False

        @classmethod
        def from_awaitable(cls, awaitable):
            return cls(awaitable)

        async def _get_stream(self):
            if self._stream is None:
                if hasattr(self._stream_source, "__aiter__"):
                    self._stream = self._stream_source
                else:
                    self._stream = await self._stream_source
            return self._stream

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._iterator is None:
                stream = await self._get_stream()
                self._iterator = stream.__aiter__()
            try:
                update = await self._iterator.__anext__()
            except StopAsyncIteration:
                self._consumed = True
                await self._run_cleanup_hooks()
                await self.get_final_response()
                raise
            self._updates.append(update)
            return update

        def __await__(self):
            async def _wrap():
                await self._get_stream()
                return self

            return _wrap().__await__()

        async def get_final_response(self):
            if not self._finalized and not self._consumed:
                async for _ in self:
                    pass
            if not self._finalized:
                if self._finalizer is None:
                    self._final_result = types.SimpleNamespace(
                        messages=[], usage_details=None
                    )
                else:
                    result = self._finalizer(self._updates)
                    if asyncio.iscoroutine(result):
                        result = await result
                    self._final_result = result
                self._finalized = True
            return self._final_result

        def with_cleanup_hook(self, hook):
            self._cleanup_hooks.append(hook)
            return self

        async def _run_cleanup_hooks(self):
            if self._cleanup_run:
                return
            self._cleanup_run = True
            for hook in self._cleanup_hooks:
                result = hook()
                if asyncio.iscoroutine(result):
                    await result

    @contextlib.contextmanager
    def _get_span(attributes, span_name_attribute):
        span = tracer.start_span(
            f"{attributes[GEN_AI_OPERATION_NAME]} {attributes[span_name_attribute]}"
        )
        span.set_attributes(attributes)
        with trace.use_span(span, end_on_exit=True) as current_span:
            yield current_span

    @contextlib.contextmanager
    def get_function_span(attributes):
        span = tracer.start_span(
            f"{attributes[GEN_AI_OPERATION_NAME]} {attributes['gen_ai.tool.name']}"
        )
        span.set_attributes(attributes)
        with trace.use_span(span, end_on_exit=True) as current_span:
            yield current_span

    class _AgentTelemetryLayer:
        id = "agent-1"
        name = "legacy-agent"
        description = None
        otel_provider_name = "microsoft.agent_framework"

        def _trace_agent_invocation(self, **kwargs):
            return kwargs["execute"]()

    def _get_span_attributes(**kwargs):
        attrs = {
            GEN_AI_OPERATION_NAME: kwargs["operation_name"],
            GEN_AI_PROVIDER_NAME: kwargs["provider_name"],
        }
        if kwargs.get("agent_name"):
            attrs["gen_ai.agent.name"] = kwargs["agent_name"]
        if kwargs.get("agent_id"):
            attrs["gen_ai.agent.id"] = kwargs["agent_id"]
        return attrs

    def _capture_response(span, attributes, **_kwargs):
        span.set_attributes(attributes)

    def _capture_messages(span, messages, output=False, **_kwargs):
        attr = "gen_ai.output.messages" if output else "gen_ai.input.messages"
        role = "assistant" if output else "user"
        span.set_attribute(
            attr,
            json.dumps(
                [
                    {
                        "role": role,
                        "parts": [{"type": "text", "content": str(message)}],
                    }
                    for message in messages
                ]
            ),
        )

    obs_mod = types.ModuleType("agent_framework.observability")
    obs_mod.OtelAttr = _OtelAttr
    obs_mod.OBSERVABILITY_SETTINGS = _Settings
    obs_mod.INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS = __import__(
        "contextvars"
    ).ContextVar("fields")
    obs_mod.INNER_ACCUMULATED_USAGE = __import__("contextvars").ContextVar(
        "usage"
    )
    obs_mod.INNER_RESPONSE_ID_CAPTURED_FIELD = "response_id"
    obs_mod.INNER_USAGE_CAPTURED_FIELD = "usage"
    obs_mod.AgentTelemetryLayer = _AgentTelemetryLayer
    obs_mod._get_span = _get_span
    obs_mod.get_function_span = get_function_span
    obs_mod.get_tracer = lambda: tracer
    obs_mod._get_span_attributes = _get_span_attributes
    obs_mod._get_instructions_from_options = lambda _options: None
    obs_mod._capture_messages = _capture_messages
    obs_mod._get_response_attributes = lambda _attrs, _response, **_kwargs: {}
    obs_mod._apply_accumulated_usage = lambda _attrs, _fields: None
    obs_mod._capture_response = _capture_response
    obs_mod.capture_exception = lambda span, exception, timestamp: (
        span.record_exception(exception)
    )

    af_mod = types.ModuleType("agent_framework")
    af_mod.observability = obs_mod
    types_mod = types.ModuleType("agent_framework._types")
    types_mod.ResponseStream = _ResponseStream
    tools_mod = types.ModuleType("agent_framework._tools")
    tools_mod.get_function_span = get_function_span
    mcp_mod = types.ModuleType("agent_framework._mcp")

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.observability", obs_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._types", types_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._tools", tools_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._mcp", mcp_mod)

    original_anext = _ResponseStream.__anext__
    original_run_cleanup_hooks = _ResponseStream._run_cleanup_hooks

    util_genai_bridge.revert_util_genai_bridge()
    util_genai_bridge.apply_util_genai_bridge()
    try:
        assert hasattr(_ResponseStream, "with_pull_context_manager")
        patched_anext = _ResponseStream.__anext__
        util_genai_bridge.apply_util_genai_bridge()
        assert _ResponseStream.__anext__ is patched_anext

        async def _updates():
            with tracer.start_as_current_span("chat qwen-plus"):
                yield "delta"

        def _execute():
            return _ResponseStream(
                _updates(),
                finalizer=lambda _updates: types.SimpleNamespace(
                    messages=["done"], usage_details=None
                ),
            )

        stream = _AgentTelemetryLayer()._trace_agent_invocation(
            messages=["hello"],
            session=None,
            merged_options={},
            client_kwargs=None,
            stream=True,
            execute=_execute,
        )

        async def _consume():
            async for _ in stream:
                pass

        asyncio.run(_consume())
    finally:
        util_genai_bridge.revert_util_genai_bridge()

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    agent = by_name["invoke_agent legacy-agent"]
    chat = by_name["chat qwen-plus"]
    assert chat.parent.span_id == agent.context.span_id
    assert chat.context.trace_id == agent.context.trace_id
    input_messages = json.loads(agent.attributes["gen_ai.input.messages"])
    assert input_messages[0]["parts"][0]["content"] == "hello"
    output_messages = json.loads(agent.attributes["gen_ai.output.messages"])
    assert output_messages[0]["parts"][0]["content"] == "done"
    assert not hasattr(_ResponseStream, "with_pull_context_manager")
    assert _ResponseStream.__anext__ is original_anext
    assert _ResponseStream._run_cleanup_hooks is original_run_cleanup_hooks
