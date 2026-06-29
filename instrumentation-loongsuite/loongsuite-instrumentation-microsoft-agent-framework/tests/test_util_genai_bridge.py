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
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == GenAIOperation.MCP
    assert span.attributes.get("gen_ai.tool.name") == "city_score"


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
