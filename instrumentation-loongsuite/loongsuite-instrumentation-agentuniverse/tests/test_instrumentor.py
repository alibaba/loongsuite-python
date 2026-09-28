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

"""Tests for the LoongSuite agentUniverse instrumentation.

Everything here runs against a real ``agentUniverse`` install. The same fixtures
are driven twice -- once under the framework's own Agent/LLM/Tool instrumentors
and once under this package -- and the two runs are compared, so the ``au.*``
contract this package has to keep is checked against its source of truth rather
than against a copy of it.

Two differences from the framework's own instrumentation are deliberate and
tested as such: user content is dropped from every span while content capture is
off, and a streamed LLM response is counted onto its parent span once instead of
twice.
"""

import ast
import asyncio
import importlib.metadata as metadata
import json
import os
import queue
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
from agentuniverse.agent.memory.conversation_memory.conversation_memory_module import (  # noqa: E501
    ConversationMemoryModule as RealConversationMemoryModule,
)
from agentuniverse.base.annotation import trace as trace_module
from agentuniverse.base.annotation.trace import trace_llm
from agentuniverse.base.util.monitor.monitor import Monitor
from agentuniverse.llm.llm_output import LLMOutput, TokenUsage

from opentelemetry.instrumentation.agentuniverse import (
    AgentUniverseInstrumentor,
)
from opentelemetry.instrumentation.agentuniverse import _agent as agent_layer
from opentelemetry.instrumentation.agentuniverse import _tool as tool_layer

from . import baseline
from .conftest import (
    FailingAgent,
    FailingLLM,
    FailingTool,
    RichAgent,
    StreamingAgent,
    StreamingLLM,
    StreamingLLMAgent,
    StubAgent,
    StubLLM,
    StubTool,
    active_native_instrumentor,
    build_agent,
    enable_native_instrumentors,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PACKAGE_ROOT
    / "src"
    / "opentelemetry"
    / "instrumentation"
    / "agentuniverse"
)

CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
SECRET = "s3cr3t-token-do-not-leak"

AGENT_SPAN = "au.agent."
LLM_SPAN = "au.llm."
TOOL_SPAN = "au.tool."

AGENT_METRICS = (
    "agent_calls_total",
    "agent_errors_total",
    "agent_call_duration",
    "agent_first_token_duration",
    "agent_total_tokens",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "agent_cached_tokens",
    "agent_reasoning_tokens",
)
LLM_METRICS = (
    "llm_calls_total",
    "llm_errors_total",
    "llm_call_duration",
    "llm_first_token_duration",
    "llm_total_tokens",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_cached_tokens",
    "llm_reasoning_tokens",
)
TOOL_METRICS = (
    "tool_calls_total",
    "tool_errors_total",
    "tool_call_duration",
    "tool_total_tokens",
    "tool_prompt_tokens",
    "tool_completion_tokens",
    "tool_cached_tokens",
    "tool_reasoning_tokens",
)

AGENT_AU_KEYS = {
    "au.span.kind",
    "au.agent.name",
    "au.agent.input",
    "au.agent.output",
    "au.agent.duration",
    "au.agent.status",
    "au.agent.pair_id",
    "au.agent.streaming",
    "au.agent.first_token.duration",
    "au.trace.caller_name",
    "au.trace.caller_type",
    "au.agent.usage.total_tokens",
    "au.agent.usage.prompt_tokens",
    "au.agent.usage.completion_tokens",
    "au.agent.usage.detail_tokens",
}
LLM_AU_KEYS = {
    "au.span.kind",
    "au.llm.name",
    "au.llm.channel_name",
    "au.llm.input",
    "au.llm.output",
    "au.llm.llm_params",
    "au.llm.streaming",
    "au.llm.duration",
    "au.llm.status",
    "au.llm.first_token.duration",
    "au.trace.caller_name",
    "au.trace.caller_type",
    "au.llm.usage.total_tokens",
    "au.llm.usage.prompt_tokens",
    "au.llm.usage.completion_tokens",
    "au.llm.usage.detail_tokens",
}
TOOL_AU_KEYS = {
    "au.span.kind",
    "au.tool.name",
    "au.tool.input",
    "au.tool.output",
    "au.tool.duration",
    "au.tool.status",
    "au.tool.pair_id",
    "au.trace.caller_name",
    "au.trace.caller_type",
    "au.tool.usage.total_tokens",
    "au.tool.usage.prompt_tokens",
    "au.tool.usage.completion_tokens",
    "au.tool.usage.detail_tokens",
}

#: Wrapper extension points this package claims.
WRAPPER_GLOBALS = (
    "_agent_wrapper_sync",
    "_agent_wrapper_async",
    "_llm_wrapper_sync",
    "_llm_wrapper_async",
    "_tool_wrapper_sync",
    "_tool_wrapper_async",
)

#: Runtime source may not name, import or call these.
FORBIDDEN_RUNTIME_REFERENCES = (
    "AgentInstrumentor",
    "LLMInstrumentor",
    "ToolInstrumentor",
    "AgentSpanAttributesSetter",
    "LLMSpanAttributesSetter",
    "ToolSpanAttributesSetter",
    "AgentSpanManager",
    "LLMSpanManager",
    "ToolSpanManager",
    "agent_instrumentor",
    "llm_instrumentor",
    "tool_instrumentor",
)


@contextmanager
def content_capture(mode: Optional[str]) -> Iterator[None]:
    """Run a block with a given content-capturing mode."""
    previous = os.environ.get(CONTENT_ENV)
    if mode is None:
        os.environ.pop(CONTENT_ENV, None)
    else:
        os.environ[CONTENT_ENV] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CONTENT_ENV, None)
        else:
            os.environ[CONTENT_ENV] = previous


def capture_on() -> Any:
    return content_capture("SPAN_ONLY")


def capture_off() -> Any:
    return content_capture("NO_CONTENT")


def layer_records(harness: Any, prefix: str) -> list:
    return [
        record
        for record in harness.records()
        if record.name.startswith(prefix)
    ]


def one_layer_record(harness: Any, prefix: str) -> Any:
    records = layer_records(harness, prefix)
    assert len(records) == 1, (
        f"expected one {prefix}* span, got {baseline.names(harness.records())}"
    )
    return records[0]


def metric_contract(metrics: dict) -> dict:
    """Metric names with the label sets recorded for them."""
    return {
        name: sorted({tuple(sorted(point.labels.items())) for point in points})
        for name, points in metrics.items()
    }


def recorded_on_success(names: Any) -> list:
    """The metrics a successful call is expected to have recorded.

    A counter with no recorded value has no data point, so the framework's
    own instrumentation exports exactly the same set as this package:
    everything but the error counters, which have their own error-path
    tests.
    """
    return [name for name in names if not name.endswith("_errors_total")]


def run_agent(agent: Any) -> Any:
    return agent.run(input="hello")


class SecretLLM(StubLLM):
    """An LLM whose output carries a secret."""

    name = "secret_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any) -> LLMOutput:
        return LLMOutput(
            text=f"echo {SECRET}",
            usage=TokenUsage(text_in=3, text_out=5),
            finish_reason="stop",
        )


class SecretFailingLLM(StubLLM):
    """An LLM that fails with a secret in its message."""

    name = "secret_failing_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any) -> LLMOutput:
        raise RuntimeError(f"llm failed on {SECRET}")


class ExplodingStreamLLM(StubLLM):
    """A streamed LLM that fails halfway through."""

    name = "exploding_stream_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any):
        yield LLMOutput(text="part-1")
        raise RuntimeError("stream exploded")


class SecretFailingTool(StubTool):
    """A tool that fails with a secret in its message."""

    name: str = "secret_failing_tool"

    def execute(self, query: str) -> str:
        raise RuntimeError(f"tool failed on {SECRET}")


class SecretAgent(StubAgent):
    """An agent whose LLM output and tool output carry a secret."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        llm_output = SecretLLM().call(prompt=input_object.get_data("input"))
        tool_output = StubTool().run(query=input_object.get_data("input"))
        return {"output": f"{llm_output.text}|{tool_output}"}


class SecretFailingAgent(StubAgent):
    """An agent that fails with a secret in its message."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        raise RuntimeError(f"agent failed on {SECRET}")


class SlowAsyncAgent(StubAgent):
    """An agent that can be cancelled while it works."""

    async def async_execute(
        self, input_object: Any, agent_input: dict
    ) -> dict:
        await asyncio.sleep(5)
        return {"output": "late"}


class FailingStreamingAgent(StubAgent):
    """An agent whose streaming body raises after the first token."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        stream = input_object.get_data("output_stream")
        if stream is not None:
            stream.put("first-token")
        raise RuntimeError("streaming agent exploded")


def memory_spy(monkeypatch: Any, module: Any) -> list:
    """Record conversation-memory calls, still running the real module."""
    calls: list = []

    class Spy:
        def __init__(self) -> None:
            self._real = RealConversationMemoryModule()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    for name in (
        "add_agent_input_info",
        "add_agent_result_info",
        "add_tool_input_info",
        "add_tool_output_info",
    ):

        def record(*args: Any, __name: str = name, **kwargs: Any) -> Any:
            calls.append(__name)
            return getattr(RealConversationMemoryModule(), __name)(
                *args, **kwargs
            )

        setattr(Spy, name, record)
    monkeypatch.setattr(module, "ConversationMemoryModule", Spy)
    return calls


# ---------------------------------------------------------------------------
# Install, suppression and restore
# ---------------------------------------------------------------------------


class TestInstallAndRestore:
    def test_extension_points_start_at_the_framework_defaults(self) -> None:
        for name in WRAPPER_GLOBALS:
            wrapper = getattr(trace_module, name)
            assert callable(wrapper), name
            assert getattr(wrapper, "__self__", None) is None, name

    def test_install_claims_all_six_extension_points(self) -> None:
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument(tracer_provider=None, meter_provider=None)
        try:
            for name in WRAPPER_GLOBALS:
                owner = getattr(trace_module, name).__self__
                assert owner is not None, name
                assert owner.__class__.__module__.startswith(
                    "opentelemetry.instrumentation.agentuniverse"
                ), name
            assert instrumentor.installed_wrappers is not None
            assert set(instrumentor.installed_wrappers) == set(WRAPPER_GLOBALS)
        finally:
            instrumentor.uninstrument()

    def test_uninstrument_restores_object_identity(self) -> None:
        saved = {name: getattr(trace_module, name) for name in WRAPPER_GLOBALS}
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument()
        try:
            for name in WRAPPER_GLOBALS:
                assert getattr(trace_module, name) is not saved[name]
        finally:
            instrumentor.uninstrument()
        for name, original in saved.items():
            assert getattr(trace_module, name) is original, name

    def test_double_instrument_does_not_double_wrap(
        self, loongsuite_harness: Any
    ) -> None:
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument()
        instrumentor.instrument()
        try:
            rich = build_agent("rich_agent", RichAgent)
            run_agent(rich)
            assert len(layer_records(loongsuite_harness, AGENT_SPAN)) == 1
            assert len(layer_records(loongsuite_harness, LLM_SPAN)) == 1
            assert len(layer_records(loongsuite_harness, TOOL_SPAN)) == 1
        finally:
            instrumentor.uninstrument()

    def test_uninstrument_twice_is_safe(self) -> None:
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument()
        instrumentor.uninstrument()
        instrumentor.uninstrument()

    def test_failed_install_rolls_back_every_extension_point(
        self, monkeypatch: Any
    ) -> None:
        saved = {name: getattr(trace_module, name) for name in WRAPPER_GLOBALS}
        instrumentor = AgentUniverseInstrumentor()
        calls = {"count": 0}
        original = instrumentor._install_wrapper

        def flaky(name: str, wrapper: Any) -> None:
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("boom")
            original(name, wrapper)

        monkeypatch.setattr(instrumentor, "_install_wrapper", flaky)
        instrumentor.instrument()
        for name, value in saved.items():
            assert getattr(trace_module, name) is value, name
        assert not instrumentor.__dict__.get(
            "_is_instrumented_by_opentelemetry"
        )

    def test_native_then_loongsuite_restores_native_wrappers(
        self, configured_providers: Any
    ) -> None:
        provider, meter_provider, _reader = configured_providers
        enabled = enable_native_instrumentors(provider, meter_provider)
        native = {
            name: getattr(trace_module, name) for name in WRAPPER_GLOBALS
        }
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument()
        assert (
            getattr(trace_module, "_agent_wrapper_sync")
            is not native["_agent_wrapper_sync"]
        )
        instrumentor.uninstrument()
        for name, wrapper in native.items():
            assert getattr(trace_module, name) is wrapper, name
        for instrumentor_ in enabled:
            instrumentor_.uninstrument()

    def test_restored_native_wrappers_still_work(
        self, configured_providers: Any
    ) -> None:
        provider, meter_provider, _reader = configured_providers
        enabled = enable_native_instrumentors(provider, meter_provider)
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument()
        instrumentor.uninstrument()
        run_agent(build_agent("native_after_restore"))
        instrumentor.instrument()
        run_agent(build_agent("loongsuite_after_restore"))
        for instrumentor_ in enabled:
            instrumentor_.uninstrument()

    def test_late_native_override_is_not_clobbered(
        self, configured_providers: Any
    ) -> None:
        from agentuniverse.base.tracing.otel.instrumentation.agent.agent_instrumentor import (  # noqa: E501
            AgentInstrumentor,
        )

        provider, meter_provider, _reader = configured_providers
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument()
        native_agent = active_native_instrumentor(AgentInstrumentor)
        assert native_agent is None, (
            "the native agent instrumentor was already on"
        )
        native_agent = AgentInstrumentor()
        native_agent.instrument(
            tracer_provider=provider, meter_provider=meter_provider
        )
        late = {
            name: getattr(trace_module, name)
            for name in ("_agent_wrapper_sync", "_agent_wrapper_async")
        }
        assert getattr(late["_agent_wrapper_sync"], "__self__") is native_agent
        instrumentor.uninstrument()
        for name, wrapper in late.items():
            assert getattr(trace_module, name) is wrapper, (
                f"{name} was restored by LoongSuite even though another "
                "instrumentation had replaced it"
            )
        native_agent.uninstrument()

    def test_instrumentor_exposes_its_layer_globals(self) -> None:
        instrumentor = AgentUniverseInstrumentor()
        assert {tuple(pair) for pair in instrumentor.layer_globals} == {
            ("_agent_wrapper_sync", "_agent_wrapper_async"),
            ("_llm_wrapper_sync", "_llm_wrapper_async"),
            ("_tool_wrapper_sync", "_tool_wrapper_async"),
        }


# ---------------------------------------------------------------------------
# Agent layer
# ---------------------------------------------------------------------------


class TestAgentLayer:
    def test_sync_run_creates_one_span(self, loongsuite_harness: Any) -> None:
        run_agent(build_agent("test_agent"))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.name == "au.agent.test_agent"
        assert record.kind == "INTERNAL"
        assert record.status == "UNSET"
        assert record.parent is None

    def test_span_carries_the_full_native_attribute_set(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            run_agent(build_agent("test_agent"))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert set(record.au) == AGENT_AU_KEYS
        assert record.au["au.span.kind"] == "agent"
        assert record.au["au.agent.name"] == "test_agent"
        assert record.au["au.agent.status"] == "success"
        assert record.au["au.agent.streaming"] is False
        assert record.au["au.agent.duration"] >= 0
        assert record.au["au.trace.caller_type"] == "user"
        assert json.loads(record.au["au.agent.input"]) == {
            "kwargs": {"input": "hello"}
        }
        assert "echo:hello" in record.au["au.agent.output"]
        assert (
            json.loads(record.au["au.agent.output"])["output"] == "echo:hello"
        )

    def test_span_carries_gen_ai_attributes(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            run_agent(build_agent("test_agent"))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert {
            "gen_ai.operation.name",
            "gen_ai.span.kind",
            "gen_ai.agent.name",
            "gen_ai.framework",
        } <= set(record.gen_ai)
        assert record.gen_ai["gen_ai.operation.name"] == "invoke_agent"
        assert record.gen_ai["gen_ai.span.kind"] == "AGENT"
        assert record.gen_ai["gen_ai.agent.name"] == "test_agent"
        assert record.gen_ai["gen_ai.framework"] == "agentuniverse"

    def test_async_run_matches_sync(self, loongsuite_harness: Any) -> None:
        agent = build_agent("async_agent")
        asyncio.run(agent.async_run(input="hello"))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.name == "au.agent.async_agent"
        assert record.au["au.agent.status"] == "success"

    def test_all_nine_agent_metrics_are_recorded(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_agent("test_agent"))
        metrics = loongsuite_harness.metrics()
        for name in recorded_on_success(AGENT_METRICS):
            assert baseline.points_for(
                metrics, name, au_agent_name="test_agent"
            ), name
        calls = baseline.points_for(
            metrics, "agent_calls_total", au_agent_name="test_agent"
        )
        assert len(calls) == 1 and calls[0].value == 1
        assert set(calls[0].labels) == {
            "au_agent_name",
            "au_trace_caller_name",
            "au_trace_caller_type",
            "au_agent_status",
        }
        for name in AGENT_METRICS[2:]:
            point = baseline.points_for(
                metrics, name, au_agent_name="test_agent"
            )[0]
            assert "au_agent_streaming" in point.labels, name

    def test_streaming_agent_first_token_is_positive(
        self, loongsuite_harness: Any
    ) -> None:
        agent = build_agent("stream_agent", StreamingAgent)
        run_agent_with_stream(agent)
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.au["au.agent.streaming"] is True
        assert record.au["au.agent.first_token.duration"] > 0
        assert record.gen_ai["gen_ai.response.time_to_first_token"] > 0
        metrics = loongsuite_harness.metrics()
        point = baseline.points_for(
            metrics,
            "agent_first_token_duration",
            au_agent_name="stream_agent",
        )[0]
        assert point.value > 0
        assert point.labels["au_agent_streaming"] is True

    def test_error_run_records_status_attribute_and_metric(
        self, loongsuite_harness: Any
    ) -> None:
        with pytest.raises(RuntimeError):
            run_agent(build_agent("boom_agent", FailingAgent))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.status == "ERROR"
        assert record.au["au.agent.status"] == "error"
        assert record.au["au.agent.error.type"] == "RuntimeError"
        # Content capture is off here, so the message is the type alone;
        # the captured form is asserted by the privacy tests.
        assert record.au["au.agent.error.message"] == "RuntimeError"
        metrics = loongsuite_harness.metrics()
        assert (
            baseline.counter_value(
                metrics,
                "agent_errors_total",
                au_agent_name="boom_agent",
                au_agent_status="RuntimeError",
            )
            == 1
        )

    def test_conversation_memory_receives_input_and_result(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        calls = memory_spy(monkeypatch, agent_layer)
        run_agent(build_agent("memory_agent"))
        assert "add_agent_input_info" in calls
        assert "add_agent_result_info" in calls


def run_agent_with_stream(agent: Any) -> Any:
    stream: queue.Queue = queue.Queue()
    return agent.run(input="hello", output_stream=stream)


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------


class TestLLMLayer:
    def test_sync_call_creates_one_span(self, loongsuite_harness: Any) -> None:
        with capture_on():
            StubLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.name == "au.llm.stub_llm"
        assert record.kind == "INTERNAL"
        assert set(record.au) == LLM_AU_KEYS
        assert record.au["au.span.kind"] == "llm"
        assert record.au["au.llm.channel_name"] == "test_channel"
        assert record.au["au.llm.streaming"] is False
        assert record.au["au.llm.output"] == "llm:hello"
        # The framework's own helper reports its -1 sentinel for a
        # configured temperature; the baseline tests pin that this package
        # reports the same payload.
        assert set(json.loads(record.au["au.llm.llm_params"])) == {
            "temperature"
        }

    def test_span_carries_gen_ai_attributes(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            StubLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert {
            "gen_ai.operation.name",
            "gen_ai.span.kind",
            "gen_ai.request.model",
            "gen_ai.provider.name",
            "gen_ai.framework",
        } <= set(record.gen_ai)
        assert record.gen_ai["gen_ai.operation.name"] == "chat"
        assert record.gen_ai["gen_ai.span.kind"] == "LLM"

    def test_all_nine_llm_metrics_are_recorded(
        self, loongsuite_harness: Any
    ) -> None:
        StubLLM().call(prompt="hello")
        metrics = loongsuite_harness.metrics()
        for name in recorded_on_success(LLM_METRICS):
            assert baseline.points_for(
                metrics, name, au_llm_name="stub_llm"
            ), name
        calls = baseline.points_for(
            metrics, "llm_calls_total", au_llm_name="stub_llm"
        )
        assert len(calls) == 1 and calls[0].value == 1
        assert set(calls[0].labels) == {
            "au_llm_name",
            "au_trace_caller_name",
            "au_trace_caller_type",
            "au_llm_status",
        }
        first_token = baseline.points_for(
            metrics, "llm_first_token_duration", au_llm_name="stub_llm"
        )[0]
        assert first_token.labels["au_llm_streaming"] is False

    def test_async_call_creates_one_span(
        self, loongsuite_harness: Any
    ) -> None:
        from .conftest import AsyncLLM

        with capture_on():
            asyncio.run(AsyncLLM().call(prompt="hello"))
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.name == "au.llm.async_llm"
        assert record.au["au.llm.output"] == "async:hello"

    def test_stream_span_is_held_until_the_stream_is_consumed(
        self, loongsuite_harness: Any
    ) -> None:
        stream = StreamingLLM().call(prompt="hello")
        assert loongsuite_harness.spans() == []
        chunks = list(stream)
        assert len(chunks) == 2
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.streaming"] is True
        assert record.au["au.llm.status"] == "success"
        assert record.au["au.llm.first_token.duration"] >= 0
        assert record.gen_ai["gen_ai.response.time_to_first_token"] >= 0

    def test_stream_usage_is_counted_once(
        self, loongsuite_harness: Any
    ) -> None:
        list(StreamingLLM().call(prompt="hello"))
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.usage.total_tokens"] == 10
        metrics = loongsuite_harness.metrics()
        assert baseline.histogram_values(
            metrics, "llm_total_tokens", au_llm_name="stream_llm"
        ) == [10]

    def test_stream_closed_early_finalizes_once(
        self, loongsuite_harness: Any
    ) -> None:
        stream = StreamingLLM().call(prompt="hello")
        next(stream)
        stream.close()
        records = layer_records(loongsuite_harness, LLM_SPAN)
        assert len(records) == 1
        assert records[0].au["au.llm.status"] == "success"

    def test_stream_error_finalizes_once_with_error_status(
        self, loongsuite_harness: Any
    ) -> None:
        with pytest.raises(RuntimeError):
            list(ExplodingStreamLLM().call(prompt="hello"))
        records = layer_records(loongsuite_harness, LLM_SPAN)
        assert len(records) == 1
        assert records[0].status == "ERROR"
        assert records[0].au["au.llm.error.type"] == "RuntimeError"
        metrics = loongsuite_harness.metrics()
        assert (
            baseline.counter_value(
                metrics,
                "llm_errors_total",
                au_llm_name="exploding_stream_llm",
            )
            == 1
        )

    def test_error_call_records_status_and_metric(
        self, loongsuite_harness: Any
    ) -> None:
        with pytest.raises(RuntimeError):
            FailingLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.status == "ERROR"
        assert record.au["au.llm.error.type"] == "RuntimeError"
        assert "au.llm.input" not in record.au
        assert "au.llm.output" not in record.au
        metrics = loongsuite_harness.metrics()
        assert (
            baseline.counter_value(
                metrics, "llm_errors_total", au_llm_name="failing_llm"
            )
            == 1
        )

    def test_llm_plugin_hook_is_applied(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        calls = {"count": 0}
        original = trace_module._llm_plugins

        def spy(func: Any) -> Any:
            calls["count"] += 1
            return original(func)

        monkeypatch.setattr(trace_module, "_llm_plugins", spy)
        StubLLM().call(prompt="hello")
        assert calls["count"] == 1
        assert (
            one_layer_record(loongsuite_harness, LLM_SPAN).au["au.llm.status"]
            == "success"
        )


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------


class TestToolLayer:
    def test_sync_run_creates_one_span(self, loongsuite_harness: Any) -> None:
        with capture_on():
            StubTool().run(query="hello")
        record = one_layer_record(loongsuite_harness, TOOL_SPAN)
        assert record.name == "au.tool.stub_tool"
        assert record.kind == "INTERNAL"
        assert set(record.au) == TOOL_AU_KEYS
        assert record.au["au.span.kind"] == "tool"
        assert (
            json.loads(record.au["au.tool.input"])["kwargs"]["query"]
            == "hello"
        )
        assert json.loads(record.au["au.tool.output"]) == "tool-output:hello"
        assert record.au["au.tool.status"] == "success"
        assert record.au["au.tool.pair_id"].startswith("tool_")

    def test_span_carries_gen_ai_attributes(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            StubTool().run(query="hello")
        record = one_layer_record(loongsuite_harness, TOOL_SPAN)
        assert {
            "gen_ai.operation.name",
            "gen_ai.span.kind",
            "gen_ai.tool.name",
            "gen_ai.framework",
        } <= set(record.gen_ai)
        assert record.gen_ai["gen_ai.operation.name"] == "execute_tool"
        assert record.gen_ai["gen_ai.span.kind"] == "TOOL"
        assert record.gen_ai["gen_ai.tool.name"] == "stub_tool"

    def test_all_eight_tool_metrics_are_recorded(
        self, loongsuite_harness: Any
    ) -> None:
        StubTool().run(query="hello")
        metrics = loongsuite_harness.metrics()
        for name in recorded_on_success(TOOL_METRICS):
            assert baseline.points_for(
                metrics, name, au_tool_name="stub_tool"
            ), name
        calls = baseline.points_for(
            metrics, "tool_calls_total", au_tool_name="stub_tool"
        )
        assert len(calls) == 1 and calls[0].value == 1
        assert set(calls[0].labels) == {
            "au_tool_name",
            "au_trace_caller_name",
            "au_trace_caller_type",
            "au_tool_status",
        }
        assert "au_tool_streaming" not in calls[0].labels

    def test_async_run_creates_one_span(self, loongsuite_harness: Any) -> None:
        with capture_on():
            asyncio.run(StubTool().async_run(query="hello"))
        record = one_layer_record(loongsuite_harness, TOOL_SPAN)
        assert record.au["au.tool.output"] == '"tool-output:hello"'

    def test_error_run_records_attributes_and_metric(
        self, loongsuite_harness: Any
    ) -> None:
        with pytest.raises(RuntimeError):
            FailingTool().run(query="hello")
        record = one_layer_record(loongsuite_harness, TOOL_SPAN)
        assert record.au["au.tool.error.type"] == "RuntimeError"
        # Capture is off, so no content from the failure reaches the span.
        assert record.au["au.tool.error.message"] == "RuntimeError"
        assert "au.tool.input" not in record.au
        assert "au.tool.output" not in record.au
        metrics = loongsuite_harness.metrics()
        assert (
            baseline.counter_value(
                metrics,
                "tool_errors_total",
                au_tool_name="failing_tool",
                au_tool_status="RuntimeError",
            )
            == 1
        )

    def test_conversation_memory_receives_input_and_output(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        calls = memory_spy(monkeypatch, tool_layer)
        StubTool().run(query="hello")
        assert "add_tool_input_info" in calls
        assert "add_tool_output_info" in calls


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_llm_usage_attributes_and_metrics(
        self, loongsuite_harness: Any
    ) -> None:
        StubLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.usage.total_tokens"] == 8
        assert record.au["au.llm.usage.prompt_tokens"] == 3
        assert record.au["au.llm.usage.completion_tokens"] == 5
        metrics = loongsuite_harness.metrics()
        assert baseline.histogram_values(
            metrics, "llm_total_tokens", au_llm_name="stub_llm"
        ) == [8]
        assert baseline.histogram_values(
            metrics, "llm_prompt_tokens", au_llm_name="stub_llm"
        ) == [3]
        assert baseline.histogram_values(
            metrics, "llm_completion_tokens", au_llm_name="stub_llm"
        ) == [5]

    def test_agent_aggregates_child_usage(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_agent("rich_agent", RichAgent))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.au["au.agent.usage.total_tokens"] == 8
        assert record.au["au.agent.usage.prompt_tokens"] == 3
        assert record.au["au.agent.usage.completion_tokens"] == 5
        metrics = loongsuite_harness.metrics()
        assert baseline.histogram_values(
            metrics, "agent_total_tokens", au_agent_name="rich_agent"
        ) == [8]
        assert baseline.histogram_values(
            metrics, "agent_prompt_tokens", au_agent_name="rich_agent"
        ) == [3]
        assert baseline.histogram_values(
            metrics, "agent_completion_tokens", au_agent_name="rich_agent"
        ) == [5]

    def test_tool_usage_stays_at_zero(self, loongsuite_harness: Any) -> None:
        StubTool().run(query="hello")
        record = one_layer_record(loongsuite_harness, TOOL_SPAN)
        assert record.au["au.tool.usage.total_tokens"] == 0

    def test_streamed_agent_usage_counts_once(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_agent("stream_llm_agent", StreamingLLMAgent))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.au["au.agent.usage.total_tokens"] == 10
        metrics = loongsuite_harness.metrics()
        assert baseline.histogram_values(
            metrics, "agent_total_tokens", au_agent_name="stream_llm_agent"
        ) == [10]

    def test_gen_ai_usage_attributes_are_set(
        self, loongsuite_harness: Any
    ) -> None:
        StubLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.gen_ai.get("gen_ai.usage.input_tokens") == 3
        assert record.gen_ai.get("gen_ai.usage.output_tokens") == 5

    def test_each_layer_records_one_call(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_agent("rich_agent", RichAgent))
        metrics = loongsuite_harness.metrics()
        assert (
            baseline.counter_value(
                metrics, "agent_calls_total", au_agent_name="rich_agent"
            )
            == 1
        )
        assert (
            baseline.counter_value(
                metrics, "llm_calls_total", au_llm_name="stub_llm"
            )
            == 1
        )
        assert (
            baseline.counter_value(
                metrics, "tool_calls_total", au_tool_name="stub_tool"
            )
            == 1
        )


# ---------------------------------------------------------------------------
# Content privacy
# ---------------------------------------------------------------------------


class TestContentPrivacy:
    def test_capture_off_drops_content_on_every_layer(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_off():
            run_agent(build_agent("rich_agent", RichAgent))
        records = loongsuite_harness.records()
        assert len(records) == 3
        for record in records:
            for key in (
                "au.agent.input",
                "au.agent.output",
                "au.llm.input",
                "au.llm.output",
                "au.tool.input",
                "au.tool.output",
                "gen_ai.input.messages",
                "gen_ai.output.messages",
                "gen_ai.tool.call.arguments",
                "gen_ai.tool.call.result",
            ):
                assert key not in record.au and key not in record.gen_ai, (
                    f"{record.name} carried {key} with capture off"
                )

    def test_capture_off_keeps_structure_and_usage(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_agent("rich_agent", RichAgent))
        agent = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert agent.au["au.agent.name"] == "rich_agent"
        assert agent.au["au.agent.status"] == "success"
        assert agent.au["au.span.kind"] == "agent"
        assert agent.au["au.agent.usage.total_tokens"] == 8
        assert agent.gen_ai["gen_ai.operation.name"] == "invoke_agent"

    def test_capture_on_writes_both_carriers(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            run_agent(build_agent("rich_agent", RichAgent))
        records = loongsuite_harness.records()
        agent = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert "au.agent.input" in agent.au and "au.agent.output" in agent.au
        assert "gen_ai.input.messages" in agent.gen_ai
        assert "gen_ai.output.messages" in agent.gen_ai
        messages = json.loads(agent.gen_ai["gen_ai.input.messages"])
        assert messages[0]["role"] == "user"
        assert "hello" in messages[0]["parts"][0]["content"]
        assert json.loads(agent.gen_ai["gen_ai.output.messages"])
        assert len(records) == 3

    def test_capture_on_writes_the_tool_call_content(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            StubTool().run(query="hello")
        record = one_layer_record(loongsuite_harness, TOOL_SPAN)
        assert record.au["au.tool.input"]
        assert record.au["au.tool.output"] == '"tool-output:hello"'
        assert json.loads(record.gen_ai["gen_ai.tool.call.arguments"]) == {
            "kwargs": {"query": "hello"}
        }
        assert record.gen_ai["gen_ai.tool.call.result"] == "tool-output:hello"

    def test_capture_mode_is_read_per_call(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_off():
            run_agent(build_agent("first_agent"))
        with capture_on():
            run_agent(build_agent("second_agent"))
        off = one_layer_record(loongsuite_harness, "au.agent.first_agent")
        on = one_layer_record(loongsuite_harness, "au.agent.second_agent")
        assert "au.agent.input" not in off.au
        assert "au.agent.input" in on.au

    def test_unset_mode_is_no_content(self, loongsuite_harness: Any) -> None:
        with content_capture(None):
            run_agent(build_agent("default_agent"))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert "au.agent.input" not in record.au
        assert "gen_ai.input.messages" not in record.gen_ai

    def test_secrets_do_not_reach_any_span_attribute_with_capture_off(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_off():
            run_agent(build_agent("secret_agent", SecretAgent))
            with pytest.raises(RuntimeError):
                run_agent(
                    build_agent("secret_failure_agent", SecretFailingAgent)
                )
            with pytest.raises(RuntimeError):
                SecretFailingLLM().call(prompt="hello")
            with pytest.raises(RuntimeError):
                SecretFailingTool().run(query="hello")
        records = loongsuite_harness.records()
        assert len(records) >= 4
        for record in records:
            for key, value in record.attributes.items():
                assert SECRET not in str(key)
                assert SECRET not in str(value), (
                    f"{record.name} leaked a secret through {key}"
                )
            assert SECRET not in record.description, (
                f"{record.name} leaked a secret through its status description"
            )

    def test_secrets_do_reach_content_attributes_with_capture_on(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            run_agent(build_agent("secret_agent", SecretAgent))
        records = loongsuite_harness.records()
        values = [
            str(value)
            for record in records
            for value in record.attributes.values()
        ]
        assert any(SECRET in value for value in values)

    def test_llm_params_are_filtered_when_capture_is_off(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_off():
            StubLLM().call(prompt="hello", api_key=SECRET)
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert SECRET not in record.au["au.llm.llm_params"]
        assert set(json.loads(record.au["au.llm.llm_params"])) == {
            "temperature"
        }

    def test_error_message_is_not_a_traceback_when_capture_is_off(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_off():
            with pytest.raises(RuntimeError):
                run_agent(build_agent("boom_agent", FailingAgent))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.au["au.agent.error.message"] == "RuntimeError"
        assert record.description == "RuntimeError"

    def test_error_message_is_a_traceback_when_capture_is_on(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            with pytest.raises(RuntimeError):
                run_agent(build_agent("boom_agent", FailingAgent))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert (
            "Traceback (most recent call last)"
            in record.au["au.agent.error.message"]
        )


# ---------------------------------------------------------------------------
# Baseline: this package against the framework's own instrumentation
# ---------------------------------------------------------------------------


class TestBaselineMatrix:
    def test_span_tree_and_au_attributes_match_native(
        self, compare_runs: Any
    ) -> None:
        with capture_on():
            result = compare_runs(
                lambda: run_agent(build_agent("rich_agent", RichAgent))
            )
        assert sorted(baseline.names(result.native_records)) == sorted(
            baseline.names(result.ours_records)
        )
        assert sorted(baseline.names(result.ours_records)) == [
            "au.agent.rich_agent",
            "au.llm.stub_llm",
            "au.tool.stub_tool",
        ]
        for name in baseline.names(result.ours_records):
            baseline.assert_same_span_shape(
                baseline.find(result.native_records, name),
                baseline.find(result.ours_records, name),
            )

    def test_native_run_has_no_gen_ai_attributes_and_ours_adds_them(
        self, compare_runs: Any
    ) -> None:
        result = compare_runs(
            lambda: run_agent(build_agent("rich_agent", RichAgent))
        )
        assert result.native_records
        for record in result.native_records:
            assert record.gen_ai == {}
        for record in result.ours_records:
            assert record.gen_ai["gen_ai.framework"] == "agentuniverse"
            assert record.gen_ai["gen_ai.span.kind"] in {
                "AGENT",
                "LLM",
                "TOOL",
            }

    def test_metrics_match_native(self, compare_runs: Any) -> None:
        result = compare_runs(
            lambda: run_agent(build_agent("rich_agent", RichAgent))
        )
        native = metric_contract(result.native_metrics)
        ours = metric_contract(result.ours_metrics)
        expected = recorded_on_success(
            AGENT_METRICS + LLM_METRICS + TOOL_METRICS
        )
        for name in expected:
            assert name in native, f"native did not record {name}"
            assert name in ours, f"LoongSuite did not record {name}"
            assert native[name] == ours[name], (
                f"{name} labels differ: {native[name]} vs {ours[name]}"
            )

    def test_metric_family_sets_are_exactly_nine_nine_eight(
        self, compare_runs: Any
    ) -> None:
        result = compare_runs(
            lambda: run_agent(build_agent("rich_agent", RichAgent))
        )
        layers = (AGENT_METRICS, LLM_METRICS, TOOL_METRICS)
        assert [len(names) for names in layers] == [9, 9, 8]
        declared = set(AGENT_METRICS) | set(LLM_METRICS) | set(TOOL_METRICS)
        assert len(declared) == 26, (
            "the three prefixes keep the families distinct"
        )
        for name in (
            "agent_errors_total",
            "llm_errors_total",
            "tool_errors_total",
        ):
            assert name in declared

        # A counter with no recorded value exports no family, so a successful
        # call shows every family but the three error counters.
        au_families = set(recorded_on_success(sorted(declared)))
        assert len(au_families) == 23
        # The shared GenAI handler records the client-side conventions on the
        # spans this package creates; the framework's own instrumentation has
        # no equivalent.
        gen_ai_client = {
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        }
        for label, metrics, extra in (
            ("native", result.native_metrics, set()),
            ("LoongSuite", result.ours_metrics, gen_ai_client),
        ):
            exported = set(baseline.metric_names(metrics))
            expected = au_families | extra
            assert exported == expected, (
                f"{label} families differ: extra {sorted(exported - expected)}, "
                f"missing {sorted(expected - exported)}"
            )

    def test_token_values_match_native(self, compare_runs: Any) -> None:
        result = compare_runs(
            lambda: run_agent(build_agent("rich_agent", RichAgent))
        )
        for records in (result.native_records, result.ours_records):
            agent = baseline.find(records, "au.agent.rich_agent")
            assert agent.au["au.agent.usage.total_tokens"] == 8
            assert agent.au["au.agent.usage.prompt_tokens"] == 3
            assert agent.au["au.agent.usage.completion_tokens"] == 5
        for metrics in (result.native_metrics, result.ours_metrics):
            assert baseline.histogram_values(
                metrics, "agent_total_tokens", au_agent_name="rich_agent"
            ) == [8]

    def test_error_path_matches_native(self, compare_runs: Any) -> None:
        def workload() -> None:
            with capture_on():
                with pytest.raises(RuntimeError):
                    run_agent(build_agent("boom_agent", FailingAgent))

        result = compare_runs(workload)
        native = baseline.find(result.native_records, "au.agent.boom_agent")
        ours = baseline.find(result.ours_records, "au.agent.boom_agent")
        baseline.assert_same_span_shape(
            native, ours, ignore_au=("au.agent.error.message",)
        )
        assert (
            native.au["au.agent.error.message"]
            .strip()
            .endswith("RuntimeError: agent exploded")
        )
        assert (
            ours.au["au.agent.error.message"]
            .strip()
            .endswith("RuntimeError: agent exploded")
        )
        for metrics in (result.native_metrics, result.ours_metrics):
            point = baseline.points_for(
                metrics,
                "agent_errors_total",
                au_agent_name="boom_agent",
                au_agent_status="RuntimeError",
            )
            assert len(point) == 1 and point[0].value == 1

    def test_tool_error_path_matches_native(self, compare_runs: Any) -> None:
        def workload() -> None:
            with capture_on():
                with pytest.raises(RuntimeError):
                    FailingTool().run(query="hello")

        result = compare_runs(workload)
        native = baseline.find(result.native_records, "au.tool.failing_tool")
        ours = baseline.find(result.ours_records, "au.tool.failing_tool")
        baseline.assert_same_span_shape(
            native, ours, ignore_au=("au.tool.error.message",)
        )
        assert native.status == ours.status == "ERROR"
        for metrics in (result.native_metrics, result.ours_metrics):
            point = baseline.points_for(
                metrics,
                "tool_errors_total",
                au_tool_name="failing_tool",
                au_tool_status="RuntimeError",
            )
            assert len(point) == 1 and point[0].value == 1

    def test_streaming_first_token_is_positive_in_both(
        self, compare_runs: Any
    ) -> None:
        result = compare_runs(
            lambda: list(StreamingLLM().call(prompt="hello"))
        )
        for records in (result.native_records, result.ours_records):
            record = baseline.find(records, "au.llm.stream_llm")
            assert record.au["au.llm.first_token.duration"] > 0
            assert record.au["au.llm.streaming"] is True

    def test_coexistence_leaves_one_span_per_layer(
        self, both_harness: Any
    ) -> None:
        run_agent(build_agent("rich_agent", RichAgent))
        records = both_harness.records()
        assert len(records) == 3, baseline.names(records)
        assert {record.name for record in records} == {
            "au.agent.rich_agent",
            "au.llm.stub_llm",
            "au.tool.stub_tool",
        }
        for record in records:
            assert record.gen_ai["gen_ai.framework"] == "agentuniverse"
        metrics = both_harness.metrics()
        assert (
            baseline.counter_value(
                metrics, "agent_calls_total", au_agent_name="rich_agent"
            )
            == 1
        )
        assert (
            baseline.counter_value(
                metrics, "llm_calls_total", au_llm_name="stub_llm"
            )
            == 1
        )
        assert (
            baseline.counter_value(
                metrics, "tool_calls_total", au_tool_name="stub_tool"
            )
            == 1
        )

    def test_intentional_difference_privacy_off(
        self, compare_runs: Any
    ) -> None:
        def workload() -> None:
            with capture_off():
                run_agent(build_agent("privacy_agent"))

        result = compare_runs(workload)
        native = baseline.find(result.native_records, "au.agent.privacy_agent")
        ours = baseline.find(result.ours_records, "au.agent.privacy_agent")
        assert "au.agent.input" in native.au
        assert "au.agent.input" not in ours.au

    def test_intentional_difference_streaming_token_count(
        self, compare_runs: Any
    ) -> None:
        result = compare_runs(
            lambda: run_agent(build_agent("stream_agent", StreamingLLMAgent))
        )
        native = baseline.find(result.native_records, "au.agent.stream_agent")
        ours = baseline.find(result.ours_records, "au.agent.stream_agent")
        assert ours.au["au.agent.usage.total_tokens"] == 10
        assert native.au["au.agent.usage.total_tokens"] == 20, (
            "the framework's own instrumentation is expected to count a "
            "streamed response onto its parent twice"
        )


# ---------------------------------------------------------------------------
# Fail-safe behaviour
# ---------------------------------------------------------------------------


class TestFailSafe:
    def test_agent_info_failure_still_runs_the_call(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("no info for you")

        monkeypatch.setattr(agent_layer, "_get_agent_info", boom)
        result = run_agent(build_agent("failsafe_agent"))
        assert result.get_data("output") == "echo:hello"
        assert loongsuite_harness.spans() == []

    def test_llm_info_failure_still_runs_the_call(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        from opentelemetry.instrumentation.agentuniverse import (
            _llm as llm_layer,
        )

        monkeypatch.setattr(
            llm_layer,
            "_get_llm_info",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
        )
        output = StubLLM().call(prompt="hello")
        assert output.text == "llm:hello"
        assert loongsuite_harness.spans() == []

    def test_tool_info_failure_still_runs_the_call(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            tool_layer,
            "_get_tool_info",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
        )
        assert StubTool().run(query="hello") == "tool-output:hello"
        assert loongsuite_harness.spans() == []

    def test_invocation_chain_failure_still_runs_the_call(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("no chain")

        monkeypatch.setattr(Monitor, "init_invocation_chain", boom)
        result = run_agent(build_agent("chainless_agent"))
        assert result.get_data("output") == "echo:hello"
        assert (
            one_layer_record(loongsuite_harness, AGENT_SPAN).au[
                "au.agent.status"
            ]
            == "success"
        )

    def test_memory_failure_still_runs_the_call(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        class Broken:
            def __init__(self) -> None:
                raise RuntimeError("no memory")

        monkeypatch.setattr(agent_layer, "ConversationMemoryModule", Broken)
        result = run_agent(build_agent("memoryless_agent"))
        assert result.get_data("output") == "echo:hello"

    def test_async_cancellation_records_an_error(
        self, loongsuite_harness: Any
    ) -> None:
        async def cancel() -> None:
            agent = build_agent("cancel_agent", SlowAsyncAgent)
            task = asyncio.ensure_future(agent.async_run(input="hello"))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel())
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert record.status == "ERROR"
        assert record.au["au.agent.error.type"] == "CancelledError"
        metrics = loongsuite_harness.metrics()
        assert (
            baseline.counter_value(
                metrics,
                "agent_errors_total",
                au_agent_name="cancel_agent",
                au_agent_status="CancelledError",
            )
            == 1
        )

    def test_streaming_agent_error_still_ends_once(
        self, loongsuite_harness: Any
    ) -> None:
        agent = build_agent("stream_boom_agent", FailingStreamingAgent)
        with pytest.raises(RuntimeError):
            run_agent_with_stream(agent)
        records = layer_records(loongsuite_harness, AGENT_SPAN)
        assert len(records) == 1
        assert records[0].status == "ERROR"
        assert records[0].au["au.agent.streaming"] is True


# ---------------------------------------------------------------------------
# Session wiring
# ---------------------------------------------------------------------------


class TestSessionScope:
    def test_instrumentor_does_not_change_global_providers(self) -> None:
        from opentelemetry import propagate, trace

        provider_before = trace.get_tracer_provider()
        textmap_before = propagate.get_global_textmap()
        instrumentor = AgentUniverseInstrumentor()
        instrumentor.instrument(tracer_provider=None, meter_provider=None)
        try:
            assert trace.get_tracer_provider() is provider_before
            assert propagate.get_global_textmap() is textmap_before
        finally:
            instrumentor.uninstrument()
        assert trace.get_tracer_provider() is provider_before
        assert propagate.get_global_textmap() is textmap_before

    def test_session_id_and_propagator_through_telemetry_manager(self) -> None:
        probe = Path(__file__).resolve().parent / "session_probe_child.py"
        completed = subprocess.run(
            [sys.executable, str(probe)],
            capture_output=True,
            text=True,
            cwd=str(PACKAGE_ROOT),
        )
        assert completed.returncode == 0, completed.stderr
        payload = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("PROBE_JSON=")
        ]
        assert payload, completed.stdout
        report = json.loads(payload[0][len("PROBE_JSON=") :])

        assert report["tracer_provider"] == "TracerProvider"
        assert report["output"] == "llm:hello"
        names = [span["name"] for span in report["spans"]]
        assert "au.agent.session_agent" in names
        assert "au.llm.session_llm" in names
        for span in report["spans"]:
            assert (
                span["attributes"]["au.trace.session.id"] == report["session"]
            )
            assert span["attributes"]["gen_ai.framework"] == "agentuniverse"
        assert report["carrier"]["AU-SessionId"] == report["session"]
        assert report["carrier"]["auSessionId"] == report["session"]
        assert report["session_after_extract"] == report["carrier_session"]


# ---------------------------------------------------------------------------
# Package shape
# ---------------------------------------------------------------------------


class TestPackage:
    def test_runtime_source_has_no_native_telemetry_references(self) -> None:
        """No module may name, import or call a framework instrumentation.

        Parsed rather than grepped: the module docstrings say what this
        package deliberately does not use, and prose is not a dependency.
        """
        sources = sorted(SOURCE_ROOT.glob("*.py"))
        assert sources
        for path in sources:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            identifiers = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    identifiers.add(node.id)
                elif isinstance(node, ast.Attribute):
                    identifiers.add(node.attr)
                elif isinstance(node, ast.alias):
                    identifiers.add(node.name.split(".")[-1])
                    if node.asname:
                        identifiers.add(node.asname)
            for token in FORBIDDEN_RUNTIME_REFERENCES:
                assert token not in identifiers, f"{path.name} uses {token}"

    def test_runtime_source_only_imports_allowed_framework_helpers(
        self,
    ) -> None:
        """The framework imports left are business helpers, not telemetry."""
        forbidden_modules = (
            "agentuniverse.base.tracing.otel",
            "agentuniverse.base.tracing.au_trace_context",
        )
        for path in sorted(SOURCE_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for forbidden in forbidden_modules:
                        assert not node.module.startswith(forbidden), (
                            f"{path.name} imports {node.module}"
                        )

    def test_entry_point_declares_the_instrumentor(self) -> None:
        pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        assert (
            "opentelemetry.instrumentation.agentuniverse:AgentUniverseInstrumentor"
            in pyproject
        )

    def test_distribution_entry_point_is_installed(self) -> None:
        try:
            distribution = metadata.distribution(
                "loongsuite-instrumentation-agentuniverse"
            )
        except metadata.PackageNotFoundError:
            pytest.skip("package not installed in this environment")
        assert distribution.version
        assert "agentuniverse" in {
            entry.name for entry in distribution.entry_points
        }

    def test_every_source_module_is_importable(self) -> None:
        import opentelemetry.instrumentation.agentuniverse as package

        for name in ("_common", "_agent", "_llm", "_tool"):
            module = __import__(f"{package.__name__}.{name}", fromlist=[name])
            assert module.__name__ == f"{package.__name__}.{name}"

    def test_wrapper_globals_are_restored_by_the_autouse_fixture(self) -> None:
        for name in WRAPPER_GLOBALS:
            wrapper = getattr(trace_module, name)
            assert getattr(wrapper, "__self__", None) is None, name


def test_agents_used_by_the_matrix_are_real_framework_objects() -> None:
    from agentuniverse.agent.action.tool.tool import Tool
    from agentuniverse.agent.agent import Agent

    assert issubclass(RichAgent, Agent)
    assert issubclass(StubTool, Tool)
    assert issubclass(SecretLLM, StubLLM)
