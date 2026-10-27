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

import asyncio
import json

import httpx
import pytest
from agentuniverse.agent.agent import Agent
from agentuniverse.base.annotation import trace as trace_module
from agentuniverse.llm.llm_channel.llm_channel import LLMChannel
from openai import OpenAI

from opentelemetry.instrumentation.agentuniverse import (
    AgentUniverseInstrumentor,
)
from opentelemetry.instrumentation.agentuniverse._step import (
    REACT_STEP_HOOKS,
)

from .baseline import collect_metrics, counter_value
from .conftest import NATIVE_WRAPPER_GLOBALS
from .runtime_app import Workload
from .runtime_provider import MODEL, prompt
from .runtime_provider import Provider as ToolProvider


def _stream_event_usage(chunk):
    """Read token usage from a stream chunk across SDK layouts.

    stable 0.0.19.1 yields raw OpenAI chunk dicts on ``chunk.raw``; alpha
    0.1.0a2 yields LLMStreamEvent objects that carry a typed ``usage`` and no
    ``raw`` attribute. Both must still prove at least one usage-bearing chunk.
    """
    raw = getattr(chunk, "raw", None)
    if isinstance(raw, dict):
        return raw.get("usage")
    return getattr(chunk, "usage", None)


def test_stable_without_native_round_hooks_enables_all_core_wrappers():
    # stable 0.0.19.1 ships no native round hooks; alpha ships them natively.
    had_native_hooks = any(hasattr(Agent, name) for name in REACT_STEP_HOOKS)
    original = {
        name: getattr(trace_module, name) for name in NATIVE_WRAPPER_GLOBALS
    }
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument()
    assert instrumentor.is_instrumented_by_opentelemetry
    assert all(
        getattr(trace_module, name) is not value
        for name, value in original.items()
    )
    instrumentor.uninstrument()
    assert all(
        getattr(trace_module, name) is value
        for name, value in original.items()
    )
    if had_native_hooks:
        # alpha: our wrappers must be removed, the native hooks restored.
        assert all(hasattr(Agent, name) for name in REACT_STEP_HOOKS)
    else:
        # stable: no native round hooks and still none after uninstall.
        assert not any(hasattr(Agent, name) for name in REACT_STEP_HOOKS)


def test_partial_round_hook_install_is_atomic(monkeypatch):
    original = {
        name: getattr(trace_module, name) for name in NATIVE_WRAPPER_GLOBALS
    }
    native_hooks = {name: getattr(Agent, name) for name in REACT_STEP_HOOKS}
    # Force the second round-hook assignment to fail mid-install; the whole
    # install must roll back, leaving neither instrumentation nor side effects.
    # Class attribute assignment goes through the metaclass __setattr__.
    metaclass = type(Agent)
    real_setattr = metaclass.__setattr__
    events: list[str] = []
    after_failures = {"n": 0}

    def fake_setattr(cls, name, value):
        if (
            cls is Agent
            and name == "on_after_llm_round"
            and after_failures["n"] == 0
        ):
            after_failures["n"] += 1
            events.append("after-assign-failed")
            raise RuntimeError("boom: second hook assignment failed")
        events.append("assign:" + name)
        return real_setattr(cls, name, value)

    monkeypatch.setattr(metaclass, "__setattr__", fake_setattr)
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument()
    assert not instrumentor.is_instrumented_by_opentelemetry
    # Exactly one failure, on the second hook; the first hook installed first.
    assert events.count("after-assign-failed") == 1
    assert events[0].startswith("assign:on_before_llm_round")
    assert all(
        getattr(trace_module, name) is value
        for name, value in original.items()
    )
    assert all(
        getattr(Agent, name) is value for name, value in native_hooks.items()
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_stable_sdk_raw_tool_calls_and_fragments(
    streaming, loongsuite_harness, monkeypatch
):
    """A real SDK/channel call; this does not claim native ReAct dispatch IDs."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    backend = ToolProvider()
    channel = LLMChannel(
        channel_name="openai",
        channel_model_name=MODEL,
        channel_api_base="http://provider.invalid/v1",
        ext_params={
            "reasoning_effort": "high",
            "stream_options": {"include_usage": True},
        },
    )
    channel.channel_model_config = dict(
        streaming=streaming, temperature=0, max_tokens=512
    )
    with OpenAI(
        api_key="synthetic",
        base_url="http://provider.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(backend.respond)
        ),
    ) as client:
        channel.client = client
        result = channel.call(
            messages=[{"role": "user", "content": prompt("rawStable", 1)}]
        )
        if streaming:
            chunks = list(result)
            assert any(_stream_event_usage(c) for c in chunks)
        else:
            assert result.raw["choices"][0]["message"]["tool_calls"]
    spans = loongsuite_harness.spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    messages = json.loads(attrs["gen_ai.output.messages"])
    calls = [
        p
        for message in messages
        for p in message["parts"]
        if p["type"] == "tool_call"
    ]
    expected = backend.responses[0]["tool_calls"]
    assert {p["id"] for p in calls} == {c["id"] for c in expected}
    for part, call in zip(calls, expected):
        assert part["name"] == call["function"]["name"]
        assert part["arguments"] == json.loads(call["function"]["arguments"])
    assert list(attrs["gen_ai.response.finish_reasons"]) == ["tool_calls"]
    assert attrs["gen_ai.usage.input_tokens"] == 20
    assert attrs["gen_ai.usage.output_tokens"] == 10


@pytest.mark.parametrize("mode", ["SPAN_ONLY", "NO_CONTENT"])
@pytest.mark.parametrize("reasoning", [True, False])
def test_stable_native_react_conversations(
    mode, reasoning, monkeypatch, configured_providers, session_exporter
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", mode
    )
    provider, meter_provider, reader = configured_providers
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider, meter_provider=meter_provider
    )
    assert instrumentor.is_instrumented_by_opentelemetry
    workload = Workload(reasoning=reasoning)
    result = asyncio.run(workload.run(provider))
    assert result["request_count"] == 24
    assert len(result["scenarios"]) == 6
    spans = session_exporter.get_finished_spans()
    llms = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"]
    agents = [
        s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
    ]
    tools = [
        s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
    ]
    assert (len(llms), len(agents), len(tools)) == (24, 6, 18)
    # Text ReAct has no provider tool-call IDs. Do not invent IDs for it.
    assert all("gen_ai.tool.call.id" not in s.attributes for s in tools)
    by_id = {s.context.span_id: s for s in spans}
    sessions = {"auSync", "auStreamA", "auStreamB"}
    for span in llms + tools + agents:
        assert span.attributes["gen_ai.session.id"] in sessions
        assert span.parent is not None
        parent = by_id[span.parent.span_id]
        assert parent.context.trace_id == span.context.trace_id
        if span in llms + tools:
            assert parent.attributes.get("gen_ai.span.kind") == "AGENT"
            assert (
                parent.attributes["gen_ai.session.id"]
                == span.attributes["gen_ai.session.id"]
            )
    expected = {r["id"]: r for r in workload.backend.responses}
    for span in llms:
        attrs = span.attributes
        assert attrs["gen_ai.usage.input_tokens"] == 20
        assert attrs["gen_ai.usage.output_tokens"] == 10
        assert attrs["au.llm.usage.total_tokens"] == 30
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.request.model"] == MODEL
        assert attrs["gen_ai.request.temperature"] == 0
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 5
        response = expected[attrs["gen_ai.response.id"]]
        assert attrs["gen_ai.session.id"] == response["worker"]
        if mode == "SPAN_ONLY":
            inputs = json.loads(attrs["gen_ai.input.messages"])
            outputs = json.loads(attrs["gen_ai.output.messages"])
            serialized = json.dumps(inputs)
            worker = response["worker"]
            assert worker in serialized
            assert not any(
                other in serialized for other in sessions - {worker}
            )
            parts = [p for m in outputs for p in m["parts"]]
            texts = [p["content"] for p in parts if p["type"] == "text"]
            reasons = [p["content"] for p in parts if p["type"] == "reasoning"]
            assert "".join(texts) == response["content"]
            assert reasons == ([response["reasoning"]] if reasoning else [])
    for span in tools:
        attrs = span.attributes
        if mode == "SPAN_ONLY":
            args = json.loads(attrs["gen_ai.tool.call.arguments"])
            assert args["worker"] == attrs["gen_ai.session.id"]
            actual = [
                x
                for x in workload.tool_calls
                if x["args"] == args and x["name"] == attrs["gen_ai.tool.name"]
            ]
            assert len(actual) == 1
            assert str(actual[0]["result"]) in attrs["gen_ai.tool.call.result"]
    for span in agents:
        assert span.attributes["gen_ai.usage.input_tokens"] == 80
        assert span.attributes["gen_ai.usage.output_tokens"] == 40
        if mode == "SPAN_ONLY":
            outputs = json.loads(span.attributes["gen_ai.output.messages"])
            assert len(outputs) == 1 and len(outputs[0]["parts"]) == 1
            assert outputs[0]["parts"][0]["content"].startswith("[")
    metrics = collect_metrics(reader)
    assert counter_value(metrics, "llm_total_tokens") == 720
    assert counter_value(metrics, "agent_total_tokens") == 720
    if mode == "NO_CONTENT":
        for span in spans:
            assert not any(
                k in span.attributes
                for k in (
                    "gen_ai.input.messages",
                    "gen_ai.output.messages",
                    "gen_ai.tool.call.arguments",
                    "gen_ai.tool.call.result",
                )
            )
            assert not any(
                k.endswith((".input", ".output"))
                for k in span.attributes
                if k.startswith("au.")
            )
