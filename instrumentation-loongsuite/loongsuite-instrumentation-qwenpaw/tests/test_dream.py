# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dream owner-name propagation without creating an Agent or changing results."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opentelemetry import baggage, context
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import LLMInvocation

memory = pytest.importorskip("qwenpaw.agents.memory.reme_light_memory_manager")
config = pytest.importorskip("qwenpaw.config.config")


@pytest.fixture
def manager(monkeypatch):
    # Run the real dream method without starting ReMe or touching memory files.
    obj = object.__new__(memory.ReMeLightMemoryManager)
    obj.agent_id = "owner-a"
    monkeypatch.setattr(
        config,
        "load_agent_config",
        lambda agent_id: SimpleNamespace(name=agent_id),
    )
    return obj


@pytest.mark.asyncio
async def test_dream_name_and_cleanup(
    manager, monkeypatch, instrument, tracer_provider, span_exporter
):
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    calls = []

    async def job(name, **kwargs):
        calls.append((name, kwargs))
        invocation = LLMInvocation(request_model="test-model", provider="test")
        handler.start_llm(invocation)
        handler.stop_llm(invocation)
        return SimpleNamespace(success=True)

    monkeypatch.setattr(manager, "_run_reme_job", job)
    token = context.attach(baggage.set_baggage("gen_ai.agent.name", "caller"))
    try:
        assert await manager.dream(hint="keep this") is None
        assert baggage.get_baggage("gen_ai.agent.name") == "caller"
    finally:
        context.detach(token)
    assert calls == [
        ("auto_dream", {"needs_llm": True, "date": "", "hint": "keep this"})
    ]
    [span] = span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.agent.name"] == "owner-a"
    assert "gen_ai.agent.id" not in span.attributes
    assert span.attributes["gen_ai.span.kind"] == "LLM"
    normal = LLMInvocation(request_model="test-model", provider="test")
    handler.start_llm(normal)
    handler.stop_llm(normal)
    assert (
        "gen_ai.agent.name"
        not in span_exporter.get_finished_spans()[-1].attributes
    )


@pytest.mark.asyncio
async def test_dream_uninstrument_restores_method(
    manager, monkeypatch, instrument
):
    async def job(*args, **kwargs):
        assert baggage.get_baggage("gen_ai.agent.name") is None
        return SimpleNamespace(success=True)

    monkeypatch.setattr(manager, "_run_reme_job", job)
    instrument.uninstrument()
    assert await manager.dream() is None


@pytest.mark.asyncio
async def test_dream_name_reload_and_default(manager, monkeypatch, instrument):
    profile = SimpleNamespace(name="First")
    monkeypatch.setattr(config, "load_agent_config", lambda agent_id: profile)
    names = []

    async def job(*args, **kwargs):
        names.append(baggage.get_baggage("gen_ai.agent.name"))
        return SimpleNamespace(success=True)

    monkeypatch.setattr(manager, "_run_reme_job", job)
    for name in ("First", "Second", ""):
        profile.name = name
        await manager.dream()
    assert names == ["First", "Second", "QwenPaw"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("business"), asyncio.CancelledError("stop")]
)
async def test_dream_exception_unchanged(
    manager, monkeypatch, instrument, error
):
    calls = []

    async def job(*args, **kwargs):
        calls.append(baggage.get_baggage("gen_ai.agent.name"))
        raise error

    monkeypatch.setattr(manager, "_run_reme_job", job)
    with pytest.raises(type(error)) as caught:
        await manager.dream()
    assert caught.value is error
    assert calls == ["owner-a"]
    assert baggage.get_baggage("gen_ai.agent.name") is None


@pytest.mark.asyncio
async def test_dream_internal_agent_preserves_owner(
    manager, monkeypatch, instrument
):
    from opentelemetry.instrumentation.agentscope._v2_middleware import (  # noqa: PLC0415
        _create_agent_invocation,
    )

    helper = SimpleNamespace(name="default", model=None, state=None)

    async def job(*args, **kwargs):
        invocation = _create_agent_invocation(helper, {})
        assert invocation.agent_name == "owner-a"
        assert helper.name == "default"
        return SimpleNamespace(success=True)

    monkeypatch.setattr(manager, "_run_reme_job", job)
    await manager.dream()
    assert context.get_value("qwenpaw.dream.agent.name") is None
    assert _create_agent_invocation(helper, {}).agent_name == "default"


@pytest.mark.asyncio
async def test_dream_concurrency(manager, monkeypatch, instrument):
    other = object.__new__(memory.ReMeLightMemoryManager)
    other.agent_id = "owner-b"
    seen = []

    async def job(*args, **kwargs):
        before = baggage.get_baggage("gen_ai.agent.name")
        await asyncio.sleep(0)
        seen.append((before, baggage.get_baggage("gen_ai.agent.name")))
        return SimpleNamespace(success=True)

    monkeypatch.setattr(manager, "_run_reme_job", job)
    monkeypatch.setattr(other, "_run_reme_job", job)
    await asyncio.gather(manager.dream(), other.dream())
    assert sorted(seen) == [("owner-a", "owner-a"), ("owner-b", "owner-b")]
    assert baggage.get_baggage("gen_ai.agent.name") is None


@pytest.mark.asyncio
async def test_dream_config_failure_is_fail_open(
    manager, monkeypatch, instrument
):
    def fail(*args):
        raise ValueError("config unavailable")

    monkeypatch.setattr(config, "load_agent_config", fail)
    calls = []

    async def job(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(success=True)

    monkeypatch.setattr(manager, "_run_reme_job", job)
    assert await manager.dream() is None
    assert calls == [1]
    assert baggage.get_baggage("gen_ai.agent.name") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("capture", ["NO_CONTENT", "SPAN_ONLY"])
@pytest.mark.parametrize("internal_agent", [False, True])
async def test_dream_real_model_replay(
    manager,
    monkeypatch,
    instrument,
    tracer_provider,
    span_exporter,
    stream,
    capture,
    internal_agent,
):
    """Reuse real-provider AgentScope cassettes; only ReMe orchestration is stubbed."""
    import aiohttp.streams  # noqa: PLC0415

    monkeypatch.setattr(
        aiohttp.streams, "AsyncStreamReaderMixin", object, raising=False
    )
    vcr = pytest.importorskip("vcr")
    from agentscope.credential import DashScopeCredential  # noqa: PLC0415
    from agentscope.message import Msg, TextBlock  # noqa: PLC0415
    from agentscope.model import DashScopeChatModel  # noqa: PLC0415

    openai_probe = pytest.importorskip(
        "opentelemetry.instrumentation.openai_v2"
    )

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", capture
    )
    if internal_agent:
        from opentelemetry.instrumentation.agentscope import (  # noqa: PLC0415
            AgentScopeInstrumentor,
        )

        agent_inst = AgentScopeInstrumentor()
    else:
        agent_inst = openai_probe.OpenAIInstrumentor()
    agent_inst.instrument(skip_dep_check=True, tracer_provider=tracer_provider)
    model = DashScopeChatModel(
        credential=DashScopeCredential(api_key="test_api_key"),
        model="qwen-plus",
        parameters=DashScopeChatModel.Parameters(
            max_tokens=16, thinking_enable=False
        ),
        stream=stream,
        max_retries=0,
    )
    system = (
        "Reply with a short sentence." if stream else "Reply with exactly: OK"
    )
    user = "Say hello in one sentence." if stream else "Say OK."

    async def job(*args, **kwargs):
        if internal_agent:
            from agentscope.agent import Agent  # noqa: PLC0415
            from agentscope.message import UserMsg  # noqa: PLC0415

            helper = Agent(name="default", system_prompt=system, model=model)
            message = UserMsg(name="user", content=user)
            if stream:
                async for _ in helper.reply_stream(message):
                    pass
            else:
                await helper.reply(message)
            assert helper.name == "default"
            return SimpleNamespace(success=True)
        response = await model(
            [
                Msg(
                    name="system",
                    role="system",
                    content=[TextBlock(text=system)],
                ),
                Msg(name="user", role="user", content=[TextBlock(text=user)]),
            ]
        )
        if stream:
            async for _ in response:
                pass
        return SimpleNamespace(success=True)

    monkeypatch.setattr(manager, "_run_reme_job", job)
    kind = "streaming" if stream else "non_streaming"
    cassette = (
        Path(__file__).resolve().parents[2]
        / "loongsuite-instrumentation-agentscope/tests/cassettes"
        / f"test_v2_agent_{kind}_e2e.yaml"
    )

    def body_match(left, right):
        assert json.loads(left.body) == json.loads(right.body)

    replay = vcr.VCR(
        record_mode="none",
        match_on=[
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "semantic_body",
        ],
    )
    replay.register_matcher("semantic_body", body_match)
    try:
        with replay.use_cassette(str(cassette)) as recorded:
            await manager.dream()
            assert recorded.all_played
    finally:
        agent_inst.uninstrument()
    [span] = [
        item
        for item in span_exporter.get_finished_spans()
        if item.attributes.get("gen_ai.span.kind") == "LLM"
    ]
    assert span.attributes["gen_ai.agent.name"] == "owner-a"
    assert "gen_ai.agent.id" not in span.attributes
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert ("gen_ai.input.messages" in span.attributes) == (
        capture == "SPAN_ONLY"
    )
    assert baggage.get_baggage("gen_ai.agent.name") is None
