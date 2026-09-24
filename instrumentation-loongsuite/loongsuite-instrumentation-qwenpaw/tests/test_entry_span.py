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

"""Assert Entry span attributes for QwenPaw ``query_handler``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
    GEN_AI_SESSION_ID,
    GEN_AI_SPAN_KIND,
    GEN_AI_USER_ID,
    GenAiSpanKindValues,
)

pytest.importorskip("agentscope.message")

from agentscope.message import Msg, TextBlock  # noqa: E402


def _attrs(span):
    return dict(span.attributes or {})


def _patch_command_path(monkeypatch, runner_module, response_text):
    async def fake_resolve(self, session_id, query):
        del self, session_id, query
        return (None, False, None)

    async def fake_run_command_path(request, msgs, runner):
        del request, msgs, runner
        response = Msg(
            name="Friday",
            role="assistant",
            content=[TextBlock(type="text", text=response_text)],
        )
        yield response, True

    if hasattr(runner_module.AgentRunner, "_resolve_pending_approval"):
        monkeypatch.setattr(
            runner_module.AgentRunner,
            "_resolve_pending_approval",
            fake_resolve,
        )
    monkeypatch.setattr(
        runner_module,
        "run_command_path",
        fake_run_command_path,
    )


@pytest.mark.asyncio
async def test_query_handler_emits_enter_ai_application_system_span(
    instrument,
    runner_module,
    span_exporter,
    monkeypatch,
):
    AgentRunner = runner_module.AgentRunner
    _patch_command_path(monkeypatch, runner_module, "hello-entry")

    runner = AgentRunner(agent_id="entry-agent")
    req = SimpleNamespace(
        session_id="sess-1", user_id="user-2", channel="console"
    )

    msgs = [Msg(name="user", role="user", content="/stop")]
    async for _ in runner.query_handler(msgs, req):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "enter_ai_application_system"
    attrs = _attrs(span)
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "enter"
    assert attrs[GEN_AI_SPAN_KIND] == GenAiSpanKindValues.ENTRY.value
    assert attrs[GEN_AI_SESSION_ID] == "sess-1"
    assert attrs[GEN_AI_USER_ID] == "user-2"
    assert attrs["qwenpaw.agent_id"] == "entry-agent"
    assert attrs["qwenpaw.channel"] == "console"
    assert attrs["copaw.agent_id"] == "entry-agent"
    assert attrs["copaw.channel"] == "console"
    assert attrs.get(GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN) is not None


@pytest.mark.asyncio
async def test_query_handler_captures_input_output_messages_in_span_only_mode(
    instrument,
    runner_module,
    span_exporter,
    monkeypatch,
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    AgentRunner = runner_module.AgentRunner
    _patch_command_path(monkeypatch, runner_module, "hello-output")

    runner = AgentRunner(agent_id="entry-agent")
    req = SimpleNamespace(
        session_id="sess-1", user_id="user-2", channel="console"
    )
    msgs = [Msg(name="user", role="user", content="/stop")]

    async for _ in runner.query_handler(msgs, req):
        pass

    span = span_exporter.get_finished_spans()[0]
    attrs = _attrs(span)
    input_messages = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
    output_messages = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])

    assert input_messages[0]["parts"][0]["content"] == "/stop"
    assert output_messages[0]["parts"][0]["content"] == "hello-output"
