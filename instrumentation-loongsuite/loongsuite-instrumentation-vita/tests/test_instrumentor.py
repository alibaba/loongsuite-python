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

"""Tests for VitaBench instrumentation.

The suite exercises all execute.md hook points. External I/O is replaced at the
HTTP/tool boundary, while the Vita agent/orchestrator call chain runs through
the real framework methods.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.instrumentation.vita import VitaInstrumentor

FAKE_MODELS_CONFIG = {
    "qwen-max": {
        "base_url": "http://fake-api.example.com/v1/chat/completions",
        "headers": {"Authorization": "Bearer test-key"},
    },
    "gpt-4": {
        "base_url": "http://fake-api.example.com/v1/chat/completions",
        "headers": {"Authorization": "Bearer test-key"},
    },
    "claude-3-opus": {
        "base_url": "http://fake-api.example.com/v1/chat/completions",
        "headers": {"Authorization": "Bearer test-key"},
    },
}


def _make_openai_response(content=None, tool_calls=None, usage=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": usage
        or {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


def _mock_requests_post(response_dict):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = response_dict
    return mock_resp


def _tool_call_response():
    return _make_openai_response(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_order",
                    "arguments": '{"order_id": "123"}',
                },
            }
        ],
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    )


def _text_response(content="Order 123 has been delivered. ###STOP###"):
    return _make_openai_response(
        content=content,
        usage={
            "prompt_tokens": 200,
            "completion_tokens": 30,
            "total_tokens": 230,
        },
    )


class FakeTool:
    name = "get_order"
    short_desc = "Get order details"
    openai_schema = {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
            },
        },
    }


class FakeTools:
    def __init__(self):
        self.db = SimpleNamespace(time="2026-01-01 00:00:00")
        self._tools = {"get_order": FakeTool()}

    def get_tools(self):
        return self._tools

    def use_tool(self, tool_name, **kwargs):
        return {"tool": tool_name, "arguments": kwargs, "status": "delivered"}

    def get_db_hash(self):
        return "fake-db-hash"


class DeterministicUser:
    def get_init_state(self, message_history=None):
        return SimpleNamespace(messages=message_history or [])

    def generate_next_message(self, message, state):
        from vita.data_model.message import UserMessage

        user_message = UserMessage(role="user", content="Check order 123")
        state.messages.append(user_message)
        return user_message, state


def _make_agent():
    from vita.agent.llm_agent import LLMAgent

    return LLMAgent(
        tools=[FakeTool()],
        domain_policy="You are helpful at {time}.",
        llm="qwen-max",
        llm_args={},
        time="2026-01-01 00:00:00",
        language="english",
    )


def _make_orchestrator():
    from vita.environment.environment import Environment
    from vita.orchestrator.orchestrator import Orchestrator

    return Orchestrator(
        domain="delivery",
        agent=_make_agent(),
        user=DeterministicUser(),
        environment=Environment(domain_name="delivery", tools=FakeTools()),
        task=SimpleNamespace(
            id="task_001",
            instructions="Check order 123",
            message_history=None,
        ),
        max_steps=6,
        max_errors=3,
        language="english",
    )


def _span_attrs(spans, name):
    span = next(s for s in spans if s.name == name)
    return dict(span.attributes)


class TestVitaInstrumentor:
    def test_instrument_and_uninstrument(
        self, tracer_provider, logger_provider, meter_provider
    ):
        instrumentor = VitaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )
        assert instrumentor._handler is not None
        instrumentor.uninstrument()
        assert instrumentor._handler is None

    def test_instrumentation_dependencies(self):
        assert VitaInstrumentor().instrumentation_dependencies() == (
            "vita >= 0.0.1",
        )


class TestLLMSpan:
    def test_llm_span_text_response(self, instrument, span_exporter):
        from vita.data_model.message import UserMessage
        from vita.utils.llm_utils import generate

        with (
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch(
                "requests.post",
                return_value=_mock_requests_post(
                    _make_openai_response(
                        content="The order has been delivered.",
                        usage={
                            "prompt_tokens": 150,
                            "completion_tokens": 30,
                            "total_tokens": 180,
                        },
                    )
                ),
            ),
        ):
            result = generate(
                model="qwen-max",
                messages=[
                    UserMessage(role="user", content="Where is my order?")
                ],
            )

        assert result.content == "The order has been delivered."
        spans = span_exporter.get_finished_spans()
        attrs = _span_attrs(spans, "chat qwen-max")
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.span.kind"] == "LLM"
        assert attrs["gen_ai.request.model"] == "qwen-max"
        assert attrs["gen_ai.provider.name"] == "alibaba_cloud"
        assert attrs["gen_ai.usage.input_tokens"] == 150
        assert attrs["gen_ai.usage.output_tokens"] == 30
        assert attrs["gen_ai.response.finish_reasons"] == ("stop",)

    def test_llm_span_tool_call_response(self, instrument, span_exporter):
        from vita.data_model.message import UserMessage
        from vita.utils.llm_utils import generate

        with (
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch(
                "requests.post",
                return_value=_mock_requests_post(_tool_call_response()),
            ),
        ):
            result = generate(
                model="gpt-4",
                messages=[UserMessage(role="user", content="Check my order")],
            )

        assert result.tool_calls is not None
        attrs = _span_attrs(span_exporter.get_finished_spans(), "chat gpt-4")
        assert attrs["gen_ai.response.finish_reasons"] == ("tool_calls",)
        assert attrs["gen_ai.provider.name"] == "openai"

    def test_llm_span_captures_positional_tools(
        self, instrument, span_exporter
    ):
        from vita.data_model.message import UserMessage
        from vita.utils.llm_utils import generate

        with (
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch(
                "requests.post",
                return_value=_mock_requests_post(_text_response("Done.")),
            ),
        ):
            generate(
                "qwen-max",
                [UserMessage(role="user", content="Check my order")],
                [FakeTool()],
            )

        attrs = _span_attrs(
            span_exporter.get_finished_spans(), "chat qwen-max"
        )
        assert "gen_ai.tool.definitions" in attrs
        assert "get_order" in attrs["gen_ai.tool.definitions"]


class TestToolSpan:
    def test_tool_span_created(self, instrument, span_exporter):
        from vita.data_model.message import ToolCall
        from vita.environment.environment import Environment

        env = Environment(domain_name="delivery", tools=FakeTools())
        result = env.get_response(
            ToolCall(
                id="tc_42", name="get_order", arguments={"order_id": "999"}
            )
        )

        assert result.content is not None
        attrs = _span_attrs(
            span_exporter.get_finished_spans(), "execute_tool get_order"
        )
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["gen_ai.span.kind"] == "TOOL"
        assert attrs["gen_ai.tool.name"] == "get_order"
        assert attrs["gen_ai.tool.call.id"] == "tc_42"

    def test_tool_span_on_error(self, instrument, span_exporter):
        from vita.data_model.message import ToolCall
        from vita.environment.environment import Environment

        tools = FakeTools()
        tools.use_tool = MagicMock(side_effect=RuntimeError("Tool failed"))
        env = Environment(domain_name="delivery", tools=tools)
        result = env.get_response(
            ToolCall(id="tc_err", name="get_order", arguments={})
        )

        assert result.error is True
        tool_span = next(
            s
            for s in span_exporter.get_finished_spans()
            if s.name == "execute_tool get_order"
        )
        assert tool_span.status.status_code.name == "ERROR"


class TestAgentSpan:
    def test_agent_span_created_for_llm_agent(self, instrument, span_exporter):
        from vita.data_model.message import UserMessage

        agent = _make_agent()
        state = agent.get_init_state([])

        with (
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch(
                "requests.post",
                return_value=_mock_requests_post(_text_response("Sure.")),
            ),
        ):
            assistant_msg, _ = agent.generate_next_message(
                UserMessage(role="user", content="I need help"), state
            )

        assert assistant_msg.content == "Sure."
        spans = span_exporter.get_finished_spans()
        agent_span = next(
            s for s in spans if s.name == "invoke_agent LLMAgent"
        )
        llm_span = next(s for s in spans if s.name == "chat qwen-max")
        attrs = dict(agent_span.attributes)
        assert attrs["gen_ai.operation.name"] == "invoke_agent"
        assert attrs["gen_ai.span.kind"] == "AGENT"
        assert attrs["gen_ai.agent.name"] == "LLMAgent"
        assert attrs["gen_ai.request.model"] == "qwen-max"
        assert llm_span.parent.span_id == agent_span.context.span_id

    def test_agent_span_created_for_llm_solo_agent(
        self, instrument, span_exporter
    ):
        from vita.agent.llm_agent import LLMSoloAgent

        agent = LLMSoloAgent(
            tools=[FakeTool()],
            domain_policy="unused",
            llm="qwen-max",
            llm_args={},
            time="2026-01-01 00:00:00",
            language="english",
        )
        state = agent.get_init_state([])

        with (
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch(
                "requests.post",
                return_value=_mock_requests_post(_tool_call_response()),
            ),
        ):
            agent.generate_next_message(None, state)

        attrs = _span_attrs(
            span_exporter.get_finished_spans(), "invoke_agent LLMSoloAgent"
        )
        assert attrs["gen_ai.span.kind"] == "AGENT"
        assert attrs["gen_ai.agent.name"] == "LLMSoloAgent"


class TestStepAndChainSpans:
    def test_orchestrator_run_creates_chain_steps_agents_llms_and_tools(
        self, instrument, span_exporter
    ):
        responses = [
            _mock_requests_post(_tool_call_response()),
            _mock_requests_post(_text_response()),
        ]

        with (
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch("requests.post", side_effect=responses),
        ):
            result = _make_orchestrator().run()

        assert result.termination_reason == "agent_stop"
        spans = span_exporter.get_finished_spans()
        chain = next(s for s in spans if s.name == "workflow delivery")
        steps = sorted(
            [s for s in spans if s.name == "react step"],
            key=lambda s: s.start_time,
        )
        agents = sorted(
            [s for s in spans if s.name == "invoke_agent LLMAgent"],
            key=lambda s: s.start_time,
        )
        llms = sorted(
            [s for s in spans if s.name == "chat qwen-max"],
            key=lambda s: s.start_time,
        )
        tools = [s for s in spans if s.name == "execute_tool get_order"]

        assert len(steps) == 2
        assert len(agents) == 2
        assert len(llms) == 2
        assert len(tools) == 1

        chain_attrs = dict(chain.attributes)
        assert chain_attrs["gen_ai.operation.name"] == "workflow"
        assert chain_attrs["gen_ai.span.kind"] == "CHAIN"
        assert chain_attrs["gen_ai.framework"] == "vitabench"

        assert dict(steps[0].attributes)["gen_ai.react.round"] == 1
        assert dict(steps[1].attributes)["gen_ai.react.round"] == 2
        for step in steps:
            assert step.parent.span_id == chain.context.span_id
        assert agents[0].parent.span_id == steps[0].context.span_id
        assert agents[1].parent.span_id == steps[1].context.span_id
        assert llms[0].parent.span_id == agents[0].context.span_id
        assert llms[1].parent.span_id == agents[1].context.span_id
        assert tools[0].parent.span_id == steps[0].context.span_id

    def test_open_step_fails_when_env_turn_raises(
        self, instrument, span_exporter
    ):
        with (
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch(
                "requests.post",
                return_value=_mock_requests_post(_tool_call_response()),
            ),
            patch(
                "vita.environment.environment.Environment.get_response",
                side_effect=RuntimeError("env broke"),
            ),
        ):
            with pytest.raises(RuntimeError, match="env broke"):
                _make_orchestrator().run()

        spans = span_exporter.get_finished_spans()
        step = next(s for s in spans if s.name == "react step")
        chain = next(s for s in spans if s.name == "workflow delivery")
        step_attrs = dict(step.attributes)
        assert step.status.status_code.name == "ERROR"
        assert step_attrs["gen_ai.react.finish_reason"] == "error"
        assert chain.status.status_code.name == "ERROR"


class TestEntrySpan:
    def test_run_task_entry_wraps_orchestrator_trace(
        self, instrument, span_exporter
    ):
        from vita.run import run_task

        def fake_internal(**kwargs):
            return _make_orchestrator().run()

        responses = [
            _mock_requests_post(_tool_call_response()),
            _mock_requests_post(_text_response()),
        ]
        task = SimpleNamespace(
            id="task_001",
            instructions="Check order 123",
            message_history=None,
        )

        with (
            patch("vita.run._run_task_internal", side_effect=fake_internal),
            patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
            patch("requests.post", side_effect=responses),
        ):
            result = run_task("delivery", task, "llm_agent", "user_simulator")

        assert result.termination_reason == "agent_stop"
        spans = span_exporter.get_finished_spans()
        entry = next(
            s for s in spans if s.name == "enter_ai_application_system"
        )
        chain = next(s for s in spans if s.name == "workflow delivery")
        attrs = dict(entry.attributes)
        assert attrs["gen_ai.operation.name"] == "enter"
        assert attrs["gen_ai.span.kind"] == "ENTRY"
        assert attrs["gen_ai.framework"] == "vitabench"
        assert "gen_ai.session.id" in attrs
        assert chain.parent.span_id == entry.context.span_id


class TestProviderInference:
    def test_common_provider_names(self, instrument, span_exporter):
        from vita.data_model.message import UserMessage
        from vita.utils.llm_utils import generate

        for model in ("gpt-4", "claude-3-opus", "qwen-max"):
            with (
                patch("vita.utils.llm_utils.models", FAKE_MODELS_CONFIG),
                patch(
                    "requests.post",
                    return_value=_mock_requests_post(
                        _make_openai_response(content="Hi")
                    ),
                ),
            ):
                generate(
                    model=model,
                    messages=[UserMessage(role="user", content="Hi")],
                )

        providers = {
            dict(s.attributes)["gen_ai.request.model"]: dict(s.attributes)[
                "gen_ai.provider.name"
            ]
            for s in span_exporter.get_finished_spans()
            if s.name.startswith("chat ")
        }
        assert providers["gpt-4"] == "openai"
        assert providers["claude-3-opus"] == "anthropic"
        assert providers["qwen-max"] == "alibaba_cloud"
