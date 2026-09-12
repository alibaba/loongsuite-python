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

"""Test configuration for VitaBench instrumentation tests."""

import json
import os
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from opentelemetry.instrumentation.vita import VitaInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def _add_module(name):
    module = ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module

    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)

    return module


def _install_vita_stubs():
    """Install the VitaBench surface used by these tests.

    The PyPI ``vita`` package currently does not expose the VitaBench module
    tree used by the instrumentation hooks, so the test environment owns a
    minimal deterministic framework surface.
    """

    vita_mod = _add_module("vita")
    data_model_mod = _add_module("vita.data_model")
    message_mod = _add_module("vita.data_model.message")
    utils_mod = _add_module("vita.utils")
    llm_utils_mod = _add_module("vita.utils.llm_utils")
    agent_mod = _add_module("vita.agent")
    llm_agent_mod = _add_module("vita.agent.llm_agent")
    environment_mod = _add_module("vita.environment")
    environment_impl_mod = _add_module("vita.environment.environment")
    orchestrator_mod = _add_module("vita.orchestrator")
    orchestrator_impl_mod = _add_module("vita.orchestrator.orchestrator")
    run_mod = _add_module("vita.run")

    class Message:
        def __init__(
            self,
            role=None,
            content=None,
            message_id=None,
            name=None,
            arguments=None,
            tool_calls=None,
            usage=None,
            error=False,
            **kwargs,
        ):
            if message_id is None:
                message_id = kwargs.pop("id", None)
            self.role = role
            self.content = content
            self.id = message_id
            self.name = name
            self.arguments = arguments or {}
            self.tool_calls = tool_calls
            self.usage = usage
            self.error = error
            for key, value in kwargs.items():
                setattr(self, key, value)

        def is_tool_call(self):
            return bool(self.tool_calls)

    class UserMessage(Message):
        def __init__(self, role="user", content=None, **kwargs):
            super().__init__(role=role, content=content, **kwargs)

    class AssistantMessage(Message):
        def __init__(self, role="assistant", content=None, **kwargs):
            super().__init__(role=role, content=content, **kwargs)

    class ToolCall(Message):
        def __init__(
            self, message_id=None, name=None, arguments=None, **kwargs
        ):
            if message_id is None:
                message_id = kwargs.pop("id", None)
            super().__init__(
                role="assistant",
                message_id=message_id,
                name=name,
                arguments=arguments,
                **kwargs,
            )

    class ToolMessage(Message):
        def __init__(
            self, message_id=None, content=None, error=False, **kwargs
        ):
            if message_id is None:
                message_id = kwargs.pop("id", None)
            super().__init__(
                role="tool",
                message_id=message_id,
                content=content,
                error=error,
                **kwargs,
            )

    def _decode_arguments(arguments):
        if not arguments:
            return {}
        try:
            return json.loads(arguments)
        except Exception:
            return arguments

    llm_utils_mod.models = {}

    def generate(model, messages, tools=None, **kwargs):
        import requests

        model_config = llm_utils_mod.models.get(model, {})
        response = requests.post(
            model_config.get("base_url", "http://example.test"),
            headers=model_config.get("headers", {}),
            json={"model": model, "messages": messages, "tools": tools},
            **kwargs,
        )
        payload = response.json()
        message = payload["choices"][0]["message"]
        usage = payload.get("usage")
        tool_calls = []
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            tool_calls.append(
                ToolCall(
                    id=tool_call.get("id"),
                    name=function.get("name"),
                    arguments=_decode_arguments(function.get("arguments")),
                )
            )

        return AssistantMessage(
            content=message.get("content"),
            tool_calls=tool_calls or None,
            usage=usage,
        )

    llm_utils_mod.generate = generate

    class LLMAgent:
        def __init__(
            self,
            tools,
            domain_policy,
            llm,
            llm_args,
            time,
            language,
        ):
            self.tools = tools
            self.domain_policy = domain_policy
            self.llm = llm
            self.llm_args = llm_args
            self.time = time
            self.language = language

        def get_init_state(self, message_history=None):
            system_content = self.domain_policy.format(time=self.time)
            return SimpleNamespace(
                messages=list(message_history or []),
                system_messages=[
                    SimpleNamespace(role="system", content=system_content)
                ],
            )

        def generate_next_message(self, message, state):
            if message is not None:
                state.messages.append(message)
            assistant_message = llm_utils_mod.generate(
                self.llm, state.messages, self.tools, **self.llm_args
            )
            state.messages.append(assistant_message)
            return assistant_message, state

    class LLMSoloAgent(LLMAgent):
        def generate_next_message(self, message, state):
            assistant_message = llm_utils_mod.generate(
                self.llm, state.messages, self.tools, **self.llm_args
            )
            state.messages.append(assistant_message)
            return assistant_message, state

    llm_agent_mod.LLMAgent = LLMAgent
    llm_agent_mod.LLMSoloAgent = LLMSoloAgent

    class Environment:
        def __init__(self, domain_name, tools):
            self.domain_name = domain_name
            self.tools = tools

        def get_response(self, message):
            try:
                result = self.tools.use_tool(message.name, **message.arguments)
                return ToolMessage(
                    id=message.id,
                    content=json.dumps(result, default=str),
                    error=False,
                )
            except Exception as exc:
                return ToolMessage(
                    id=message.id,
                    content=str(exc),
                    error=True,
                )

    environment_impl_mod.Environment = Environment

    class Role(Enum):
        AGENT = "agent"
        USER = "user"
        ENVIRONMENT = "environment"

    class Orchestrator:
        def __init__(
            self,
            domain,
            agent,
            user,
            environment,
            task,
            max_steps,
            max_errors,
            language,
        ):
            self.domain = domain
            self.agent = agent
            self.user = user
            self.environment = environment
            self.task = task
            self.max_steps = max_steps
            self.max_errors = max_errors
            self.language = language
            self.to_role = Role.AGENT
            self.done = False
            self.termination_reason = None
            self.message = None
            self.state = self.agent.get_init_state(task.message_history)
            self.user_state = self.user.get_init_state(task.message_history)

        def run(self):
            for _ in range(self.max_steps):
                self.to_role = Role.AGENT
                self.step()
                if self.done:
                    break
            return SimpleNamespace(
                termination_reason=self.termination_reason or "agent_stop",
                reward_info=None,
            )

        def step(self):
            if self.message is None:
                self.message, self.user_state = (
                    self.user.generate_next_message(None, self.user_state)
                )

            assistant_message, self.state = self.agent.generate_next_message(
                self.message, self.state
            )
            self.message = assistant_message

            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_message = self.environment.get_response(tool_call)
                    self.state.messages.append(tool_message)
                return assistant_message

            self.done = True
            self.termination_reason = "agent_stop"
            return assistant_message

    orchestrator_impl_mod.Role = Role
    orchestrator_impl_mod.Orchestrator = Orchestrator

    def _run_task_internal(**kwargs):
        return kwargs

    def run_task(domain, task, agent_type, user_type, **kwargs):
        return run_mod._run_task_internal(
            domain=domain,
            task=task,
            agent_type=agent_type,
            user_type=user_type,
            **kwargs,
        )

    run_mod._run_task_internal = _run_task_internal
    run_mod.run_task = run_task

    data_model_mod.message = message_mod
    message_mod.Message = Message
    message_mod.UserMessage = UserMessage
    message_mod.AssistantMessage = AssistantMessage
    message_mod.ToolCall = ToolCall
    message_mod.ToolMessage = ToolMessage
    vita_mod.data_model = data_model_mod
    vita_mod.utils = utils_mod
    vita_mod.agent = agent_mod
    vita_mod.environment = environment_mod
    vita_mod.orchestrator = orchestrator_mod
    vita_mod.run = run_mod


def pytest_configure(config: pytest.Config):
    _install_vita_stubs()
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
        "SPAN_ONLY"
    )


# ==================== Exporters ====================


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    reader = InMemoryMetricReader()
    yield reader


# ==================== Providers ====================


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
    )
    return meter_provider


# ==================== Instrumentation ====================


@pytest.fixture(scope="function")
def instrument(tracer_provider, logger_provider, meter_provider):
    instrumentor = VitaInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
