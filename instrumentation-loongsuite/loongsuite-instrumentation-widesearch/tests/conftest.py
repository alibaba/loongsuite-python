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

"""Test configuration for WideSearch instrumentation tests.

Injects lightweight stub modules for `src.agent.*` into sys.modules
so that wrap_function_wrapper can find them without installing WideSearch.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

# Ensure workspace opentelemetry-util-genai is imported (not stale site-packages).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTIL_GENAI_SRC = _REPO_ROOT / "util" / "opentelemetry-util-genai" / "src"
if _UTIL_GENAI_SRC.is_dir() and str(_UTIL_GENAI_SRC) not in sys.path:
    sys.path.insert(0, str(_UTIL_GENAI_SRC))
    # Plugins or other loaders may pull opentelemetry.util.genai.* from
    # site-packages before this conftest runs — drop caches so imports resolve here.
    for _m in list(sys.modules):
        if _m == "opentelemetry.util.genai" or _m.startswith(
            "opentelemetry.util.genai."
        ):
            del sys.modules[_m]

_WIDESEARCH_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
if (
    _WIDESEARCH_PLUGIN_SRC.is_dir()
    and str(_WIDESEARCH_PLUGIN_SRC) not in sys.path
):
    sys.path.insert(0, str(_WIDESEARCH_PLUGIN_SRC))

import pytest

# ---------------------------------------------------------------------------
# Stub modules for WideSearch (src.agent.*)
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    USER = "USER"
    FINISHED = "FINISHED"
    CONTINUE = "CONTINUE"
    ERROR = "ERROR"


@dataclass
class ActionStepError:
    message: str
    source: Literal["llm"] = "llm"


@dataclass
class ToolCall:
    tool_name: str
    arguments: Any
    tool_call_id: str


@dataclass
class ErrorMarker:
    message: str

    def __getitem__(self, key):
        if key == "message":
            return self.message
        raise KeyError(key)


@dataclass
class ToolCallResult:
    tool_call_id: str
    content: str | None = None
    error_marker: Any = None
    system_error_marker: Any = None
    extra: dict = field(default_factory=dict)


@dataclass
class LLMOutputItem:
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    signature: str | None = None
    tool_calls: list = field(default_factory=list)


@dataclass
class ModelResponse:
    outputs: list = field(default_factory=list)
    session_id: str | None = None
    error_marker: Any = None


@dataclass
class ActionStep:
    step_status: StepStatus = StepStatus.CONTINUE
    content: str | None = None
    reasoning_content: str | None = None
    signature: str | None = None
    tool_calls: list = field(default_factory=list)
    tool_call_results: list = field(default_factory=list)
    error_marker: Any = None


@dataclass
class UserInputStep:
    user_input: str
    step_status: StepStatus = StepStatus.USER


@dataclass
class MemoryTurn:
    steps: list = field(default_factory=list)

    @property
    def step_number(self):
        return sum(1 for s in self.steps if isinstance(s, ActionStep))

    def is_finished(self) -> bool:
        if not self.steps:
            return False
        return self.steps[-1].step_status == StepStatus.FINISHED


@dataclass
class MemoryAgent:
    system_instructions: str | None = None
    turns: list = field(default_factory=list)

    def insert_user_input(self, user_input: str):
        turn = MemoryTurn()
        turn.steps.append(UserInputStep(user_input=user_input))
        self.turns.append(turn)
        return turn

    def insert_action_step(self, action_step):
        last_turn = self.turns[-1]
        last_turn.steps.append(action_step)
        return last_turn

    def to_message(self, **kwargs):
        return []


@dataclass
class InternalResponse:
    data: Any = None
    error: str | None = None
    system_error: str | None = None
    extra: dict | None = None


@dataclass
class Agent:
    name: str = "test-agent"
    instructions: str | None = "You are a helpful agent."
    tools: dict = field(default_factory=dict)
    tools_desc: list = field(default_factory=list)
    model_config_name: str = "gpt-4o"

    def get_tool_by_name(self, tool_name: str):
        return self.tools.get(tool_name)


DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_ERROR_COUNT = 3


class Runner:
    _step_override = None  # Set to a callable to override _step behavior

    @classmethod
    async def run(
        cls,
        starting_agent,
        user_input: str,
        memory=None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        llm_error_strategy: str = "retry",
    ):
        if memory is None:
            memory = MemoryAgent(
                system_instructions=starting_agent.instructions
            )
        memory.insert_user_input(user_input)
        step_result = await cls._step(agent=starting_agent, memory=memory)
        if not isinstance(step_result, ActionStepError):
            yield step_result

    @classmethod
    async def _step(cls, *, agent, memory) -> ActionStep | ActionStepError:
        if cls._step_override is not None:
            return await cls._step_override(agent=agent, memory=memory)
        return ActionStep(step_status=StepStatus.FINISHED, content="Done")

    @classmethod
    async def _invoke_tool_call(cls, agent, model_response) -> list:
        return []


async def run_single_query(
    query: str,
    agent_name: str = "",
    model_config_name: str = "",
    tools: dict = None,
    tools_desc: list = None,
    system_prompt: str = "",
):
    agent_instructions = (
        system_prompt if system_prompt else "You are a helpful agent."
    )
    agent = Agent(
        name=agent_name,
        tools=tools or {},
        tools_desc=tools_desc or [],
        model_config_name=model_config_name,
        instructions=agent_instructions,
    )
    memory = MemoryAgent(system_instructions=system_prompt)

    # Mirrors real implementation: calls Runner.run as async generator
    async for step in Runner.run(agent, query, memory):
        pass

    last_content = "final answer"
    if memory.turns:
        last_turn = memory.turns[-1]
        for s in reversed(last_turn.steps):
            if isinstance(s, ActionStep) and s.content:
                last_content = s.content
                break

    return [
        {"role": "user", "content": query},
        {"role": "assistant", "content": {"content": last_content}},
    ]


def _default_tools():
    return {}


def get_system_prompt(language="zh"):
    return "You are a helpful assistant."


def create_sub_agents_wrap(
    agent_name, model_config_name, tools, tools_desc, system_prompt
):
    async def create_sub_agents(sub_agents: list) -> InternalResponse:
        import json

        results = []
        for sa in sub_agents:
            results.append(
                {
                    "index": sa.get("index"),
                    "prompt": sa.get("prompt", ""),
                    "response": "sub result",
                }
            )
        return InternalResponse(data=json.dumps(results, ensure_ascii=False))

    return create_sub_agents


def _inject_stub_modules():
    """Inject stub modules into sys.modules so that wrapt can resolve them."""
    # Create module hierarchy: src -> src.agent -> src.agent.run, etc.
    src_mod = types.ModuleType("src")
    src_agent_mod = types.ModuleType("src.agent")
    src_agent_run_mod = types.ModuleType("src.agent.run")
    src_agent_multi_agent_tools_mod = types.ModuleType(
        "src.agent.multi_agent_tools"
    )
    src_agent_memory_mod = types.ModuleType("src.agent.memory")
    src_agent_schema_mod = types.ModuleType("src.agent.schema")
    src_agent_tools_mod = types.ModuleType("src.agent.tools")
    src_agent_prompt_mod = types.ModuleType("src.agent.prompt")
    src_utils_mod = types.ModuleType("src.utils")
    src_utils_config_mod = types.ModuleType("src.utils.config")

    # Populate src.agent.run
    src_agent_run_mod.Runner = Runner
    src_agent_run_mod.run_single_query = run_single_query
    src_agent_run_mod.run_turn = None
    src_agent_run_mod.extract_messages_from_memory = None

    # Populate src.agent.multi_agent_tools
    src_agent_multi_agent_tools_mod.create_sub_agents_wrap = (
        create_sub_agents_wrap
    )

    # Populate src.agent.memory
    src_agent_memory_mod.ActionStep = ActionStep
    src_agent_memory_mod.ActionStepError = ActionStepError
    src_agent_memory_mod.MemoryAgent = MemoryAgent
    src_agent_memory_mod.StepStatus = StepStatus
    src_agent_memory_mod.UserInputStep = UserInputStep

    # Populate src.agent.schema
    src_agent_schema_mod.ToolCall = ToolCall
    src_agent_schema_mod.ToolCallResult = ToolCallResult
    src_agent_schema_mod.ModelResponse = ModelResponse
    src_agent_schema_mod.ErrorMarker = ErrorMarker
    src_agent_schema_mod.LLMOutputItem = LLMOutputItem

    # Populate src.agent.tools
    src_agent_tools_mod.InternalResponse = InternalResponse
    src_agent_tools_mod._default_tools = {}

    # Populate src.agent.prompt
    src_agent_prompt_mod.get_system_prompt = get_system_prompt

    # Populate src.agent.agent
    src_agent_agent_mod = types.ModuleType("src.agent.agent")
    src_agent_agent_mod.Agent = Agent
    src_agent_agent_mod.DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS
    src_agent_agent_mod.DEFAULT_MAX_ERROR_COUNT = DEFAULT_MAX_ERROR_COUNT

    # Populate src.utils.config
    src_utils_config_mod.model_config = {
        "gpt-4o": {"model_name": "gpt-4o-2024-05-13"},
    }

    # Wire up parent references
    src_mod.agent = src_agent_mod
    src_mod.utils = src_utils_mod
    src_agent_mod.run = src_agent_run_mod
    src_agent_mod.multi_agent_tools = src_agent_multi_agent_tools_mod
    src_agent_mod.memory = src_agent_memory_mod
    src_agent_mod.schema = src_agent_schema_mod
    src_agent_mod.tools = src_agent_tools_mod
    src_agent_mod.prompt = src_agent_prompt_mod
    src_agent_mod.agent = src_agent_agent_mod

    # Register in sys.modules
    sys.modules["src"] = src_mod
    sys.modules["src.agent"] = src_agent_mod
    sys.modules["src.agent.run"] = src_agent_run_mod
    sys.modules["src.agent.multi_agent_tools"] = (
        src_agent_multi_agent_tools_mod
    )
    sys.modules["src.agent.memory"] = src_agent_memory_mod
    sys.modules["src.agent.schema"] = src_agent_schema_mod
    sys.modules["src.agent.tools"] = src_agent_tools_mod
    sys.modules["src.agent.prompt"] = src_agent_prompt_mod
    sys.modules["src.agent.agent"] = src_agent_agent_mod
    sys.modules["src.utils"] = src_utils_mod
    sys.modules["src.utils.config"] = src_utils_config_mod


# Inject stubs before any test imports the instrumentation module
_inject_stub_modules()


# ---------------------------------------------------------------------------
# OTel test fixtures
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
        "span_only"
    )


for _m in list(sys.modules):
    if _m.startswith("opentelemetry.instrumentation.widesearch"):
        del sys.modules[_m]

from opentelemetry.instrumentation.widesearch import WideSearchInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    return meter_provider


@pytest.fixture(scope="function")
def instrument(tracer_provider, meter_provider):
    instrumentor = WideSearchInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
