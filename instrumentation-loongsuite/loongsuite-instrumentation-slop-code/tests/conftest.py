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

"""Test configuration for slop-code instrumentation tests."""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


def _make_module(name):
    """Create a real module object."""
    mod = types.ModuleType(name)
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    return mod


def _create_mock_slop_code_modules():
    """Create mock modules for slop_code so instrumentation can wrap them."""
    # Create all parent modules
    mod_slop_code = _make_module("slop_code")
    mod_entrypoints = _make_module("slop_code.entrypoints")
    mod_commands = _make_module("slop_code.entrypoints.commands")
    mod_run_agent = _make_module("slop_code.entrypoints.commands.run_agent")
    mod_problem_runner = _make_module("slop_code.entrypoints.problem_runner")
    mod_worker = _make_module("slop_code.entrypoints.problem_runner.worker")
    mod_driver = _make_module("slop_code.entrypoints.problem_runner.driver")
    mod_agent_runner = _make_module("slop_code.agent_runner")
    mod_runner = _make_module("slop_code.agent_runner.runner")
    mod_agent = _make_module("slop_code.agent_runner.agent")
    mod_agents = _make_module("slop_code.agent_runner.agents")
    mod_miniswe = _make_module("slop_code.agent_runner.agents.miniswe")
    mod_metrics = _make_module("slop_code.metrics")
    mod_rubric = _make_module("slop_code.metrics.rubric")
    mod_router = _make_module("slop_code.metrics.rubric.router")

    # --- ENTRY: run_agent ---
    def run_agent(*args, **kwargs):
        return {"status": "completed"}

    mod_run_agent.run_agent = run_agent

    # --- WORKFLOW: run_agent_on_problem ---
    def run_agent_on_problem(*args, **kwargs):
        return {"summary": {"state": "completed", "passed_policy": True}}

    mod_worker.run_agent_on_problem = run_agent_on_problem
    # driver re-imports the worker name at module load time. This mock mirrors
    # the same pattern so the instrumentor's driver-side patch has a target.
    mod_driver.run_agent_on_problem = run_agent_on_problem

    # --- TASK: AgentRunner._run_checkpoint ---
    class AgentRunner:
        def __init__(self):
            self.agent = MagicMock()
            self.agent.usage = MagicMock()
            self.agent.usage.net_tokens = MagicMock()
            self.agent.usage.net_tokens.input = 100
            self.agent.usage.net_tokens.output = 50
            self.run_spec = MagicMock()
            self.run_spec.problem = MagicMock()
            self.run_spec.problem.name = "test_problem"
            self.run_spec.problem.prompt = "Solve the coding problem"

        def run(self):
            return {"status": "completed"}

        def _run_checkpoint(
            self, checkpoint, checkpoint_save_dir, is_first_checkpoint=False
        ):
            result = MagicMock()
            result.had_error = False
            result.passed_policy = True
            return result

    mod_runner.AgentRunner = AgentRunner

    # --- AGENT: Agent.run_checkpoint ---
    class Agent:
        def __init__(self, problem_name="test_problem"):
            self.problem_name = problem_name
            self.system_template = (
                "You are a coding agent. Solve the given programming problem."
            )
            self.usage = MagicMock()
            self.usage.net_tokens = MagicMock()
            self.usage.net_tokens.input = 100
            self.usage.net_tokens.output = 50
            self.usage.steps = 0
            self.usage.cost = 0.05

        def run_checkpoint(self, task):
            result = MagicMock()
            result.usage = self.usage
            result.elapsed = 10.5
            return result

    mod_agent.Agent = Agent

    # --- STEP: MiniSWEAgent.agent_step ---
    class MiniSWEAgent(Agent):
        def __init__(self, problem_name="test_problem"):
            super().__init__(problem_name)
            self._messages = [
                {"role": "system", "content": "You are a coding assistant"},
                {"role": "user", "content": "Fix the bug"},
            ]

        def agent_step(self):
            return {
                "token_usage": MagicMock(
                    input=200, output=80, cache_read=50, cache_write=10
                ),
                "step_cost": 0.01,
                "content": "I will fix this bug by editing the code.",
            }

        def get_observation(self):
            return "File modified successfully"

        def execute_action(self, action):
            return {"output": "command executed", "exit_code": 0}

    mod_miniswe.MiniSWEAgent = MiniSWEAgent

    # Also register under the internal module name that the instrumentor patches
    mod_miniswe_agent = _make_module(
        "slop_code.agent_runner.agents._miniswe_agent"
    )
    mod_miniswe_agent.MiniSWEAgent = MiniSWEAgent

    # --- LLM: grade_file_async ---
    async def grade_file_async(*args, **kwargs):
        grades = [{"score": 8, "reasoning": "Good code"}]
        response_data = {
            "id": "resp-123",
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 50,
            },
        }
        return grades, response_data

    mod_router.grade_file_async = grade_file_async

    # Wire parent-child relationships
    mod_slop_code.entrypoints = mod_entrypoints
    mod_slop_code.agent_runner = mod_agent_runner
    mod_slop_code.metrics = mod_metrics
    mod_entrypoints.commands = mod_commands
    mod_entrypoints.problem_runner = mod_problem_runner
    mod_commands.run_agent = mod_run_agent
    mod_problem_runner.worker = mod_worker
    mod_problem_runner.driver = mod_driver
    mod_agent_runner.runner = mod_runner
    mod_agent_runner.agent = mod_agent
    mod_agent_runner.agents = mod_agents
    mod_agents.miniswe = mod_miniswe
    mod_agents._miniswe_agent = mod_miniswe_agent
    mod_metrics.rubric = mod_rubric
    mod_rubric.router = mod_router

    # Register all modules in sys.modules
    modules = {
        "slop_code": mod_slop_code,
        "slop_code.entrypoints": mod_entrypoints,
        "slop_code.entrypoints.commands": mod_commands,
        "slop_code.entrypoints.commands.run_agent": mod_run_agent,
        "slop_code.entrypoints.problem_runner": mod_problem_runner,
        "slop_code.entrypoints.problem_runner.worker": mod_worker,
        "slop_code.entrypoints.problem_runner.driver": mod_driver,
        "slop_code.agent_runner": mod_agent_runner,
        "slop_code.agent_runner.runner": mod_runner,
        "slop_code.agent_runner.agent": mod_agent,
        "slop_code.agent_runner.agents": mod_agents,
        "slop_code.agent_runner.agents._miniswe_agent": mod_miniswe_agent,
        "slop_code.agent_runner.agents.miniswe": mod_miniswe,
        "slop_code.metrics": mod_metrics,
        "slop_code.metrics.rubric": mod_rubric,
        "slop_code.metrics.rubric.router": mod_router,
    }

    for name, mod in modules.items():
        sys.modules[name] = mod

    return modules


# Install mock modules before any instrumentation imports
_mock_modules = _create_mock_slop_code_modules()


@pytest.fixture(scope="function")
def span_exporter():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function")
def tracer_provider(span_exporter):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrument(tracer_provider):
    from opentelemetry.instrumentation.slop_code import SlopCodeInstrumentor

    instrumentor = SlopCodeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
