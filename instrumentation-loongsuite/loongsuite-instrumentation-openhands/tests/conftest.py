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

"""Shared pytest fixtures and stub modules for the OpenHands instrumentation.

We deliberately don't require ``openhands-ai`` to be installed at test time:
instead we register lightweight stub modules under the same dotted paths so
``wrap_function_wrapper`` can patch them. The wrappers themselves only rely on
the *call signatures* documented in ``execute.md`` — which we faithfully
reproduce in the stubs.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def _ensure_stub_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = _ensure_stub_module(parent_name)
        setattr(parent, leaf, mod)
    return mod


def _install_v0_stub_modules() -> None:
    """Stubs for the V0 (Legacy CodeAct) hook points."""
    _ensure_stub_module("openhands")
    _ensure_stub_module("openhands.core")
    main_mod = _ensure_stub_module("openhands.core.main")
    loop_mod = _ensure_stub_module("openhands.core.loop")
    _ensure_stub_module("openhands.controller")
    ctrl_mod = _ensure_stub_module("openhands.controller.agent_controller")
    _ensure_stub_module("openhands.runtime")
    rt_base = _ensure_stub_module("openhands.runtime.base")

    @dataclass
    class _AgentState:
        value: str = "running"

    @dataclass
    class _State:
        agent_state: _AgentState = field(default_factory=_AgentState)
        history: list = field(default_factory=list)

    @dataclass
    class _LLMConfig:
        model: str = "qwen3-coder-plus"

    @dataclass
    class _LLM:
        config: _LLMConfig = field(default_factory=_LLMConfig)

    @dataclass
    class _Agent:
        name: str = "CodeActAgent"
        llm: _LLM = field(default_factory=_LLM)
        # Mirrors litellm ChatCompletionToolParam dicts as produced by
        # openhands.agenthub.codeact_agent.codeact_agent.CodeActAgent._get_tools.
        tools: list = field(
            default_factory=lambda: [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_bash",
                        "description": "Run a bash command on the runtime sandbox.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"},
                            },
                            "required": ["command"],
                        },
                    },
                },
            ]
        )

    class AgentController:
        step_calls = 0
        close_calls = 0

        def __init__(self, agent=None, sid="sid-test", is_delegate=False):
            self.agent = agent or _Agent()
            self.id = sid
            self.state = _State()
            self._pending_action = None
            self.is_delegate = is_delegate

        async def _step(self) -> None:
            # Support test error injection via flag
            err = getattr(self, "_test_raise_in_step", None)
            if err:
                self._test_raise_in_step = None  # one-shot
                raise err
            # Support empty-step testing: skip work when flag is set
            if getattr(self, "_test_skip_work", False):
                self._test_skip_work = False  # one-shot
                return
            type(self).step_calls += 1

            class _Pending:
                action = "run"
                command = "echo step"
                thought = "trying"

            self._pending_action = _Pending()
            # Simulate work: grow history so the wrapper's work-detection passes.
            self.state.history.append(_Pending())

        async def close(self, set_stop_state: bool = True) -> None:
            # Support test error injection via flag
            err = getattr(self, "_test_raise_in_close", None)
            if err:
                self._test_raise_in_close = None
                raise err
            type(self).close_calls += 1

    ctrl_mod.AgentController = AgentController

    class _ToolCallMetadata:
        """Stand-in for :class:`openhands.events.tool.ToolCallMetadata`."""

        def __init__(self, function_name="", tool_call_id="", arguments=None):
            import json as _json

            self.function_name = function_name
            self.tool_call_id = tool_call_id

            class _Fn:
                def __init__(self, name, args):
                    self.name = name
                    self.arguments = _json.dumps(args or {})

            class _TC:
                def __init__(self, tcid, fn):
                    self.id = tcid
                    self.function = fn

            class _Msg:
                def __init__(self, tcs):
                    self.tool_calls = tcs

            class _Choice:
                def __init__(self, msg):
                    self.message = msg

            class _ModelResp:
                def __init__(self, choices):
                    self.choices = choices

            self.model_response = _ModelResp(
                [
                    _Choice(
                        _Msg(
                            [_TC(tool_call_id, _Fn(function_name, arguments))]
                        )
                    )
                ]
            )

    class _Action:
        def __init__(
            self,
            action_type="run",
            command="echo hi",
            tool_call_metadata=None,
        ):
            self.action = action_type
            self.command = command
            self.tool_call_metadata = tool_call_metadata

    class _Observation:
        def __init__(self, exit_code=0, content=""):
            self.exit_code = exit_code
            self.content = content
            self.observation = "run"

    class Runtime:
        run_action_calls = 0
        # Tests can override on the instance to drive observation values.
        _next_observation: _Observation | None = None

        def __init__(self, sid="sid-test"):
            self.sid = sid

        def run_action(self, action) -> _Observation:
            # Support test error injection via flag
            err = getattr(self, "_test_raise_in_run", None)
            if err:
                self._test_raise_in_run = None
                raise err
            type(self).run_action_calls += 1
            obs = self._next_observation
            if obs is not None:
                self._next_observation = None
                return obs
            return _Observation(exit_code=0)

    rt_base.Runtime = Runtime
    rt_base.Action = _Action
    rt_base.Observation = _Observation
    rt_base.ToolCallMetadata = _ToolCallMetadata

    # LLM stub — lets the LLMInitWrapper patch succeed.
    _ensure_stub_module("openhands.llm")
    llm_mod = _ensure_stub_module("openhands.llm.llm")

    class LLM:
        def __init__(self, config=None):
            self.config = config
            self._completion = lambda *a, **kw: None
            self._completion_unwrapped = lambda *a, **kw: None

    llm_mod.LLM = LLM

    @dataclass
    class _State2:
        agent_state: _AgentState = field(
            default_factory=lambda: _AgentState("finished")
        )

    async def run_controller(
        config=None,
        initial_user_action=None,
        sid: str | None = None,
        **kwargs,
    ):
        if getattr(main_mod, "_test_raise_cancelled", False):
            raise asyncio.CancelledError()
        # Mirror real V0: invoke the agent loop *inside* run_controller so
        # the AGENT span lives within the ENTRY span (and inherits its
        # stashed OTel context). Tests can install
        # ``main_mod._test_inner_args = (controller, runtime)`` to opt in.
        inner_args = getattr(main_mod, "_test_inner_args", None)
        if inner_args is not None:
            controller, runtime = inner_args
            await loop_mod.run_agent_until_done(controller, runtime, None, [])
        return _State2()

    main_mod.run_controller = run_controller

    async def run_agent_until_done(controller, runtime, memory, end_states):
        # Tests can install a custom inner callback to drive STEP / TOOL
        # spans inside the AGENT span; default is a no-op.
        cb = getattr(loop_mod, "_test_inner_callback", None)
        if callable(cb):
            await cb(controller, runtime)
        return None

    loop_mod.run_agent_until_done = run_agent_until_done


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracer_provider() -> TracerProvider:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider._exporter = exporter  # type: ignore[attr-defined]
    return provider


@pytest.fixture
def stub_openhands_v0_modules() -> None:
    _install_v0_stub_modules()


@pytest.fixture(autouse=True)
def _reset_global_tracer():
    """Avoid bleed-through of the SDK provider between tests."""
    yield
    trace_api._TRACER_PROVIDER = None  # type: ignore[attr-defined]
