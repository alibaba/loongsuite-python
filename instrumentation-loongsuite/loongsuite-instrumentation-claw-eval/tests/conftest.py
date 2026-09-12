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

"""Test configuration and mock claw_eval modules for claw-eval instrumentation tests."""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Environment setup -- must happen before any OTel semconv import
# ---------------------------------------------------------------------------

os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
)

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


@dataclass
class ContentBlock:
    """Minimal claw-eval ContentBlock mock."""

    type: str = "text"
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any = None
    tool_use_id: str | None = None
    content: list | None = (
        None  # for tool_result blocks containing inner blocks
    )


@dataclass
class ToolUse:
    """Minimal claw-eval ToolUse mock."""

    name: str = "bash"
    id: str = "tool_use_001"
    input: dict | None = None


@dataclass
class ToolResultBlock:
    """Minimal claw-eval ToolResultBlock mock."""

    content: list | None = None
    is_error: bool = False


@dataclass
class Usage:
    """Minimal claw-eval Usage mock."""

    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class Message:
    """Minimal claw-eval Message mock."""

    role: str = "assistant"
    content: list | None = None


@dataclass
class Prompt:
    """Minimal claw-eval Prompt mock."""

    text: str = "Do the task"


@dataclass
class TaskDefinition:
    """Minimal claw-eval TaskDefinition mock."""

    task_id: str = "T001"
    prompt: Prompt = field(default_factory=Prompt)


@dataclass
class DispatchEvent:
    """Minimal claw-eval DispatchEvent mock."""

    latency_ms: float = 42.0
    response_status: int = 200


@dataclass
class ToolSpec:
    """Minimal claw-eval ToolSpec mock for tool definitions."""

    name: str = "bash"
    description: str | None = "Run bash commands"
    input_schema: dict | None = None
    parameters: dict | None = None


# ---------------------------------------------------------------------------
# Helpers to build mock claw_eval module tree
# ---------------------------------------------------------------------------


def _make_module(
    name: str, parent: types.ModuleType | None = None
) -> types.ModuleType:
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    sys.modules[name] = mod
    if parent is not None:
        attr_name = name.rsplit(".", 1)[-1]
        setattr(parent, attr_name, mod)
    return mod


def _install_mock_claw_eval() -> dict[str, types.ModuleType]:
    """Install a complete mock claw_eval module tree into sys.modules.

    Returns a dict mapping dotted module name to the module object so tests
    can inspect / mutate the fake modules easily.
    """
    mods: dict[str, types.ModuleType] = {}

    # ---- top-level ----
    claw_eval = _make_module("claw_eval")
    mods["claw_eval"] = claw_eval

    # ---- claw_eval.cli ----
    cli = _make_module("claw_eval.cli", claw_eval)

    def cmd_run(*args, **kwargs):
        return "run_result"

    def cmd_batch(*args, **kwargs):
        return "batch_result"

    def _run_single_task(*args, **kwargs):
        return {"task_id": "T001", "score": 1.0}

    cli.cmd_run = cmd_run
    cli.cmd_batch = cmd_batch
    cli._run_single_task = _run_single_task
    mods["claw_eval.cli"] = cli

    # ---- claw_eval.runner ----
    runner = _make_module("claw_eval.runner", claw_eval)
    mods["claw_eval.runner"] = runner

    # ---- claw_eval.runner.loop ----
    loop = _make_module("claw_eval.runner.loop", runner)

    def run_task(task, provider, *args, **kwargs):
        # Simulate a minimal agent loop: call provider.chat once
        msgs = [
            Message(
                role="system",
                content=[
                    ContentBlock(
                        type="text", text="You are a helpful assistant."
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlock(
                        type="text",
                        text=getattr(
                            getattr(task, "prompt", None), "text", ""
                        ),
                    )
                ],
            ),
        ]
        response, usage = provider.chat(msgs)
        return {"task_id": getattr(task, "task_id", ""), "response": response}

    loop.run_task = run_task
    mods["claw_eval.runner.loop"] = loop

    # ---- claw_eval.runner.providers ----
    providers = _make_module("claw_eval.runner.providers", runner)
    mods["claw_eval.runner.providers"] = providers

    # ---- claw_eval.runner.providers.openai_compat ----
    openai_compat = _make_module(
        "claw_eval.runner.providers.openai_compat", providers
    )

    class OpenAICompatProvider:
        model_id = "gpt-4o"

        def chat(self, messages, *args, **kwargs):
            response = Message(
                role="assistant",
                content=[
                    ContentBlock(type="text", text="Hello from the model")
                ],
            )
            usage = Usage(input_tokens=100, output_tokens=50)
            return response, usage

    openai_compat.OpenAICompatProvider = OpenAICompatProvider
    mods["claw_eval.runner.providers.openai_compat"] = openai_compat

    # ---- claw_eval.runner.compact ----
    compact = _make_module("claw_eval.runner.compact", runner)

    def do_auto_compact(*args, **kwargs):
        return "compacted"

    compact.do_auto_compact = do_auto_compact
    mods["claw_eval.runner.compact"] = compact

    # ---- claw_eval.runner.dispatcher ----
    dispatcher = _make_module("claw_eval.runner.dispatcher", runner)

    class ToolDispatcher:
        def dispatch(self, tool_use, *args, **kwargs):
            result = ToolResultBlock(
                content=[ContentBlock(type="text", text="tool output")]
            )
            event = DispatchEvent(latency_ms=42.0, response_status=200)
            return result, event

    dispatcher.ToolDispatcher = ToolDispatcher
    mods["claw_eval.runner.dispatcher"] = dispatcher

    # ---- claw_eval.runner.sandbox_dispatcher ----
    sandbox_dispatcher = _make_module(
        "claw_eval.runner.sandbox_dispatcher", runner
    )

    class SandboxToolDispatcher:
        _http = True  # presence signals sandbox dispatcher
        _sandbox_url = "http://sandbox:8080"

        def dispatch(self, tool_use, *args, **kwargs):
            result = ToolResultBlock(
                content=[ContentBlock(type="text", text="sandbox output")]
            )
            event = DispatchEvent(latency_ms=55.0, response_status=200)
            return result, event

    sandbox_dispatcher.SandboxToolDispatcher = SandboxToolDispatcher
    mods["claw_eval.runner.sandbox_dispatcher"] = sandbox_dispatcher

    # ---- claw_eval.graders ----
    graders = _make_module("claw_eval.graders", claw_eval)
    mods["claw_eval.graders"] = graders

    # ---- claw_eval.graders.llm_judge ----
    llm_judge = _make_module("claw_eval.graders.llm_judge", graders)

    class LLMJudge:
        def evaluate(self, *args, **kwargs):
            return {"score": 0.9}

        def evaluate_actions(self, *args, **kwargs):
            return {"score": 0.8}

        def evaluate_visual(self, *args, **kwargs):
            return {"score": 0.7}

    llm_judge.LLMJudge = LLMJudge
    mods["claw_eval.graders.llm_judge"] = llm_judge

    # ---- claw_eval.graders.registry ----
    registry = _make_module("claw_eval.graders.registry", graders)

    class _DummyGrader:
        """Grader returned by get_grader that may have eval methods."""

        def _llm_score_classifications(self, *args, **kwargs):
            return {"classifications": []}

    def get_grader(*args, **kwargs):
        return _DummyGrader()

    registry.get_grader = get_grader
    registry._DummyGrader = _DummyGrader
    mods["claw_eval.graders.registry"] = registry

    # ---- claw_eval.graders.base ----
    base = _make_module("claw_eval.graders.base", graders)

    class _PeerGraderBase:
        """Peer grader class returned by load_peer_grader."""

        def _llm_score_classifications(self, *args, **kwargs):
            return {"peer_classifications": []}

    def load_peer_grader(*args, **kwargs):
        return _PeerGraderBase

    base.load_peer_grader = load_peer_grader
    base._PeerGraderBase = _PeerGraderBase
    mods["claw_eval.graders.base"] = base

    return mods


# ---------------------------------------------------------------------------
# Module-scoped mock install (happens once per test session)
# ---------------------------------------------------------------------------

# Collect names of all mock modules we inject so cleanup can be surgical.
_MOCK_MODULE_NAMES: list[str] = []


def pytest_configure(config):
    """Install mock claw_eval modules before collection."""
    mods = _install_mock_claw_eval()
    _MOCK_MODULE_NAMES.extend(mods.keys())


def pytest_unconfigure(config):
    """Remove mock claw_eval modules so they don't leak into other tests."""
    for name in _MOCK_MODULE_NAMES:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    """Create an in-memory span exporter."""
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.shutdown()


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    """Create a tracer provider with the in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider
    provider.shutdown()


@pytest.fixture(scope="function")
def instrument(tracer_provider):
    """Instrument claw-eval, yield the instrumentor, then uninstrument."""
    from opentelemetry.instrumentation.claw_eval import ClawEvalInstrumentor

    instrumentor = ClawEvalInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    yield instrumentor
    instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Re-export domain objects so test modules can import from conftest
# ---------------------------------------------------------------------------

__all__ = [
    "ContentBlock",
    "DispatchEvent",
    "Message",
    "Prompt",
    "TaskDefinition",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUse",
    "Usage",
]
