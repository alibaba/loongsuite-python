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

"""Tests for ``loongsuite-instrumentation-deer-flow``.

These tests verify import wiring, instrument/uninstrument behavior, and the
no-deer-flow-installed silent skip. Full E2E span emission is covered by the
deployment examples under ``${WORKSPACE_ROOT}/example-deploy/deer-flow/``.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

from opentelemetry.instrumentation.deer_flow import DeerFlowInstrumentor
from opentelemetry.instrumentation.deer_flow.package import _instruments
from opentelemetry.instrumentation.deer_flow.version import __version__


def test_package_metadata() -> None:
    assert _instruments == ("deer-flow >= 2.1, < 3.0",)
    assert __version__


def test_import_silent_skip_without_deer_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``import deerflow`` raises ImportError, ``instrument`` is a no-op."""
    monkeypatch.setitem(sys.modules, "deerflow", None)
    instrumentor = DeerFlowInstrumentor()
    # Should not raise.
    instrumentor.instrument()
    assert instrumentor._handler is None
    assert instrumentor._targets == []


def test_instrument_and_uninstrument_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub out the 7 DeerFlow public APIs and verify wrap/unwrap round-trip."""
    worker = types.ModuleType("deerflow.runtime.runs.worker")
    subagents_executor = types.ModuleType("deerflow.subagents.executor")
    task_tool_mod = types.ModuleType("deerflow.tools.builtins.task_tool")
    sandbox_mod = types.ModuleType("deerflow.sandbox.sandbox")
    sandbox_provider_mod = types.ModuleType("deerflow.sandbox.sandbox_provider")
    storage_mod = types.ModuleType("deerflow.agents.memory.storage")
    updater_mod = types.ModuleType("deerflow.agents.memory.updater")

    class RunAgent:
        async def run_agent(self, *a: Any, **k: Any) -> str:
            return "ok"

    class SubagentExecutor:
        async def _aexecute(self, task: str) -> str:
            return f"done:{task}"

    async def task_tool(*a: Any, **k: Any) -> str:
        return "task_done"

    class Sandbox:
        def execute_command(self, command: str) -> str:
            return f"out:{command}"

        def read_file(self, path: str) -> str:
            return "file"

        def write_file(self, path: str, content: str, append: bool = False) -> None:
            return None

        def glob(self, path: str, pattern: str) -> tuple[list[str], bool]:
            return [], False

        def grep(self, *a: Any, **k: Any) -> str:
            return ""

        def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
            return []

    class SandboxProvider:
        def acquire(self, thread_id: str | None = None) -> str:
            return "sid"

        async def acquire_async(self, thread_id: str | None = None) -> str:
            return "sid"

        def release(self, sandbox_id: str) -> None:
            return None

    class FileMemoryStorage:
        def load(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict:
            return {}

        def save(self, memory_data: dict, agent_name: str | None = None, *, user_id: str | None = None) -> bool:
            return True

    class MemoryUpdater:
        async def aupdate_memory(self, messages: list, thread_id: str | None = None) -> bool:
            return True

    worker.run_agent = RunAgent.run_agent  # type: ignore[attr-defined]
    subagents_executor.SubagentExecutor = SubagentExecutor  # type: ignore[attr-defined]
    task_tool_mod.task_tool = task_tool  # type: ignore[attr-defined]
    sandbox_mod.Sandbox = Sandbox  # type: ignore[attr-defined]
    sandbox_provider_mod.SandboxProvider = SandboxProvider  # type: ignore[attr-defined]
    storage_mod.FileMemoryStorage = FileMemoryStorage  # type: ignore[attr-defined]
    updater_mod.MemoryUpdater = MemoryUpdater  # type: ignore[attr-defined]

    for name, mod in [
        ("deerflow", types.ModuleType("deerflow")),
        ("deerflow.runtime", types.ModuleType("deerflow.runtime")),
        ("deerflow.runtime.runs", types.ModuleType("deerflow.runtime.runs")),
        ("deerflow.runtime.runs.worker", worker),
        ("deerflow.subagents", types.ModuleType("deerflow.subagents")),
        ("deerflow.subagents.executor", subagents_executor),
        ("deerflow.tools", types.ModuleType("deerflow.tools")),
        ("deerflow.tools.builtins", types.ModuleType("deerflow.tools.builtins")),
        ("deerflow.tools.builtins.task_tool", task_tool_mod),
        ("deerflow.sandbox", types.ModuleType("deerflow.sandbox")),
        ("deerflow.sandbox.sandbox", sandbox_mod),
        ("deerflow.sandbox.sandbox_provider", sandbox_provider_mod),
        ("deerflow.agents", types.ModuleType("deerflow.agents")),
        ("deerflow.agents.memory", types.ModuleType("deerflow.agents.memory")),
        ("deerflow.agents.memory.storage", storage_mod),
        ("deerflow.agents.memory.updater", updater_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    importlib.invalidate_caches()

    instrumentor = DeerFlowInstrumentor()
    # Call ``_instrument`` directly: ``BaseInstrumentor.instrument`` runs a
    # dependency check against the real ``deer-flow`` distribution, which is
    # not installable in CI (no PyPI release). The stub modules installed
    # above let ``import deerflow`` succeed inside ``_instrument``.
    instrumentor._instrument()
    assert instrumentor._handler is not None
    assert len(instrumentor._targets) > 0

    instrumentor._uninstrument()
    assert instrumentor._handler is None
    assert instrumentor._targets == []
