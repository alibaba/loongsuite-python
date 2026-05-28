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

"""Shared fixtures for mini-swe-agent instrumentation tests.

Creates stub ``minisweagent`` packages in ``sys.modules`` so that the
instrumentation code can be imported without having the real
``mini-swe-agent`` package installed.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Environment ──────────────────────────────────────────────────────
# Must be set before any OpenTelemetry semconv import.
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


# ── Helpers ──────────────────────────────────────────────────────────


def _make_module(
    name: str, parent: types.ModuleType | None = None
) -> types.ModuleType:
    """Create a stub module and register it in ``sys.modules``."""
    mod = types.ModuleType(name)
    mod.__package__ = name
    sys.modules[name] = mod
    if parent is not None:
        attr = name.rsplit(".", 1)[-1]
        setattr(parent, attr, mod)
    return mod


# ── Stub minisweagent package tree ───────────────────────────────────


class _StubConfig:
    """Mimics ``minisweagent`` agent/model config objects."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        step_limit: int = 0,
        cost_limit: float = 0,
    ):
        self.model_name = model_name
        self.step_limit = step_limit
        self.cost_limit = cost_limit


class _StubModel:
    """Mimics ``minisweagent`` model with a config attribute."""

    def __init__(self, config: _StubConfig | None = None):
        self.config = config or _StubConfig()


class _StubDefaultAgent:
    """Minimal duck-type of ``minisweagent.agents.default.DefaultAgent``."""

    def __init__(
        self,
        messages: list[dict[str, Any]] | None = None,
        model: _StubModel | None = None,
        config: _StubConfig | None = None,
    ):
        self.messages: list[dict[str, Any]] = messages or []
        self.model = model or _StubModel()
        self.config = config or _StubConfig()
        self.n_calls: int = 0
        self.cost: float = 0.0

    def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        return {"exit_status": "submitted", "submission": "done"}

    def step(self) -> None:
        pass


class _StubEnvironment:
    """Mimics ``minisweagent.environments.Environment``."""

    def execute(
        self, action: dict, cwd: str = "", **kwargs: Any
    ) -> dict[str, Any]:  # noqa: ARG002
        return {"output": "ok", "exit_code": 0}


class _InterruptAgentFlow(Exception):
    """Mimics ``minisweagent.exceptions.InterruptAgentFlow``."""


# ── BASH_TOOL constant ──────────────────────────────────────────────

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


# ── Fixture: inject stubs into sys.modules ───────────────────────────


@pytest.fixture(autouse=True)
def _patch_minisweagent_modules():
    """Register fake ``minisweagent`` modules before every test, and
    clean them up afterwards so tests are isolated."""
    # Snapshot keys so we only remove what we added.
    keys_before = set(sys.modules.keys())

    # Root package
    mini = _make_module("minisweagent")

    # minisweagent.agents / .agents.default
    agents = _make_module("minisweagent.agents", parent=mini)
    agents_default = _make_module("minisweagent.agents.default", parent=agents)
    agents_default.DefaultAgent = _StubDefaultAgent  # type: ignore[attr-defined]

    # minisweagent.environments
    envs = _make_module("minisweagent.environments", parent=mini)
    envs.get_environment = MagicMock(return_value=_StubEnvironment())  # type: ignore[attr-defined]
    envs.Environment = _StubEnvironment  # type: ignore[attr-defined]

    # minisweagent.models / .models.utils / .models.utils.actions_toolcall
    models = _make_module("minisweagent.models", parent=mini)
    models_utils = _make_module("minisweagent.models.utils", parent=models)
    actions_tc = _make_module(
        "minisweagent.models.utils.actions_toolcall", parent=models_utils
    )
    actions_tc.BASH_TOOL = BASH_TOOL  # type: ignore[attr-defined]

    # minisweagent.exceptions
    exceptions = _make_module("minisweagent.exceptions", parent=mini)
    exceptions.InterruptAgentFlow = _InterruptAgentFlow  # type: ignore[attr-defined]

    # minisweagent.run / .run.mini
    run_pkg = _make_module("minisweagent.run", parent=mini)
    run_mini = _make_module("minisweagent.run.mini", parent=run_pkg)
    run_mini.app = MagicMock(name="typer_app")  # type: ignore[attr-defined]
    run_mini.get_environment = MagicMock()  # type: ignore[attr-defined]

    yield

    # Teardown: remove only the keys we added.
    added = set(sys.modules.keys()) - keys_before
    for key in added:
        sys.modules.pop(key, None)

    # Also reset the instrumentor class-level cached original
    try:
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        MiniSweAgentInstrumentor._original_get_environment = None
    except Exception:
        pass


# ── Re-export helpers so tests can use them directly ─────────────────


@pytest.fixture()
def stub_agent():
    """Return a fresh ``_StubDefaultAgent``."""
    return _StubDefaultAgent(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the bug"},
            {
                "role": "assistant",
                "content": "I'll fix it.",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": "file1.py\nfile2.py",
            },
            {"role": "assistant", "content": "Done."},
        ],
        model=_StubModel(_StubConfig(model_name="gpt-4o")),
    )


@pytest.fixture()
def stub_environment():
    """Return a fresh ``_StubEnvironment``."""
    return _StubEnvironment()
