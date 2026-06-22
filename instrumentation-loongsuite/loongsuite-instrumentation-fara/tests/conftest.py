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

"""Test configuration for Fara instrumentation tests.

Injects lightweight stub modules for Fara's package layout
(``fara.run_fara``, ``fara.fara_agent``) into ``sys.modules`` so that
``wrapt.wrap_function_wrapper`` can resolve them without installing
the real Fara framework.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Ensure our package source is importable
# ---------------------------------------------------------------------------

_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
if _PLUGIN_SRC.is_dir() and str(_PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_SRC))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTIL_GENAI_SRC = _REPO_ROOT / "util" / "opentelemetry-util-genai" / "src"
if _UTIL_GENAI_SRC.is_dir() and str(_UTIL_GENAI_SRC) not in sys.path:
    sys.path.insert(0, str(_UTIL_GENAI_SRC))
    for _m in list(sys.modules):
        if _m == "opentelemetry.util.genai" or _m.startswith(
            "opentelemetry.util.genai."
        ):
            del sys.modules[_m]


# ---------------------------------------------------------------------------
# Stub types that mimic Fara's data structures
# ---------------------------------------------------------------------------


@dataclass
class FunctionCall:
    name: str = "default"
    id: str = "dummy"
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    content: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class FaraAgent:
    """Stub for ``fara.fara_agent.FaraAgent``.

    Implements the public methods exercised by the wrappers:
    ``run``, ``generate_model_call``, ``execute_action``. The
    internals are simplified (no real browser / OpenAI calls).
    """

    def __init__(
        self,
        client_config: dict[str, Any] | None = None,
        max_rounds: int = 3,
        actions: list[str] | None = None,
    ) -> None:
        self.client_config = client_config or {
            "model": "microsoft/Fara-7B",
            "base_url": "http://localhost:5000/v1",
            "api_key": "not-needed",
        }
        self.max_rounds = max_rounds
        # Sequence of action names the stub will return from
        # generate_model_call + execute_action. The last one should be
        # "terminate" to end the loop normally.
        # Copy to avoid sharing the default list across instances.
        self._actions: list[str] = list(
            actions if actions is not None else ["left_click", "type", "terminate"]
        )
        self._chat_history: list[Any] = []

    async def generate_model_call(
        self,
        is_first_round: bool,
        first_screenshot: Any = None,
    ) -> tuple[list[FunctionCall], str]:
        action_name = self._actions.pop(0) if self._actions else "terminate"
        if isinstance(action_name, str) and action_name.startswith("RAISE:"):
            raise RuntimeError(action_name[len("RAISE:") :])
        # EXEC_RAISE:<msg> — generate returns normally, execute_action
        # raises ValueError. Used to test TOOL-span error handling.
        if isinstance(action_name, str) and action_name.startswith("EXEC_RAISE:"):
            action_name = action_name[len("EXEC_RAISE:") :]
            fc = FunctionCall(
                name=action_name,
                id="dummy",
                arguments={"action": action_name, "thoughts": "thinking",
                           "_exec_raise": True},
            )
            raw = f"<action>{{\"action\": \"{action_name}\"}}</action>"
            return [fc], raw
        fc = FunctionCall(
            name=action_name,
            id="dummy",
            arguments={"action": action_name, "thoughts": "thinking"},
        )
        raw = (
            f"<thought>do {action_name}</thought>\n"
            f"<action>{{\"action\": \"{action_name}\"}}</action>"
        )
        return [fc], raw

    async def execute_action(
        self,
        function_call: list[FunctionCall],
    ) -> tuple[bool, bytes, str]:
        fc = function_call[0]
        action = fc.arguments["action"]
        if fc.arguments.get("_exec_raise"):
            raise ValueError(action)
        is_stop = action in {"terminate", "stop"}
        return (is_stop, b"", f"executed {action}")

    async def run(self, user_message: str) -> tuple[str, list[str], list[str]]:
        final_answer = "<no_answer>"
        all_actions: list[str] = []
        all_observations: list[str] = []
        for _ in range(self.max_rounds):
            fc, raw = await self.generate_model_call(True)
            all_actions.append(raw)
            is_stop, _, desc = await self.execute_action(fc)
            all_observations.append(desc)
            if is_stop:
                final_answer = f"done: {user_message}"
                break
        return final_answer, all_actions, all_observations


async def run_fara_agent(
    initial_task: str | None = None,
    endpoint_config: dict[str, str] | None = None,
    start_page: str = "https://www.bing.com/",
    headless: bool = True,
    downloads_folder: str | None = None,
    save_screenshots: bool = True,
    max_rounds: int = 100,
    use_browser_base: bool = False,
    agent_factory: Any = None,
) -> dict[str, Any]:
    """Stub for ``fara.run_fara.run_fara_agent``.

    Returns a dict so callers can assert it was called. If
    ``agent_factory`` is provided, it is called to build the agent and
    ``agent.run(initial_task)`` is awaited so the AGENT/STEP/TOOL spans
    fire underneath the ENTRY span.
    """

    factory = agent_factory or (
        lambda: FaraAgent(
            client_config=endpoint_config
            or {"model": "microsoft/Fara-7B"},
            max_rounds=max_rounds,
        )
    )
    agent = factory()
    final_answer, actions, observations = await agent.run(initial_task or "")
    return {
        "final_answer": final_answer,
        "actions": actions,
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# Inject stub modules into sys.modules
# ---------------------------------------------------------------------------


def _inject_stub_modules() -> None:
    """Build the module tree that Fara would normally install.

    Idempotent: if the stubs are already present in ``sys.modules``,
    this function is a no-op so that re-imports of conftest do not
    replace the modules that ``wrapt`` has already patched.
    """
    if "fara.fara_agent" in sys.modules and hasattr(
        sys.modules["fara.fara_agent"], "FaraAgent"
    ):
        return

    fara_mod = types.ModuleType("fara")
    fara_run_mod = types.ModuleType("fara.run_fara")
    fara_agent_mod = types.ModuleType("fara.fara_agent")

    fara_run_mod.run_fara_agent = run_fara_agent
    fara_agent_mod.FaraAgent = FaraAgent
    fara_agent_mod.FunctionCall = FunctionCall
    fara_agent_mod.ModelResponse = ModelResponse
    fara_agent_mod.FARA_ACTION_DEFINITIONS = {
        "key": {"keys"},
        "type": {"text", "coordinate"},
        "mouse_move": {"coordinate"},
        "left_click": {"coordinate"},
        "scroll": {"coordinate", "pixels"},
        "visit_url": {"url"},
        "web_search": {"query"},
        "history_back": set(),
        "pause_and_memorize_fact": {"fact"},
        "wait": {"time"},
        "terminate": {"status"},
    }

    sys.modules["fara"] = fara_mod
    sys.modules["fara.run_fara"] = fara_run_mod
    sys.modules["fara.fara_agent"] = fara_agent_mod


# Inject before any instrumentation import
_inject_stub_modules()

# Clear any cached instrumentation imports so they pick up fresh stubs
for _m in list(sys.modules):
    if _m.startswith("opentelemetry.instrumentation.fara"):
        del sys.modules[_m]


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


# ---------------------------------------------------------------------------
# OTel test fixtures
# ---------------------------------------------------------------------------

from opentelemetry.instrumentation.fara import FaraInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrument(tracer_provider, span_exporter):
    """Instrument Fara, yield the instrumentor, then uninstrument."""
    instrumentor = FaraInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
    span_exporter.clear()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, span_exporter):
    """Same as ``instrument`` but with message content capture enabled."""
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_ONLY"
    instrumentor = FaraInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
    span_exporter.clear()
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)


# ---------------------------------------------------------------------------
# Helpers available to all test modules
# ---------------------------------------------------------------------------


def spans_by_name(spans, name: str):
    """Filter finished spans by name substring."""
    return [s for s in spans if name in s.name]


def span_attr(span, key: str) -> Any:
    """Safely read an attribute from a finished span."""
    return span.attributes.get(key)
