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

"""Test configuration for WebArena instrumentation tests.

Injects lightweight stub modules for WebArena's flat package layout
(``browser_env``, ``agent``, ``llms``, etc.) into ``sys.modules``
so that ``wrapt.wrap_function_wrapper`` can resolve them without
installing the real WebArena framework.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from enum import Enum
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
# Stub types that mimic WebArena's data structures
# ---------------------------------------------------------------------------


class ActionTypes(Enum):
    """Minimal reproduction of ``browser_env.actions.ActionTypes``."""

    CLICK = 0
    TYPE = 1
    HOVER = 2
    SCROLL = 3
    GOTO = 4
    GO_BACK = 5
    GO_FORWARD = 6
    STOP = 7
    NONE = 8


class _FakePage:
    """Stub for Playwright page object used by ScriptBrowserEnv."""

    def __init__(self, url: str = "http://example.com") -> None:
        self.url = url


class ScriptBrowserEnv:
    """Stub for ``browser_env.envs.ScriptBrowserEnv``."""

    def __init__(self) -> None:
        self.page = _FakePage()
        self.main_observation_type = "accessibility_tree"

    def reset(
        self, *, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Simulate env.reset returning (obs, info)."""
        return ({"text": "Initial observation"}, {"page": self.page})

    def step(
        self, action: dict[str, Any]
    ) -> tuple[str, bool, bool, bool, dict[str, Any]]:
        """Simulate env.step returning (obs, reward, terminated, truncated, info)."""
        atype = action.get("action_type")
        terminated = False
        if isinstance(atype, ActionTypes):
            terminated = atype == ActionTypes.STOP
        return ("obs_text", False, terminated, False, {"fail_error": ""})

    def close(self) -> None:
        """Simulate env.close."""
        pass


class _FakePromptConstructor:
    """Stub for ``agent.prompts.prompt_constructor`` classes."""

    instruction_path = Path("/fake/instructions.json")
    instruction = {"intro": "You are a web browsing agent."}
    obs_modality = "text"

    def construct(
        self,
        trajectory: list[dict[str, Any]] | None = None,
        intent: str = "",
        meta_data: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"Do: {intent}"}]


class DirectPromptConstructor(_FakePromptConstructor):
    pass


class CoTPromptConstructor(_FakePromptConstructor):
    pass


@dataclass
class _FakeLMConfig:
    provider: str = "openai"
    model: str = "gpt-4"


class PromptAgent:
    """Stub for ``agent.agent.PromptAgent``."""

    def __init__(self) -> None:
        self.prompt_constructor = _FakePromptConstructor()
        self.lm_config = _FakeLMConfig()

    def next_action(
        self,
        obs: str,
        intent: str = "",
        meta_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a fake action dict."""
        return {
            "action_type": ActionTypes.CLICK,
            "element_id": "42",
            "raw_prediction": "click [42]",
        }


def construct_agent(args: Any) -> PromptAgent:
    """Stub for ``agent.agent.construct_agent``."""
    return PromptAgent()


def generate_from_huggingface_completion(
    prompt: str = "",
    model_endpoint: str = "http://hf-endpoint",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
    stop_sequences: list[str] | None = None,
) -> str:
    """Stub for ``llms.providers.hf_utils.generate_from_huggingface_completion``."""
    return "Generated text response"


# ---------------------------------------------------------------------------
# Inject stub modules into sys.modules
# ---------------------------------------------------------------------------


def _inject_stub_modules() -> None:
    """Build the module tree that WebArena would normally install.

    Idempotent: if the stubs are already present in ``sys.modules``,
    this function is a no-op so that re-imports of conftest (e.g. from
    test modules that ``from conftest import ...``) do not replace the
    modules that ``wrapt`` has already patched.
    """
    if "browser_env.envs" in sys.modules and hasattr(
        sys.modules["browser_env.envs"], "ScriptBrowserEnv"
    ):
        return

    # --- browser_env ---
    browser_env_mod = types.ModuleType("browser_env")
    browser_env_envs_mod = types.ModuleType("browser_env.envs")
    browser_env_actions_mod = types.ModuleType("browser_env.actions")

    browser_env_envs_mod.ScriptBrowserEnv = ScriptBrowserEnv
    browser_env_actions_mod.ActionTypes = ActionTypes
    browser_env_mod.envs = browser_env_envs_mod
    browser_env_mod.actions = browser_env_actions_mod

    sys.modules["browser_env"] = browser_env_mod
    sys.modules["browser_env.envs"] = browser_env_envs_mod
    sys.modules["browser_env.actions"] = browser_env_actions_mod

    # --- agent ---
    agent_mod = types.ModuleType("agent")
    agent_agent_mod = types.ModuleType("agent.agent")
    agent_prompts_mod = types.ModuleType("agent.prompts")
    agent_prompts_pc_mod = types.ModuleType("agent.prompts.prompt_constructor")

    agent_agent_mod.PromptAgent = PromptAgent
    agent_agent_mod.construct_agent = construct_agent
    agent_mod.construct_agent = construct_agent
    agent_mod.agent = agent_agent_mod

    agent_prompts_pc_mod.DirectPromptConstructor = DirectPromptConstructor
    agent_prompts_pc_mod.CoTPromptConstructor = CoTPromptConstructor
    agent_prompts_mod.prompt_constructor = agent_prompts_pc_mod

    sys.modules["agent"] = agent_mod
    sys.modules["agent.agent"] = agent_agent_mod
    sys.modules["agent.prompts"] = agent_prompts_mod
    sys.modules["agent.prompts.prompt_constructor"] = agent_prompts_pc_mod

    # --- llms ---
    llms_mod = types.ModuleType("llms")
    llms_providers_mod = types.ModuleType("llms.providers")
    llms_providers_hf_mod = types.ModuleType("llms.providers.hf_utils")
    llms_providers_hf_mod.generate_from_huggingface_completion = (
        generate_from_huggingface_completion
    )
    llms_providers_mod.hf_utils = llms_providers_hf_mod
    llms_mod.providers = llms_providers_mod

    sys.modules["llms"] = llms_mod
    sys.modules["llms.providers"] = llms_providers_mod
    sys.modules["llms.providers.hf_utils"] = llms_providers_hf_mod


# Inject before any instrumentation import
_inject_stub_modules()

# Clear any cached instrumentation imports so they pick up fresh stubs
for _m in list(sys.modules):
    if _m.startswith("opentelemetry.instrumentation.webarena"):
        del sys.modules[_m]


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


# ---------------------------------------------------------------------------
# OTel test fixtures
# ---------------------------------------------------------------------------

from opentelemetry.instrumentation.webarena import WebarenaInstrumentor
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
    """Instrument WebArena, yield the instrumentor, then uninstrument."""
    instrumentor = WebarenaInstrumentor()
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
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    instrumentor = WebarenaInstrumentor()
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


def make_action(
    action_type: ActionTypes = ActionTypes.CLICK,
    element_id: str = "42",
    raw_prediction: str = "click [42]",
    **extra: Any,
) -> dict[str, Any]:
    """Build a WebArena action dict for test convenience."""
    d: dict[str, Any] = {
        "action_type": action_type,
        "element_id": element_id,
        "raw_prediction": raw_prediction,
    }
    d.update(extra)
    return d


def make_ns_args(
    agent_type: str = "prompt",
    provider: str = "openai",
    model: str = "gpt-4",
    instruction_path: str = "/fake/p_cot_id_actree_2s.json",
    action_set_tag: str = "id_accessibility_tree",
    observation_type: str = "accessibility_tree",
) -> types.SimpleNamespace:
    """Build a namespace object mimicking argparse output for construct_agent."""
    return types.SimpleNamespace(
        agent_type=agent_type,
        provider=provider,
        model=model,
        instruction_path=instruction_path,
        action_set_tag=action_set_tag,
        observation_type=observation_type,
    )


def spans_by_name(spans, name: str):
    """Filter finished spans by name substring."""
    return [s for s in spans if name in s.name]


def span_attr(span, key: str) -> Any:
    """Safely read an attribute from a finished span."""
    return span.attributes.get(key)
