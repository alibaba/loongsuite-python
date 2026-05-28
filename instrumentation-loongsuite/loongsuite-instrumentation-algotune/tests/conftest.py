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

"""Test configuration for AlgoTune instrumentation tests.

Injects lightweight stub modules for ``AlgoTuner.*`` into ``sys.modules``
so that ``wrap_function_wrapper`` can resolve them without installing the
real AlgoTune package.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure workspace opentelemetry sources are importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTIL_GENAI_SRC = _REPO_ROOT / "util" / "opentelemetry-util-genai" / "src"
if _UTIL_GENAI_SRC.is_dir() and str(_UTIL_GENAI_SRC) not in sys.path:
    sys.path.insert(0, str(_UTIL_GENAI_SRC))
    for _m in list(sys.modules):
        if _m == "opentelemetry.util.genai" or _m.startswith(
            "opentelemetry.util.genai."
        ):
            del sys.modules[_m]

_ALGOTUNE_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
if _ALGOTUNE_PLUGIN_SRC.is_dir() and str(_ALGOTUNE_PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(_ALGOTUNE_PLUGIN_SRC))


import pytest

# ---------------------------------------------------------------------------
# Stub modules for AlgoTuner
# ---------------------------------------------------------------------------


def _make_main_function():
    """Create a stub ``AlgoTuner.main.main`` function."""

    def main(*args: Any, **kwargs: Any) -> str:
        return "main_result"

    return main


class LLMInterface:
    """Stub for ``AlgoTuner.interfaces.llm_interface.LLMInterface``."""

    def __init__(self, model_name: str = "openai/gpt-4o"):
        self.model_name = model_name
        self.state = types.SimpleNamespace(messages=[], spend=0.0)
        self._final_eval_success = False
        self._final_eval_metrics = {}

    def run_task(self, *args: Any, **kwargs: Any) -> str:
        return "task_done"

    def get_response(self, *args: Any, **kwargs: Any) -> dict | None:
        return {"content": "response text"}

    def handle_function_call(self, *args: Any, **kwargs: Any) -> dict:
        return {"command": "edit", "success": True}

    def check_limits(self) -> bool:
        return False


class CommandHandlers:
    """Stub for ``AlgoTuner.interfaces.commands.handlers.CommandHandlers``."""

    def __init__(self):
        self.interface = types.SimpleNamespace(max_samples=None)

    def handle_command(
        self, command_str: Any, *args: Any, **kwargs: Any
    ) -> dict:
        return {"success": True, "message": "ok"}

    def _runner_eval_dataset(
        self,
        data_subset: str = "",
        command_source: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return types.SimpleNamespace(
            success=True,
            status="ok",
            message="evaluated",
            data={
                "num_evaluated": 10,
                "mean_speedup": 1.5,
                "num_valid": 8,
                "num_invalid": 1,
                "num_timeout": 1,
            },
        )


class EvaluationOrchestrator:
    """Stub for ``AlgoTuner.utils.evaluator.evaluation_orchestrator.EvaluationOrchestrator``."""

    def evaluate_single(
        self,
        *args: Any,
        problem_id: str = "problem_1",
        problem_index: int = 0,
        baseline_time_ms: float | None = None,
        **kwargs: Any,
    ) -> Any:
        return types.SimpleNamespace(
            speedup=2.0,
            solver_time_ms=150.0,
            is_valid=True,
            execution=types.SimpleNamespace(
                timeout_occurred=False, error_type=None
            ),
        )


class BaselineManager:
    """Stub for ``AlgoTuner.utils.evaluator.baseline_manager.BaselineManager``."""

    def __init__(self):
        self._cache: dict = {}

    def get_baseline_times(
        self, subset: str = "", *args: Any, **kwargs: Any
    ) -> dict:
        return {"problem_1": 100.0, "problem_2": 200.0}


class LiteLLMModel:
    """Stub for ``AlgoTuner.models.lite_llm_model.LiteLLMModel``."""

    def __init__(self, model_name: str = "openai/gpt-4o"):
        self.model_name = model_name

    def query(self, *args: Any, **kwargs: Any) -> str:
        return "llm_response"

    def _execute_query(self, *args: Any, **kwargs: Any) -> str:
        return "executed"


class TogetherModel:
    """Stub for ``AlgoTuner.models.together_model.TogetherModel``."""

    def __init__(self, model_name: str = "together/model-x"):
        self.model_name = model_name
        self.default_params: dict = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024,
        }

    def query(self, *args: Any, **kwargs: Any) -> dict:
        return {
            "message": "together response",
            "cost": 0.01,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }


def _inject_stub_modules() -> None:
    """Register stub modules in ``sys.modules`` so ``wrapt`` can resolve them."""

    # Top-level AlgoTuner package
    algotune_mod = types.ModuleType("AlgoTuner")

    # AlgoTuner.main
    algotune_main_mod = types.ModuleType("AlgoTuner.main")
    algotune_main_mod.main = _make_main_function()

    # AlgoTuner.interfaces
    algotune_interfaces_mod = types.ModuleType("AlgoTuner.interfaces")

    # AlgoTuner.interfaces.llm_interface
    algotune_llm_mod = types.ModuleType("AlgoTuner.interfaces.llm_interface")
    algotune_llm_mod.LLMInterface = LLMInterface

    # AlgoTuner.interfaces.commands
    algotune_commands_mod = types.ModuleType("AlgoTuner.interfaces.commands")

    # AlgoTuner.interfaces.commands.handlers
    algotune_handlers_mod = types.ModuleType(
        "AlgoTuner.interfaces.commands.handlers"
    )
    algotune_handlers_mod.CommandHandlers = CommandHandlers

    # AlgoTuner.utils
    algotune_utils_mod = types.ModuleType("AlgoTuner.utils")

    # AlgoTuner.utils.evaluator
    algotune_evaluator_mod = types.ModuleType("AlgoTuner.utils.evaluator")

    # AlgoTuner.utils.evaluator.evaluation_orchestrator
    algotune_eval_orch_mod = types.ModuleType(
        "AlgoTuner.utils.evaluator.evaluation_orchestrator"
    )
    algotune_eval_orch_mod.EvaluationOrchestrator = EvaluationOrchestrator

    # AlgoTuner.utils.evaluator.baseline_manager
    algotune_baseline_mod = types.ModuleType(
        "AlgoTuner.utils.evaluator.baseline_manager"
    )
    algotune_baseline_mod.BaselineManager = BaselineManager

    # AlgoTuner.models
    algotune_models_mod = types.ModuleType("AlgoTuner.models")

    # AlgoTuner.models.lite_llm_model
    algotune_litellm_mod = types.ModuleType("AlgoTuner.models.lite_llm_model")
    algotune_litellm_mod.LiteLLMModel = LiteLLMModel

    # AlgoTuner.models.together_model
    algotune_together_mod = types.ModuleType("AlgoTuner.models.together_model")
    algotune_together_mod.TogetherModel = TogetherModel

    # Wire parent references
    algotune_mod.main = algotune_main_mod
    algotune_mod.interfaces = algotune_interfaces_mod
    algotune_mod.utils = algotune_utils_mod
    algotune_mod.models = algotune_models_mod

    algotune_interfaces_mod.llm_interface = algotune_llm_mod
    algotune_interfaces_mod.commands = algotune_commands_mod
    algotune_commands_mod.handlers = algotune_handlers_mod

    algotune_utils_mod.evaluator = algotune_evaluator_mod
    algotune_evaluator_mod.evaluation_orchestrator = algotune_eval_orch_mod
    algotune_evaluator_mod.baseline_manager = algotune_baseline_mod

    algotune_models_mod.lite_llm_model = algotune_litellm_mod
    algotune_models_mod.together_model = algotune_together_mod

    # Register every module in sys.modules
    sys.modules["AlgoTuner"] = algotune_mod
    sys.modules["AlgoTuner.main"] = algotune_main_mod
    sys.modules["AlgoTuner.interfaces"] = algotune_interfaces_mod
    sys.modules["AlgoTuner.interfaces.llm_interface"] = algotune_llm_mod
    sys.modules["AlgoTuner.interfaces.commands"] = algotune_commands_mod
    sys.modules["AlgoTuner.interfaces.commands.handlers"] = (
        algotune_handlers_mod
    )
    sys.modules["AlgoTuner.utils"] = algotune_utils_mod
    sys.modules["AlgoTuner.utils.evaluator"] = algotune_evaluator_mod
    sys.modules["AlgoTuner.utils.evaluator.evaluation_orchestrator"] = (
        algotune_eval_orch_mod
    )
    sys.modules["AlgoTuner.utils.evaluator.baseline_manager"] = (
        algotune_baseline_mod
    )
    sys.modules["AlgoTuner.models"] = algotune_models_mod
    sys.modules["AlgoTuner.models.lite_llm_model"] = algotune_litellm_mod
    sys.modules["AlgoTuner.models.together_model"] = algotune_together_mod


# Inject stubs before any test imports the instrumentation module.
_inject_stub_modules()


# ---------------------------------------------------------------------------
# OTel test fixtures
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


# Flush cached instrumentation module state so the stubs take effect.
for _m in list(sys.modules):
    if _m.startswith("opentelemetry.instrumentation.algotune"):
        del sys.modules[_m]

from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor
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
def instrument(tracer_provider):
    """Instrument, yield, then uninstrument.

    Uses ``skip_dep_check=True`` because we use stub modules.
    """
    instrumentor = AlgoTuneInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
