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

"""Shared fixtures for bfclv4 instrumentation tests.

Sets up mock ``bfcl_eval`` modules via ``sys.modules`` so that wrapper
code that imports from ``bfcl_eval`` can work without the real package.
Also provides a standard OTel TracerProvider with InMemorySpanExporter.
"""

from __future__ import annotations

import enum
import sys
import types

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# ---------------------------------------------------------------------------
# Mock bfcl_eval modules
# ---------------------------------------------------------------------------


class _MockModelStyle(enum.Enum):
    OPENAI_COMPLETIONS = "openai_completions"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    AMAZON = "amazon"
    FIREWORK_AI = "firework_ai"
    WRITER = "writer"
    NOVITA_AI = "novita_ai"
    NEXUS = "nexus"
    GORILLA = "gorilla"
    OSSMODEL = "ossmodel"


class _MockModelConfig:
    def __init__(self, handler_cls):
        self.model_handler = handler_cls


class _MockBaseHandler:
    model_name = "test-model"
    model_style = _MockModelStyle.OPENAI_COMPLETIONS

    def inference(
        self, test_entry, include_input_log=True, exclude_state_log=True
    ):
        return ("result", {}), {}

    def _query_FC(self, *args, **kwargs):
        return "api_response", 0.1

    def _query_prompting(self, *args, **kwargs):
        return "api_response", 0.2

    def add_first_turn_message_FC(self, *args, **kwargs):
        pass

    def add_first_turn_message_prompting(self, *args, **kwargs):
        pass

    def _add_next_turn_user_message_FC(self, *args, **kwargs):
        pass

    def _add_next_turn_user_message_prompting(self, *args, **kwargs):
        pass


def _install_bfcl_mocks():
    """Install mock bfcl_eval modules into sys.modules."""

    mods = {}

    # Top-level
    bfcl_eval = types.ModuleType("bfcl_eval")
    mods["bfcl_eval"] = bfcl_eval

    # bfcl_eval.constants
    constants = types.ModuleType("bfcl_eval.constants")
    mods["bfcl_eval.constants"] = constants

    # bfcl_eval.constants.enums
    enums_mod = types.ModuleType("bfcl_eval.constants.enums")
    enums_mod.ModelStyle = _MockModelStyle
    mods["bfcl_eval.constants.enums"] = enums_mod
    constants.enums = enums_mod

    # bfcl_eval.constants.model_config
    model_config_mod = types.ModuleType("bfcl_eval.constants.model_config")
    model_config_mod.MODEL_CONFIG_MAPPING = {
        "test_model": _MockModelConfig(_MockBaseHandler),
    }
    mods["bfcl_eval.constants.model_config"] = model_config_mod
    constants.model_config = model_config_mod

    # bfcl_eval.constants.executable_backend_config
    exec_cfg = types.ModuleType(
        "bfcl_eval.constants.executable_backend_config"
    )
    exec_cfg.CLASS_FILE_PATH_MAPPING = {}
    mods["bfcl_eval.constants.executable_backend_config"] = exec_cfg
    constants.executable_backend_config = exec_cfg

    # bfcl_eval.model_handler
    model_handler = types.ModuleType("bfcl_eval.model_handler")
    mods["bfcl_eval.model_handler"] = model_handler

    # bfcl_eval.model_handler.base_handler
    base_handler_mod = types.ModuleType("bfcl_eval.model_handler.base_handler")
    base_handler_mod.BaseHandler = _MockBaseHandler

    def _mock_execute_multi_turn_func_call(
        func_call_list,
        initial_config=None,
        involved_classes=None,
        model_name=None,
        test_entry_id=None,
        long_context=False,
        is_eval_run=False,
    ):
        results = [f"result_{i}" for i in range(len(func_call_list))]
        return results, {}

    base_handler_mod.execute_multi_turn_func_call = (
        _mock_execute_multi_turn_func_call
    )
    mods["bfcl_eval.model_handler.base_handler"] = base_handler_mod
    model_handler.base_handler = base_handler_mod

    # bfcl_eval._llm_response_generation
    gen_mod = types.ModuleType("bfcl_eval._llm_response_generation")
    from concurrent.futures import ThreadPoolExecutor

    gen_mod.ThreadPoolExecutor = ThreadPoolExecutor

    def _mock_generate_results(args, model_name, test_cases_total):
        return {"status": "ok"}

    gen_mod.generate_results = _mock_generate_results
    mods["bfcl_eval._llm_response_generation"] = gen_mod
    bfcl_eval._llm_response_generation = gen_mod

    # bfcl_eval.eval_checker
    eval_checker = types.ModuleType("bfcl_eval.eval_checker")
    mods["bfcl_eval.eval_checker"] = eval_checker

    # bfcl_eval.eval_checker.multi_turn_eval
    mt_eval = types.ModuleType("bfcl_eval.eval_checker.multi_turn_eval")
    mods["bfcl_eval.eval_checker.multi_turn_eval"] = mt_eval

    # bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils
    mt_utils = types.ModuleType(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"
    )
    mt_utils.execute_multi_turn_func_call = _mock_execute_multi_turn_func_call
    mods["bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"] = mt_utils

    return mods


@pytest.fixture(autouse=True)
def _bfcl_mocks():
    """Install and clean up mock bfcl_eval modules for every test."""
    mods = _install_bfcl_mocks()
    saved = {}
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    yield mods
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


@pytest.fixture()
def tracer_provider_and_exporter():
    """Return (TracerProvider, InMemorySpanExporter) for test assertions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture()
def reset_handler_singleton():
    """Reset the get_extended_telemetry_handler singleton before/after each test."""
    from opentelemetry.util.genai.extended_handler import (
        get_extended_telemetry_handler,
    )

    old = getattr(get_extended_telemetry_handler, "_default_handler", None)
    get_extended_telemetry_handler._default_handler = None
    yield
    get_extended_telemetry_handler._default_handler = old


@pytest.fixture()
def handler_with_tracer(tracer_provider_and_exporter, reset_handler_singleton):
    """Create a fresh ExtendedTelemetryHandler backed by the in-memory exporter."""
    provider, exporter = tracer_provider_and_exporter
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
        get_extended_telemetry_handler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=provider)
    get_extended_telemetry_handler._default_handler = handler
    return handler, exporter
