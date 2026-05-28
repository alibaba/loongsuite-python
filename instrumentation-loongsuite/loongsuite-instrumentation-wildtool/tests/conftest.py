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

"""Test configuration for WildToolBench instrumentation tests.

Installs a complete mock ``wtb`` module hierarchy into ``sys.modules`` so
that test modules can ``from wtb.model_handler.base_handler import BaseHandler``
without the real ``wtb`` package being installed.
"""

from __future__ import annotations

import json
import os
import sys
import types
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Environment setup -- must happen before any OTel semconv import
# ---------------------------------------------------------------------------

os.environ.setdefault("OPENAI_API_KEY", "test_key_not_real")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
)


# ---------------------------------------------------------------------------
# Helpers to build mock module tree
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


# ---------------------------------------------------------------------------
# Mock wtb module tree
# ---------------------------------------------------------------------------

_MOCK_MODULE_NAMES: list[str] = []


class BaseHandler:
    """Minimal stub of wtb.model_handler.base_handler.BaseHandler.

    Provides the multi-turn inference orchestration that the real WildToolBench
    framework implements.  Only the methods needed by the instrumentation
    wrappers and the existing test stubs are implemented.
    """

    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    # -- abstract methods that concrete subclasses must override -------------

    def _request_tool_call(self, inference_data: dict) -> tuple:
        raise NotImplementedError

    def _parse_api_response(self, api_response: Any) -> dict:
        raise NotImplementedError

    # -- high-level orchestration -------------------------------------------

    def inference(self, test_entry: dict):
        """Default implementation delegates to inference_multi_turn."""
        return self.inference_multi_turn(test_entry)

    def inference_multi_turn(self, test_entry: dict):
        """Run one multi-turn evaluation.  Returns a list of per-task results."""
        answer_lists = test_entry.get("english_answer_list", [])
        tasks = test_entry.get("english_tasks", [])
        tools = test_entry.get("english_tools", [])

        results = []
        for task_idx, answer_steps in enumerate(answer_lists):
            task_text = tasks[task_idx] if task_idx < len(tasks) else ""
            inference_data = {
                "task_idx": task_idx,
                "test_entry_id": test_entry.get("id", ""),
                "messages": [{"role": "user", "content": task_text}],
                "tools": tools,
                "answer_list": answer_steps,
            }
            result = self.inference_and_eval_multi_step(inference_data)
            results.append(result)
        return results

    def inference_and_eval_multi_step(self, inference_data: dict) -> dict:
        """Execute one task: loop through expected answer steps, calling the
        model (via ``_request_tool_call`` / ``_parse_api_response``) for each,
        and comparing against the expected answer list.
        """
        answer_list = inference_data.get("answer_list", [])
        messages = inference_data.get("messages", [])
        inference_log: dict[str, Any] = {}
        input_token_count: list[int] = []
        output_token_count: list[int] = []

        action_name_label = "correct"
        is_optimal = True

        # Collect all candidate action names for mismatch detection.
        candidate_names = set()
        for step in answer_list:
            action = step.get("action", {})
            name = action.get("name")
            if name:
                candidate_names.add(name)

        for step_idx, expected_step in enumerate(answer_list):
            expected_action = expected_step.get("action", {}).get("name", "")

            # --- call model ---
            response, latency = self._request_tool_call(inference_data)
            parsed = self._parse_api_response(response)

            input_token_count.append(parsed.get("input_token", 0))
            output_token_count.append(parsed.get("output_token", 0))

            tool_calls = parsed.get("tool_calls")
            content = parsed.get("content")

            step_output: dict[str, Any] = {}
            step_data: dict[str, Any] = {"inference_output": step_output}

            if tool_calls:
                step_output["tool_calls"] = tool_calls
                tool_name = tool_calls[0]["function"]["name"]
                tool_call_id = tool_calls[0].get("id")

                if tool_name == expected_action:
                    step_output["current_action_name_label"] = "correct"
                    observation = expected_step.get("observation", "")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": observation,
                        }
                    )
                    step_data["inference_answer"] = {
                        "candidate_0_answer_function_list": {
                            "observation": observation,
                        }
                    }
                elif tool_name not in candidate_names:
                    step_output["current_action_name_label"] = "error"
                    step_output["error_reason"] = (
                        f"action name not in candidate: {tool_name}"
                    )
                    action_name_label = "error"
                    is_optimal = False
                    inference_log[f"step_{step_idx}"] = step_data
                    break
                else:
                    # Tool name is in candidates but not the expected one
                    step_output["current_action_name_label"] = "error"
                    step_output["error_reason"] = (
                        f"action name not in candidate: {tool_name}"
                    )
                    action_name_label = "error"
                    is_optimal = False
                    inference_log[f"step_{step_idx}"] = step_data
                    break

            elif content:
                step_output["content"] = content
                if expected_action == "prepare_to_answer":
                    step_output["current_action_name_label"] = "correct"
                else:
                    step_output["current_action_name_label"] = "error"
                    step_output["error_reason"] = (
                        "unexpected text response when tool call expected"
                    )
                    action_name_label = "error"
                    is_optimal = False
                    inference_log[f"step_{step_idx}"] = step_data
                    break

            else:
                # Neither tool_calls nor content
                step_output["current_action_name_label"] = "error"
                step_output["error_reason"] = "tool_calls and content are None"
                action_name_label = "error"
                is_optimal = False
                inference_log[f"step_{step_idx}"] = step_data
                break

            inference_log[f"step_{step_idx}"] = step_data

        return {
            "action_name_label": action_name_label,
            "is_optimal": is_optimal,
            "inference_log": inference_log,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
        }


def multi_threaded_inference(
    handler, model_name: str, test_case: dict
) -> dict:
    """Stub of wtb._llm_response_generation.multi_threaded_inference.

    Calls ``handler.inference(test_case)`` and wraps the result in a dict
    that carries ``test_case["id"]``.  Non-rate-limit exceptions are caught
    and converted into an error dict (matching real wtb behaviour).
    """
    try:
        result = handler.inference(test_case)
        return {"id": test_case.get("id"), "result": result}
    except Exception as exc:
        return {
            "id": test_case.get("id"),
            "result": f"Error during inference: {exc}",
        }


def _install_mock_wtb() -> dict[str, types.ModuleType]:
    """Install a complete mock wtb module tree into sys.modules."""
    mods: dict[str, types.ModuleType] = {}

    # ---- top-level ----
    wtb = _make_module("wtb")
    mods["wtb"] = wtb

    # ---- wtb.model_handler ----
    model_handler = _make_module("wtb.model_handler", wtb)
    mods["wtb.model_handler"] = model_handler

    # ---- wtb.model_handler.base_handler ----
    base_handler = _make_module(
        "wtb.model_handler.base_handler", model_handler
    )
    base_handler.BaseHandler = BaseHandler
    mods["wtb.model_handler.base_handler"] = base_handler

    # ---- wtb._llm_response_generation ----
    llm_gen = _make_module("wtb._llm_response_generation", wtb)
    llm_gen.multi_threaded_inference = multi_threaded_inference
    mods["wtb._llm_response_generation"] = llm_gen

    return mods


# ---------------------------------------------------------------------------
# Module-scoped mock install (happens once per test session)
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Install mock wtb modules before test collection."""
    mods = _install_mock_wtb()
    _MOCK_MODULE_NAMES.extend(mods.keys())


def pytest_unconfigure(config):
    """Remove mock wtb modules so they don't leak into other tests."""
    for name in _MOCK_MODULE_NAMES:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# OTel imports (after environment is set up)
# ---------------------------------------------------------------------------

from opentelemetry.instrumentation.wildtool import WildToolInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    instrumentor = WildToolInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()


# ==================== Minimal test data fixtures ====================


def _make_chat_completion_response(
    content=None,
    tool_calls=None,
    input_tokens=10,
    output_tokens=5,
    model="gpt-4o",
):
    """Build a minimal ChatCompletion-like dict that can be JSON-serialized."""
    message = {"role": "assistant", "content": content or ""}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


class FakeChatCompletion:
    """Mimics openai.types.chat.ChatCompletion enough for _parse_api_response."""

    def __init__(self, data: dict):
        self._data = data

    def json(self):
        return json.dumps(self._data)

    def __getattr__(self, name):
        return self._data[name]


@pytest.fixture()
def make_completion():
    """Factory fixture to build FakeChatCompletion objects."""

    def _factory(**kwargs):
        return FakeChatCompletion(_make_chat_completion_response(**kwargs))

    return _factory


@pytest.fixture()
def simple_test_entry():
    """A minimal WildToolBench test_entry with 1 task, 1 step (prepare_to_answer)."""
    return {
        "id": "wild_tool_bench_test_001",
        "english_env_info": "2025-01-01",
        "english_tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "english_tasks": ["What is the weather in Beijing?"],
        "english_answer_list": [
            [
                {
                    "action": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                    "observation": "Sunny, 25°C",
                    "dependency_list": [],
                },
                {
                    "action": {
                        "name": "prepare_to_answer",
                        "arguments": {},
                    },
                    "observation": "The weather in Beijing is Sunny, 25°C",
                    "dependency_list": [0],
                },
            ]
        ],
    }


@pytest.fixture()
def tool_call_response_factory():
    """Factory to make tool_call ChatCompletion responses."""

    def _factory(tool_name, arguments, tool_call_id="call_001"):
        tc = [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": (
                        json.dumps(arguments)
                        if isinstance(arguments, dict)
                        else arguments
                    ),
                },
            }
        ]
        return FakeChatCompletion(
            _make_chat_completion_response(tool_calls=tc)
        )

    return _factory


@pytest.fixture()
def text_response_factory():
    """Factory to make text-only ChatCompletion responses."""

    def _factory(content, input_tokens=10, output_tokens=5):
        return FakeChatCompletion(
            _make_chat_completion_response(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    return _factory
