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

"""Instrumentor entrypoint for Hermes telemetry."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from .metrics import HermesMetrics
from .wrappers import (
    LLMCallWrapper,
    RunConversationWrapper,
    ToolBatchWrapper,
    ToolCallWrapper,
    ToolDispatchWrapper,
    ToolExecutionWrapper,
)

_INSTRUMENTATION_DEPENDENCIES = ("openai >= 1.0.0",)
_logger = logging.getLogger(__name__)


def _is_missing_target_module(
    exc: ModuleNotFoundError, module_name: str
) -> bool:
    missing_name = exc.name
    return missing_name == module_name or (
        missing_name is not None and module_name.startswith(f"{missing_name}.")
    )


def _safe_wrap_function_wrapper(
    module_name: str, name: str, wrapper: Any
) -> None:
    try:
        wrap_function_wrapper(module_name, name, wrapper)
    except ModuleNotFoundError as exc:
        if not _is_missing_target_module(exc, module_name):
            raise
        _logger.debug(
            "Skipping Hermes instrumentation for missing module %s",
            module_name,
        )
    except AttributeError:
        _logger.debug(
            "Skipping Hermes instrumentation for missing target %s.%s",
            module_name,
            name,
            exc_info=True,
        )


def _safe_import_module(module_name: str) -> Any | None:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if not _is_missing_target_module(exc, module_name):
            raise
        _logger.debug(
            "Skipping Hermes uninstrumentation for missing module %s",
            module_name,
        )
        return None


def _safe_unwrap(parent: Any | None, attribute: str) -> None:
    if parent is None:
        return
    try:
        unwrap(parent, attribute)
    except AttributeError:
        _logger.debug(
            "Skipping Hermes uninstrumentation for missing target %r.%s",
            parent,
            attribute,
            exc_info=True,
        )


class HermesAgentInstrumentor(BaseInstrumentor):
    """Instrumentation for Hermes Agent."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _INSTRUMENTATION_DEPENDENCIES

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )
        metrics = HermesMetrics(meter_provider=meter_provider)

        _safe_wrap_function_wrapper(
            "run_agent",
            "AIAgent.run_conversation",
            RunConversationWrapper(handler),
        )
        _safe_wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_api_call",
            LLMCallWrapper(handler, metrics, streaming=False),
        )
        _safe_wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_streaming_api_call",
            LLMCallWrapper(handler, metrics, streaming=True),
        )
        _safe_wrap_function_wrapper(
            "run_agent",
            "AIAgent._invoke_tool",
            ToolCallWrapper(handler),
        )
        _safe_wrap_function_wrapper(
            "run_agent",
            "AIAgent._execute_tool_calls",
            ToolBatchWrapper(handler),
        )
        _safe_wrap_function_wrapper(
            "model_tools",
            "handle_function_call",
            ToolDispatchWrapper(handler),
        )
        _safe_wrap_function_wrapper(
            "run_agent",
            "handle_function_call",
            ToolDispatchWrapper(handler),
        )
        _safe_wrap_function_wrapper(
            "tools.memory_tool",
            "memory_tool",
            ToolExecutionWrapper(handler, "memory"),
        )
        _safe_wrap_function_wrapper(
            "tools.todo_tool",
            "todo_tool",
            ToolExecutionWrapper(handler, "todo"),
        )
        _safe_wrap_function_wrapper(
            "tools.session_search_tool",
            "session_search",
            ToolExecutionWrapper(handler, "session_search"),
        )
        _safe_wrap_function_wrapper(
            "tools.delegate_tool",
            "delegate_task",
            ToolExecutionWrapper(handler, "delegate_task"),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        model_tools = _safe_import_module("model_tools")
        run_agent = _safe_import_module("run_agent")
        delegate_tool = _safe_import_module("tools.delegate_tool")
        memory_tool = _safe_import_module("tools.memory_tool")
        session_search_tool = _safe_import_module("tools.session_search_tool")
        todo_tool = _safe_import_module("tools.todo_tool")

        ai_agent = getattr(run_agent, "AIAgent", None)
        _safe_unwrap(ai_agent, "run_conversation")
        _safe_unwrap(ai_agent, "_interruptible_api_call")
        _safe_unwrap(ai_agent, "_interruptible_streaming_api_call")
        _safe_unwrap(ai_agent, "_invoke_tool")
        _safe_unwrap(ai_agent, "_execute_tool_calls")
        _safe_unwrap(model_tools, "handle_function_call")
        _safe_unwrap(run_agent, "handle_function_call")
        _safe_unwrap(memory_tool, "memory_tool")
        _safe_unwrap(todo_tool, "todo_tool")
        _safe_unwrap(session_search_tool, "session_search")
        _safe_unwrap(delegate_tool, "delegate_task")
