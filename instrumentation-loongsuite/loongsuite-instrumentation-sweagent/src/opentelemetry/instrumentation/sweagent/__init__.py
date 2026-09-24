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

"""LongSuite instrumentation for SWE-agent using ExtendedTelemetryHandler."""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.sweagent.package import _instruments
from opentelemetry.instrumentation.sweagent.patch import (
    _AGENT_HOOKS_MODULE,
    _AGENTS_MODULE,
    _RUN_HOOKS_MODULE,
    bind_extended_handler,
    wrap_combined_agent_hook_on_run_done,
    wrap_combined_agent_hook_on_run_start,
    wrap_combined_agent_hook_on_step_done,
    wrap_combined_agent_hook_on_step_start,
    wrap_combined_run_hooks_on_instance_completed,
    wrap_combined_run_hooks_on_instance_start,
    wrap_default_agent_handle_action,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

_COMBINED_RUN_HOOK_PATCHES: tuple[tuple[str, Any], ...] = (
    ("on_instance_start", wrap_combined_run_hooks_on_instance_start),
    ("on_instance_completed", wrap_combined_run_hooks_on_instance_completed),
)

_COMBINED_AGENT_HOOK_PATCHES: tuple[tuple[str, Any], ...] = (
    ("on_run_start", wrap_combined_agent_hook_on_run_start),
    ("on_run_done", wrap_combined_agent_hook_on_run_done),
    ("on_step_start", wrap_combined_agent_hook_on_step_start),
    ("on_step_done", wrap_combined_agent_hook_on_step_done),
)

__all__ = ["SweagentInstrumentor"]


class SweagentInstrumentor(BaseInstrumentor):
    """Instrument SWE-agent run and agent hooks with GenAI semantic spans."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: ExtendedTelemetryHandler | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        handler = self._handler

        for name, fn in _COMBINED_RUN_HOOK_PATCHES:
            try:
                wrap_function_wrapper(
                    _RUN_HOOKS_MODULE,
                    f"CombinedRunHooks.{name}",
                    bind_extended_handler(handler, fn),
                )
                logger.debug("Wrapped CombinedRunHooks.%s", name)
            except Exception as e:
                logger.warning(
                    "Failed to wrap CombinedRunHooks.%s: %s", name, e
                )

        for name, fn in _COMBINED_AGENT_HOOK_PATCHES:
            try:
                wrap_function_wrapper(
                    _AGENT_HOOKS_MODULE,
                    f"CombinedAgentHook.{name}",
                    bind_extended_handler(handler, fn),
                )
                logger.debug("Wrapped CombinedAgentHook.%s", name)
            except Exception as e:
                logger.warning(
                    "Failed to wrap CombinedAgentHook.%s: %s", name, e
                )

        try:
            wrap_function_wrapper(
                _AGENTS_MODULE,
                "DefaultAgent.handle_action",
                bind_extended_handler(
                    handler, wrap_default_agent_handle_action
                ),
            )
            logger.debug("Wrapped DefaultAgent.handle_action")
        except Exception as e:
            logger.warning("Failed to wrap DefaultAgent.handle_action: %s", e)

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        try:
            import sweagent.agent.agents as agent_mod  # noqa: PLC0415

            unwrap(agent_mod.DefaultAgent, "handle_action")
        except Exception as e:
            logger.warning(
                "Failed to unwrap DefaultAgent.handle_action: %s", e
            )

        try:
            import sweagent.agent.hooks.abstract as agent_hooks  # noqa: PLC0415

            for name, _ in _COMBINED_AGENT_HOOK_PATCHES:
                try:
                    unwrap(agent_hooks.CombinedAgentHook, name)
                except Exception as e:
                    logger.warning(
                        "Failed to unwrap CombinedAgentHook.%s: %s", name, e
                    )
        except Exception as e:
            logger.warning(
                "Failed to import agent hooks for uninstrument: %s", e
            )

        try:
            import sweagent.run.hooks.abstract as run_hooks  # noqa: PLC0415

            for name, _ in _COMBINED_RUN_HOOK_PATCHES:
                try:
                    unwrap(run_hooks.CombinedRunHooks, name)
                except Exception as e:
                    logger.warning(
                        "Failed to unwrap CombinedRunHooks.%s: %s", name, e
                    )
        except Exception as e:
            logger.warning(
                "Failed to import run hooks for uninstrument: %s", e
            )

        self._handler = None
