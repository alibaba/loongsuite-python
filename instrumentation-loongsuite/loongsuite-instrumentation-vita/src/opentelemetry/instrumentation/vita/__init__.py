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

"""
OpenTelemetry VitaBench Instrumentation

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.vita import VitaInstrumentor

    VitaInstrumentor().instrument()

    # ... run vitabench tasks ...

    VitaInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.vita.package import _instruments
from opentelemetry.instrumentation.vita.patch import (
    wrap_generate,
    wrap_generate_next_message,
    wrap_get_response,
    wrap_orchestrator_run,
    wrap_orchestrator_step,
    wrap_run_task,
)
from opentelemetry.instrumentation.vita.version import __version__
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

__all__ = ["VitaInstrumentor", "__version__"]


class VitaInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for VitaBench framework.

    Instruments the following components:
    - vita.run.run_task(): Entry spans (ENTRY)
    - Orchestrator.run(): Workflow spans (CHAIN)
    - Orchestrator.step(): ReAct step spans (STEP)
    - LLMAgent.generate_next_message(): Agent spans (AGENT)
    - generate(): LLM call spans (LLM)
    - Environment.get_response(): Tool execution spans (TOOL)
    """

    def __init__(self):
        super().__init__()
        self._handler = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable VitaBench instrumentation."""
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # Hook #5: generate -> LLM. Wrap this first so modules that import
        # generate directly (for example vita.agent.llm_agent) bind to the
        # instrumented function during their import.
        try:
            wrap_function_wrapper(
                module="vita.utils.llm_utils",
                name="generate",
                wrapper=lambda w, i, a, k: wrap_generate(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented vita.utils.llm_utils.generate")
        except Exception as e:
            logger.warning(
                f"Could not wrap vita.utils.llm_utils.generate: {e}"
            )

        # Hook #1: run_task -> ENTRY
        try:
            wrap_function_wrapper(
                module="vita.run",
                name="run_task",
                wrapper=lambda w, i, a, k: wrap_run_task(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented vita.run.run_task")
        except Exception as e:
            logger.warning(f"Could not wrap vita.run.run_task: {e}")

        # Hook #2: Orchestrator.run -> CHAIN
        try:
            wrap_function_wrapper(
                module="vita.orchestrator.orchestrator",
                name="Orchestrator.run",
                wrapper=lambda w, i, a, k: wrap_orchestrator_run(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented Orchestrator.run")
        except Exception as e:
            logger.warning(f"Could not wrap Orchestrator.run: {e}")

        # Hook #3: Orchestrator.step -> STEP
        try:
            wrap_function_wrapper(
                module="vita.orchestrator.orchestrator",
                name="Orchestrator.step",
                wrapper=lambda w, i, a, k: wrap_orchestrator_step(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented Orchestrator.step")
        except Exception as e:
            logger.warning(f"Could not wrap Orchestrator.step: {e}")

        # Hook #4a: LLMAgent.generate_next_message -> AGENT
        try:
            wrap_function_wrapper(
                module="vita.agent.llm_agent",
                name="LLMAgent.generate_next_message",
                wrapper=lambda w, i, a, k: wrap_generate_next_message(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented LLMAgent.generate_next_message")
        except Exception as e:
            logger.warning(
                f"Could not wrap LLMAgent.generate_next_message: {e}"
            )

        # Hook #4b: LLMSoloAgent.generate_next_message -> AGENT
        try:
            wrap_function_wrapper(
                module="vita.agent.llm_agent",
                name="LLMSoloAgent.generate_next_message",
                wrapper=lambda w, i, a, k: wrap_generate_next_message(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented LLMSoloAgent.generate_next_message")
        except Exception as e:
            logger.warning(
                f"Could not wrap LLMSoloAgent.generate_next_message: {e}"
            )

        # Hook #6: Environment.get_response -> TOOL
        try:
            wrap_function_wrapper(
                module="vita.environment.environment",
                name="Environment.get_response",
                wrapper=lambda w, i, a, k: wrap_get_response(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented Environment.get_response")
        except Exception as e:
            logger.warning(f"Could not wrap Environment.get_response: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable VitaBench instrumentation."""
        try:
            import vita.run  # noqa: PLC0415

            unwrap(vita.run, "run_task")
        except Exception as e:
            logger.debug(f"Failed to uninstrument vita.run.run_task: {e}")

        try:
            import vita.orchestrator.orchestrator  # noqa: PLC0415

            unwrap(vita.orchestrator.orchestrator.Orchestrator, "run")
            unwrap(vita.orchestrator.orchestrator.Orchestrator, "step")
        except Exception as e:
            logger.debug(f"Failed to uninstrument Orchestrator: {e}")

        try:
            import vita.agent.llm_agent  # noqa: PLC0415

            unwrap(vita.agent.llm_agent.LLMAgent, "generate_next_message")
            unwrap(vita.agent.llm_agent.LLMSoloAgent, "generate_next_message")
        except Exception as e:
            logger.debug(f"Failed to uninstrument LLMAgent: {e}")

        try:
            import vita.utils.llm_utils  # noqa: PLC0415

            unwrap(vita.utils.llm_utils, "generate")
        except Exception as e:
            logger.debug(f"Failed to uninstrument generate: {e}")

        try:
            import vita.environment.environment  # noqa: PLC0415

            unwrap(vita.environment.environment.Environment, "get_response")
        except Exception as e:
            logger.debug(f"Failed to uninstrument Environment: {e}")

        self._handler = None
