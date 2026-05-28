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
WideSearch instrumentation supporting `widesearch >= 0.1.0`.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.widesearch import WideSearchInstrumentor

    WideSearchInstrumentor().instrument()

API
---
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.widesearch.package import _instruments
from opentelemetry.instrumentation.widesearch.patch import (
    wrap_create_sub_agents_factory,
    wrap_invoke_tool_call,
    wrap_run_single_query,
    wrap_runner_run,
    wrap_runner_step,
)
from opentelemetry.instrumentation.widesearch.version import __version__
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

_RUN_MODULE = "src.agent.run"
_MULTI_AGENT_MODULE = "src.agent.multi_agent_tools"

__all__ = ["WideSearchInstrumentor", "__version__"]


class WideSearchInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for WideSearch framework.

    Instruments the following components:
    - run_single_query(): ENTRY span
    - Runner.run(): AGENT span (async generator)
    - Runner._step(): STEP span
    - Runner._invoke_tool_call(): TOOL spans
    - create_sub_agents_wrap(): TASK span
    """

    def __init__(self):
        super().__init__()
        self._handler = None

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

        # H1: ENTRY span
        try:
            wrap_function_wrapper(
                module=_RUN_MODULE,
                name="run_single_query",
                wrapper=lambda w, i, a, k: wrap_run_single_query(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented run_single_query")
        except Exception as e:
            logger.warning(f"Failed to instrument run_single_query: {e}")

        # H2: AGENT span
        try:
            wrap_function_wrapper(
                module=_RUN_MODULE,
                name="Runner.run",
                wrapper=lambda w, i, a, k: wrap_runner_run(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented Runner.run")
        except Exception as e:
            logger.warning(f"Failed to instrument Runner.run: {e}")

        # H3: STEP span
        try:
            wrap_function_wrapper(
                module=_RUN_MODULE,
                name="Runner._step",
                wrapper=lambda w, i, a, k: wrap_runner_step(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented Runner._step")
        except Exception as e:
            logger.warning(f"Failed to instrument Runner._step: {e}")

        # H4: TOOL spans
        try:
            wrap_function_wrapper(
                module=_RUN_MODULE,
                name="Runner._invoke_tool_call",
                wrapper=lambda w, i, a, k: wrap_invoke_tool_call(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented Runner._invoke_tool_call")
        except Exception as e:
            logger.warning(
                f"Failed to instrument Runner._invoke_tool_call: {e}"
            )

        # H5: TASK span (wrap factory)
        try:
            wrap_function_wrapper(
                module=_MULTI_AGENT_MODULE,
                name="create_sub_agents_wrap",
                wrapper=lambda w, i, a, k: wrap_create_sub_agents_factory(
                    w, i, a, k, handler=self._handler
                ),
            )
            logger.debug("Instrumented create_sub_agents_wrap")
        except Exception as e:
            logger.warning(f"Failed to instrument create_sub_agents_wrap: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import src.agent.run  # noqa: PLC0415

            unwrap(src.agent.run, "run_single_query")
            unwrap(src.agent.run.Runner, "run")
            unwrap(src.agent.run.Runner, "_step")
            unwrap(src.agent.run.Runner, "_invoke_tool_call")
            logger.debug("Uninstrumented src.agent.run")
        except Exception as e:
            logger.warning(f"Failed to uninstrument src.agent.run: {e}")

        try:
            import src.agent.multi_agent_tools  # noqa: PLC0415

            unwrap(src.agent.multi_agent_tools, "create_sub_agents_wrap")
            logger.debug("Uninstrumented src.agent.multi_agent_tools")
        except Exception as e:
            logger.warning(
                f"Failed to uninstrument src.agent.multi_agent_tools: {e}"
            )

        self._handler = None
