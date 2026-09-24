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
LoongSuite mini-swe-agent Instrumentation
=========================================

Automatic instrumentation for the `mini-swe-agent
<https://github.com/SWE-agent/mini-swe-agent>`_ framework.

Uses **Method C (hybrid)**:

* factory injection via ``get_environment`` → ``TracingEnvironment`` (TOOL / ``execute_tool``)
* ``wrapt`` on ``DefaultAgent.run`` / ``DefaultAgent.step``, and ENTRY on Typer ``minisweagent.run.mini:app``

LLM spans stay in LiteLLM/OpenAI instrumentation; this package adds Agent/ReAct/ENTRY/TOOL spans and (with the env vars described in the instrumentor docstring) full ARMS-aligned message / tool payloads.

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.minisweagent import MiniSweAgentInstrumentor

    MiniSweAgentInstrumentor().instrument()

    # Then use mini-swe-agent as normal
    from minisweagent.models import get_model
    from minisweagent.environments import get_environment
    from minisweagent.agents.default import DefaultAgent

    model = get_model("gpt-4o")
    env = get_environment({"environment_class": "local"})
    agent = DefaultAgent(model=model, environment=env)
    agent.run("Fix the bug")

API
---
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.minisweagent.package import _instruments
from opentelemetry.instrumentation.minisweagent.version import __version__

logger = logging.getLogger(__name__)

__all__ = ["MiniSweAgentInstrumentor"]


class MiniSweAgentInstrumentor(BaseInstrumentor):
    """An instrumentor for the mini-swe-agent framework.

    Covers GenAI span kinds (ARMS / LoongSuite conventions when
    ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` and
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY``):

    * **ENTRY** – Typer ``mini`` callable ``app`` (``minisweagent.run.mini:app``), span name ``enter_ai_application_system``
    * **AGENT** – ``DefaultAgent.run`` via ``invoke_agent`` (+ messages / system instruction / tool definitions)
    * **STEP** – ``DefaultAgent.step`` (ReAct round)
    * **TOOL** – ``TracingEnvironment.execute`` (``execute_tool`` for bash)

    LLM-call spans remain with the underlying LiteLLM/OpenAI instrumentation.
    """

    _original_get_environment = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider=tracer_provider,
        )

        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentRunWrapper,
            DefaultAgentStepWrapper,
        )
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            patch_mini_cli_app_module,
        )
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        # --- factory injection: get_environment ---
        try:
            import minisweagent.environments as _envs_mod

            if self.__class__._original_get_environment is None:
                self.__class__._original_get_environment = (
                    _envs_mod.get_environment
                )

            def _wrapped_get_environment(*args: Any, **kw: Any) -> Any:
                env = MiniSweAgentInstrumentor._original_get_environment(
                    *args, **kw
                )
                return TracingEnvironment(env, tracer)

            _envs_mod.get_environment = _wrapped_get_environment
        except Exception as exc:
            logger.warning("Could not wrap get_environment: %s", exc)

        try:
            patch_mini_cli_app_module()
        except Exception as exc:
            logger.warning(
                "Could not patch minisweagent.run.mini.app (ENTRY): %s", exc
            )

        # --- wrapt: DefaultAgent.run / DefaultAgent.step ---
        try:
            wrap_function_wrapper(
                module="minisweagent.agents.default",
                name="DefaultAgent.run",
                wrapper=DefaultAgentRunWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap DefaultAgent.run: %s", exc)

        try:
            wrap_function_wrapper(
                module="minisweagent.agents.default",
                name="DefaultAgent.step",
                wrapper=DefaultAgentStepWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap DefaultAgent.step: %s", exc)

    def _uninstrument(self, **kwargs: Any) -> None:
        # --- restore wrapt patches on DefaultAgent ---
        try:
            from minisweagent.agents.default import DefaultAgent

            if hasattr(DefaultAgent.run, "__wrapped__"):
                DefaultAgent.run = DefaultAgent.run.__wrapped__  # type: ignore[attr-defined]
            if hasattr(DefaultAgent.step, "__wrapped__"):
                DefaultAgent.step = DefaultAgent.step.__wrapped__  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Could not unwrap DefaultAgent: %s", exc)

        try:
            from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
                unpatch_mini_cli_app_module,
            )

            unpatch_mini_cli_app_module()
        except Exception as exc:
            logger.debug("Could not unpatch mini app: %s", exc)

        # --- restore original factory ---
        if self.__class__._original_get_environment is not None:
            try:
                import minisweagent.environments as _envs_mod

                _envs_mod.get_environment = (
                    self.__class__._original_get_environment
                )
                self.__class__._original_get_environment = None
            except Exception as exc:
                logger.debug("Could not restore get_environment: %s", exc)
