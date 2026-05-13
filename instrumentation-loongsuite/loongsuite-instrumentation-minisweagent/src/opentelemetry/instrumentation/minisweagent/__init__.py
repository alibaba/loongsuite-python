"""
LoongSuite mini-swe-agent Instrumentation
=========================================

Automatic instrumentation for the `mini-swe-agent
<https://github.com/SWE-agent/mini-swe-agent>`_ framework.

Uses **Method C (hybrid)**: factory injection via ``get_environment`` to
wrap the returned object as a ``TracingEnvironment``, plus ``wrapt``
wrapping of ``DefaultAgent.run`` / ``DefaultAgent.step``.

LLM-call spans are intentionally NOT produced here. The underlying
LiteLLM/OpenAI instrumentation already emits a high-quality LLM span for
each model call, so re-instrumenting at the minisweagent layer would only
produce duplicate spans/metrics.

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

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.minisweagent.package import _instruments
from opentelemetry.instrumentation.minisweagent.version import __version__
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

__all__ = ["MiniSweAgentInstrumentor"]


class MiniSweAgentInstrumentor(BaseInstrumentor):
    """An instrumentor for the mini-swe-agent framework.

    Covers three span kinds:

    * **AGENT** – ``DefaultAgent.run`` (wrapt)
    * **STEP**  – ``DefaultAgent.step`` (wrapt)
    * **TOOL**  – ``TracingEnvironment.execute`` (factory injection via ``get_environment``)

    LLM-call spans are intentionally left to the underlying
    LiteLLM/OpenAI instrumentation to avoid duplicate telemetry.
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
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
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

        # --- factory injection: get_environment ---
        try:
            import minisweagent.environments as _envs_mod

            self.__class__._original_get_environment = _envs_mod.get_environment

            def _wrapped_get_environment(*args: Any, **kw: Any) -> Any:
                env = MiniSweAgentInstrumentor._original_get_environment(*args, **kw)
                return TracingEnvironment(env, tracer)

            _envs_mod.get_environment = _wrapped_get_environment
        except Exception as exc:
            logger.warning("Could not wrap get_environment: %s", exc)

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

        # --- restore original factory ---
        if self.__class__._original_get_environment is not None:
            try:
                import minisweagent.environments as _envs_mod

                _envs_mod.get_environment = self.__class__._original_get_environment
                self.__class__._original_get_environment = None
            except Exception as exc:
                logger.debug("Could not restore get_environment: %s", exc)
