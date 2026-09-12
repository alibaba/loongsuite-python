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

"""OpenTelemetry OpenHands Instrumentation.

Wraps the legacy V0 (CodeAct + AgentController + Runtime) path:

* V0 — ``python -m openhands.core.main``. We add
  ``ENTRY → AGENT → STEP → TOOL`` directly on top of the controller / runtime
  call chain. LLM spans come from the bundled LiteLLM instrumentor.

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    OpenHandsInstrumentor().instrument()
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openhands.config import (
    OTEL_INSTRUMENTATION_OPENHANDS_AUTO_INSTRUMENT_LITELLM,
    OTEL_INSTRUMENTATION_OPENHANDS_ENABLED,
    OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS,
)
from opentelemetry.instrumentation.openhands.package import _instruments
from opentelemetry.instrumentation.openhands.version import __version__

logger = logging.getLogger(__name__)

__all__ = ["OpenHandsInstrumentor"]


# ---------------------------------------------------------------------------
# Wrap-point registry — single source of truth shared with _uninstrument.
# Entries: (module, qualified_name)
# ---------------------------------------------------------------------------

_PATCH_TARGETS: list[tuple[str, str]] = [
    ("openhands.core.main", "run_controller"),
    ("openhands.core.loop", "run_agent_until_done"),
    # AgentController.__init__ / .close are the *primary* ENTRY+AGENT
    # span source for V0 — they're class methods, so they're patchable
    # regardless of the from-import binding problem in main.py
    # (see v0_wrappers.AgentControllerInitWrapper docstring).
    (
        "openhands.controller.agent_controller",
        "AgentController.__init__",
    ),
    (
        "openhands.controller.agent_controller",
        "AgentController.close",
    ),
    (
        "openhands.controller.agent_controller",
        "AgentController._step",
    ),
    ("openhands.runtime.base", "Runtime.run_action"),
    # LLM context bridge — re-attaches the current sid-stashed context
    # (STEP while a step is open) onto every ``LLM.completion`` invocation
    # so the downstream LiteLLM / Aliyun GenAI auto-instrumentation emits
    # the LLM span as a child of STEP and shares its ``trace_id``.
    ("openhands.llm.llm", "LLM.__init__"),
]


def _module_importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ModuleNotFoundError:
        return False
    except Exception:
        # Other import errors should still let the wrap attempt surface a
        # warning.
        return True


def _safe_wrap(module: str, name: str, wrapper: Any) -> bool:
    """Patch ``module.name`` with ``wrapper``; classify failures sensibly."""
    if not _module_importable(module):
        # OpenHands versions can move modules around. Missing V0 modules
        # should not prevent applications from starting.
        logger.debug(
            "OpenHands instrumentation: module %s not importable, skipping %s",
            module,
            name,
        )
        return False
    try:
        wrap_function_wrapper(module=module, name=name, wrapper=wrapper)
        logger.debug("OpenHands instrumentation: wrapped %s.%s", module, name)
        return True
    except (AttributeError, ImportError) as exc:
        # Attribute missing inside the module — usually a version-skew issue.
        logger.warning(
            "OpenHands instrumentation: could not wrap %s.%s: %s",
            module,
            name,
            exc,
        )
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "OpenHands instrumentation: unexpected error wrapping %s.%s: %s",
            module,
            name,
            exc,
        )
        return False


def _safe_unwrap(module: str, qualname: str) -> None:
    """Unwrap a previously ``wrapt``-patched function or method."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return
    parts = qualname.split(".")
    obj: Any = mod
    parents: list[Any] = [mod]
    try:
        for p in parts:
            obj = getattr(obj, p)
            parents.append(obj)
    except Exception:
        return
    if not hasattr(obj, "__wrapped__"):
        return
    parent = parents[-2]
    try:
        setattr(parent, parts[-1], obj.__wrapped__)
    except Exception:
        pass


class OpenHandsInstrumentor(BaseInstrumentor):
    """Instrumentation entry point for OpenHands V0."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not OTEL_INSTRUMENTATION_OPENHANDS_ENABLED:
            logger.info("OpenHands instrumentation disabled via env var")
            return

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__, __version__, tracer_provider=tracer_provider
        )

        from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
            AgentControllerCloseWrapper,
            AgentControllerInitWrapper,
            AgentControllerStepWrapper,
            LLMInitWrapper,
            RunAgentUntilDoneWrapper,
            RunControllerWrapper,
            RuntimeRunActionWrapper,
        )

        if OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS:
            self._install_v0_patches(
                tracer,
                {
                    "run_controller": RunControllerWrapper,
                    "run_agent_until_done": RunAgentUntilDoneWrapper,
                    "agent_init": AgentControllerInitWrapper,
                    "agent_close": AgentControllerCloseWrapper,
                    "agent_step": AgentControllerStepWrapper,
                    "runtime_run_action": RuntimeRunActionWrapper,
                    "llm_init": LLMInitWrapper,
                },
            )

        # Auto-enable bundled LiteLLM instrumentation so SDK / V0 LLM
        # ``litellm.completion()`` calls become LLM spans.
        if OTEL_INSTRUMENTATION_OPENHANDS_AUTO_INSTRUMENT_LITELLM:
            self._maybe_enable_litellm(**kwargs)

    def _install_v0_patches(self, tracer, factories) -> None:
        RunControllerWrapper = factories["run_controller"]
        RunAgentUntilDoneWrapper = factories["run_agent_until_done"]
        AgentControllerInitWrapper = factories["agent_init"]
        AgentControllerCloseWrapper = factories["agent_close"]
        AgentControllerStepWrapper = factories["agent_step"]
        RuntimeRunActionWrapper = factories["runtime_run_action"]
        LLMInitWrapper = factories["llm_init"]

        # `run_controller` and `run_agent_until_done` patches are best-effort:
        # they only fire when run_controller is called via the proper module
        # path (programmatic / test). When OpenHands is launched via
        # ``python -m openhands.core.main``, the from-import binding in
        # main.py bypasses these patches — the AgentController.__init__ /
        # .close patches below take over and produce ENTRY+AGENT spans
        # reliably (class methods are immune to from-import binding).
        _safe_wrap(
            "openhands.core.main",
            "run_controller",
            RunControllerWrapper(tracer),
        )
        _safe_wrap(
            "openhands.core.loop",
            "run_agent_until_done",
            RunAgentUntilDoneWrapper(tracer),
        )
        _safe_wrap(
            "openhands.controller.agent_controller",
            "AgentController.__init__",
            AgentControllerInitWrapper(tracer),
        )
        _safe_wrap(
            "openhands.controller.agent_controller",
            "AgentController.close",
            AgentControllerCloseWrapper(tracer),
        )
        _safe_wrap(
            "openhands.controller.agent_controller",
            "AgentController._step",
            AgentControllerStepWrapper(tracer),
        )
        _safe_wrap(
            "openhands.runtime.base",
            "Runtime.run_action",
            RuntimeRunActionWrapper(tracer),
        )
        # LLM context bridge — patches ``LLM.__init__`` so every instance's
        # ``self._completion`` re-attaches the latest sid-stashed context.
        # See ``LLMInitWrapper`` for why we need this even though the LLM
        # call is synchronous: in real OpenHands deployments LiteLLM ends
        # up creating its span in a thread / context that ``contextvars``
        # didn't propagate STEP into, so we re-attach explicitly.
        _safe_wrap(
            "openhands.llm.llm",
            "LLM.__init__",
            LLMInitWrapper(tracer),
        )

    def _maybe_enable_litellm(self, **kwargs: Any) -> None:
        try:
            from opentelemetry.instrumentation.litellm import (
                LiteLLMInstrumentor,
            )
        except Exception as exc:
            logger.debug(
                "LiteLLM instrumentation not available, skipping: %s", exc
            )
            return
        try:
            instr = LiteLLMInstrumentor()
            already = getattr(
                instr, "_is_instrumented_by_opentelemetry", False
            )
            if not already:
                instr.instrument(**kwargs)
        except Exception as exc:
            logger.debug(
                "Could not auto-enable LiteLLM instrumentation: %s", exc
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        for module, qualname in _PATCH_TARGETS:
            _safe_unwrap(module, qualname)
