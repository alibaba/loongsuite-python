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

"""OpenTelemetry instrumentation for ByteDance DeerFlow.

Implements the hybrid plan documented in
``llm-dev/deer-flow/investigate/execute.md``: LLM / Tool / ReAct Step spans are
delegated to ``loongsuite-instrumentation-langchain`` (DeerFlow is built on
``langchain.agents.create_agent``), while DeerFlow-specific Entry / Agent /
Task / Sandbox / Memory spans are produced by this package via seven
``wrapt`` monkey patches on DeerFlow public APIs:

* ``runtime.runs.worker.run_agent`` → ENTRY span
* ``subagents.executor.SubagentExecutor._aexecute`` → AGENT span (subagent)
* ``tools.builtins.task_tool.task_tool`` → TASK span (subagent dispatch)
* ``sandbox.sandbox.Sandbox.execute_command`` (and the other abstract
  methods) → TOOL span (``bash`` / ``read_file`` / ...)
* ``sandbox.sandbox_provider.SandboxProvider.acquire`` /
  ``acquire_async`` / ``release`` → TASK span (sandbox lifecycle)
* ``agents.memory.storage.FileMemoryStorage.load`` / ``save`` → TASK span
  (``memory.load`` / ``memory.save``)
* ``agents.memory.updater.MemoryUpdater.aupdate_memory`` → TASK span
  (``memory.update``)

This package **does not** wrap ``BaseCallbackManager`` — that path is owned by
``loongsuite-instrumentation-langchain`` and wrapping it again would create
duplicate LLM / Tool spans.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry.instrumentation.deer_flow.package import _instruments
from opentelemetry.instrumentation.deer_flow.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

# (module_name, attr_path) tuples registered by ``_instrument`` so that
# ``_uninstrument`` can call ``unwrap`` on each target.
_UNINSTRUMENT_TARGETS: list[tuple[str, str]] = []


class DeerFlowInstrumentor(BaseInstrumentor):
    """Instrument ByteDance DeerFlow 2.1+ with OpenTelemetry."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: ExtendedTelemetryHandler | None = None
        self._targets: list[tuple[str, str]] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            import deerflow  # noqa: F401  — exercise the dependency.
        except ImportError:
            logger.debug(
                "DeerFlow is not installed; skipping instrumentation."
            )
            return
        except Exception as exc:
            logger.warning(
                "DeerFlow import raised an unexpected error: %s", exc
            )
            return

        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        from opentelemetry.instrumentation.deer_flow.patches import (
            entry,
            memory,
            sandbox,
            subagent,
            task_tool,
        )

        self._targets = []
        for module in (entry, subagent, task_tool, sandbox, memory):
            try:
                targets = module.instrument(self._handler)
            except Exception as exc:
                logger.warning(
                    "DeerFlow patch module %s failed to instrument: %s",
                    module.__name__,
                    exc,
                    exc_info=True,
                )
                continue
            self._targets.extend(targets)

        _UNINSTRUMENT_TARGETS[:] = list(self._targets)

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        for module_name, attr in self._targets:
            try:
                module = import_module(module_name)
                if "." in attr:
                    class_name, method_name = attr.split(".", 1)
                    cls = getattr(module, class_name, None)
                    if cls is None:
                        continue
                    unwrap(cls, method_name)
                else:
                    unwrap(module, attr)
            except Exception as exc:
                logger.debug(
                    "DeerFlow: could not unwrap %s.%s: %s",
                    module_name,
                    attr,
                    exc,
                )
        self._targets = []
        _UNINSTRUMENT_TARGETS[:] = []
        self._handler = None


__all__ = ["DeerFlowInstrumentor", "__version__"]
