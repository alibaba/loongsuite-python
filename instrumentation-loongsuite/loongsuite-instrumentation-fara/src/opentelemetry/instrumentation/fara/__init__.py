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

"""OpenTelemetry Fara Instrumentation.

Automatic instrumentation for the
`Microsoft Fara <https://github.com/microsoft/fara>`_ Computer Use Agent
framework.

Span hierarchy
--------------

::

    ENTRY  enter_ai_application_system                (run_fara_agent)
    └── AGENT  invoke_agent FaraAgent                 (FaraAgent.run)
        ├── STEP  react step (round=1)                (generate_model_call)
        │   ├── LLM  chat {model}                     (AsyncCompletions.create)
        │   │                                         * emitted by opentelemetry-
        │   │                                           instrumentation-openai-v2
        │   └── TOOL  execute_tool {action}           (execute_action)
        ├── STEP  react step (round=2)
        │   ├── LLM  chat {model}
        │   └── TOOL  execute_tool {action}
        └── ...

Design principles
-----------------

* **LLM spans are owned by the OpenAI instrumentation.** Fara calls
  ``AsyncOpenAI.chat.completions.create`` in ``FaraAgent._make_model_call``;
  ``opentelemetry-instrumentation-openai-v2`` already wraps that exact
  method, so we rely on it for token usage / model / request attrs and
  let its LLM span attach naturally as a child of the current STEP.
* **STEP rotation via ``generate_model_call``.``FaraAgent.run`` is a
  single async method with an internal for-loop; we can't hook code
  blocks. ``generate_model_call`` is called exactly once per round at
  the top of each iteration, so wrapping it is equivalent to "start of
  round" and mirrors the WebArena ``NextActionWrapper`` pattern.
* **ContextVar isolation.`` ``FaraAgent.run`` is async, so all per-task
  state lives in ContextVars for safe concurrent execution.

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.fara import FaraInstrumentor

    FaraInstrumentor().instrument()

    # Then run Fara as normal (e.g. ``python -m fara.run_fara --task "..."``).
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.fara.package import _instruments
from opentelemetry.instrumentation.fara.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

__all__ = ["FaraInstrumentor", "__version__"]


# (module, qualname, wrapper_attr_name)
_PATCH_TARGETS = (
    ("fara.run_fara", "run_fara_agent", "_entry_wrapper"),
    ("fara.fara_agent", "FaraAgent.run", "_agent_run_wrapper"),
    ("fara.fara_agent", "FaraAgent.generate_model_call", "_generate_model_call_wrapper"),
    ("fara.fara_agent", "FaraAgent.execute_action", "_tool_wrapper"),
)


class FaraInstrumentor(BaseInstrumentor):
    """An ``opentelemetry-instrumentation`` plugin for Microsoft Fara.

    Spans (see module docstring) are emitted via ``wrapt`` hooks on four
    Fara functions. LLM spans are intentionally **not** emitted here (the
    OpenAI SDK probe handles them).
    """

    _patched: list[tuple[str, str]] = []
    _handler: ExtendedTelemetryHandler | None = None

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

        from opentelemetry.instrumentation.fara.internal._wrappers import (
            AgentRunWrapper,
            EntryWrapper,
            GenerateModelCallWrapper,
            ToolWrapper,
        )

        wrappers = {
            "_entry_wrapper": EntryWrapper(self._handler),
            "_agent_run_wrapper": AgentRunWrapper(self._handler),
            "_generate_model_call_wrapper": GenerateModelCallWrapper(self._handler),
            "_tool_wrapper": ToolWrapper(self._handler),
        }

        type(self)._patched = []
        for module, qualname, wrapper_key in _PATCH_TARGETS:
            try:
                wrap_function_wrapper(
                    module=module,
                    name=qualname,
                    wrapper=wrappers[wrapper_key],
                )
                type(self)._patched.append((module, qualname))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FaraInstrumentor: could not wrap %s.%s: %s",
                    module,
                    qualname,
                    exc,
                )

    def _uninstrument(self, **kwargs: Any) -> None:
        import importlib  # noqa: PLC0415

        for module, qualname in list(type(self)._patched):
            try:
                mod = importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "FaraInstrumentor: could not import %s for unwrap: %s",
                    module,
                    exc,
                )
                continue
            parts = qualname.split(".")
            try:
                target = mod
                for p in parts[:-1]:
                    target = getattr(target, p)
                if hasattr(target, parts[-1]):
                    unwrap(target, parts[-1])
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "FaraInstrumentor: could not unwrap %s.%s: %s",
                    module,
                    qualname,
                    exc,
                )
        type(self)._patched = []
        self._handler = None
