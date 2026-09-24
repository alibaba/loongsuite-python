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
LoongSuite LangGraph instrumentation supporting ``langgraph >= 0.2``.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.langgraph import LangGraphInstrumentor

    LangGraphInstrumentor().instrument()

    # ... use LangGraph as normal ...

    LangGraphInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Collection

from wrapt import ObjectProxy, wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.langgraph.internal.patch import (
    _astream_wrapper,
    _create_react_agent_wrapper,
    _stream_wrapper,
)
from opentelemetry.instrumentation.langgraph.package import _instruments
from opentelemetry.instrumentation.utils import unwrap

logger = logging.getLogger(__name__)

__all__ = ["LangGraphInstrumentor"]

_PREGEL_MODULE = "langgraph.pregel"


def _get_create_react_agent_locations() -> list[tuple[str, str]]:
    """Return (module, attribute) pairs for ``create_react_agent``.

    LangGraph >= 0.2 exposes it at ``langgraph.prebuilt``.
    We also try the internal module path for robustness.
    """
    locations: list[tuple[str, str]] = []
    try:
        from langgraph.prebuilt import (  # noqa: PLC0415
            create_react_agent,  # noqa: F401
        )

        locations.append(("langgraph.prebuilt", "create_react_agent"))
    except ImportError as exc:
        logger.debug(
            "langgraph.prebuilt.create_react_agent not available: %s", exc
        )

    try:
        from langgraph.prebuilt.chat_agent_executor import (  # noqa: PLC0415
            create_react_agent,  # noqa: F401
        )

        locations.append(
            ("langgraph.prebuilt.chat_agent_executor", "create_react_agent")
        )
    except ImportError as exc:
        logger.debug(
            "langgraph.prebuilt.chat_agent_executor.create_react_agent "
            "not available: %s",
            exc,
        )
    return locations


class LangGraphInstrumentor(BaseInstrumentor):
    """An instrumentor for LangGraph."""

    def __init__(self) -> None:
        super().__init__()
        # Module paths that were successfully patched by this instance.
        # Only initialize if not already set (singleton may reuse instance).
        if not hasattr(self, "_patched_cra_locations"):
            self._patched_cra_locations = []
        if not hasattr(self, "_pregel_patched"):
            self._pregel_patched = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        self._instrument_create_react_agent()
        self._instrument_pregel()

    def _instrument_create_react_agent(self) -> None:
        locations = _get_create_react_agent_locations()
        if not locations:
            logger.warning(
                "create_react_agent not found in langgraph; "
                "LangGraph ReAct instrumentation skipped."
            )
            return

        # Skip targets already wrapped (e.g. wrapt proxy) to avoid nested
        # wrappers when the same callable is re-exported across modules.
        patched: list[tuple[str, str]] = []
        for module_path, attr_name in locations:
            try:
                mod = importlib.import_module(module_path)
                target = getattr(mod, attr_name)
            except (ImportError, AttributeError):
                continue
            if isinstance(target, ObjectProxy):
                logger.debug(
                    "Skipping %s.%s (already wrapped)",
                    module_path,
                    attr_name,
                )
                continue
            wrap_function_wrapper(
                module_path, attr_name, _create_react_agent_wrapper
            )
            logger.debug("Patched %s.%s", module_path, attr_name)
            patched.append((module_path, attr_name))

        self._patched_cra_locations = patched

    def _instrument_pregel(self) -> None:
        """Patch ``Pregel.stream`` and ``Pregel.astream`` to inject
        metadata when the graph is a marked ReAct agent.
        """
        try:
            wrap_function_wrapper(
                _PREGEL_MODULE, "Pregel.stream", _stream_wrapper
            )
            wrap_function_wrapper(
                _PREGEL_MODULE, "Pregel.astream", _astream_wrapper
            )
            self._pregel_patched = True
            logger.debug("Patched Pregel.stream and Pregel.astream")
        except (ImportError, AttributeError) as exc:
            logger.debug(
                "Pregel class not available; stream patching skipped: %s", exc
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        self._uninstrument_create_react_agent()
        self._uninstrument_pregel()

    def _uninstrument_create_react_agent(self) -> None:
        for module_path, attr_name in self._patched_cra_locations:
            try:
                mod = importlib.import_module(module_path)
                unwrap(mod, attr_name)
                logger.debug("Restored %s.%s", module_path, attr_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Failed to restore %s.%s: %s", module_path, attr_name, exc
                )

        self._patched_cra_locations = []

    def _uninstrument_pregel(self) -> None:
        if not self._pregel_patched:
            return

        try:
            from langgraph.pregel import Pregel  # noqa: PLC0415

            unwrap(Pregel, "stream")
            unwrap(Pregel, "astream")
            logger.debug("Restored Pregel.stream and Pregel.astream")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to restore Pregel methods: %s", exc)

        self._pregel_patched = False
