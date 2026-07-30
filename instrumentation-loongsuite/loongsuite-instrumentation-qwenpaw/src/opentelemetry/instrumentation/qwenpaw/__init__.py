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
LoongSuite QwenPaw instrumentation.

Instruments QwenPaw 1 ``AgentRunner.query_handler`` and QwenPaw 2
``Runtime.run`` with ``ExtendedTelemetryHandler.entry``
(``enter_ai_application_system``). Agent / tool / LLM spans come from AgentScope
and other instrumentations.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.qwenpaw import QwenPawInstrumentor

    QwenPawInstrumentor().instrument()
    # ... run QwenPaw app ...
    QwenPawInstrumentor().uninstrument()
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.qwenpaw.package import (
    _instruments_any,
    get_installed_instrumentation_dependencies,
    get_installed_runtime_targets,
)
from opentelemetry.instrumentation.qwenpaw.patch import (
    make_query_handler_wrapper,
    make_runtime_wrapper,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

__all__ = ["QwenPawInstrumentor", "CoPawInstrumentor"]


class QwenPawInstrumentor(BaseInstrumentor):
    """LoongSuite instrumentor for QwenPaw Entry telemetry."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: ExtendedTelemetryHandler | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        installed = get_installed_instrumentation_dependencies()
        return installed or _instruments_any

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        runtime_targets = tuple(get_installed_runtime_targets())
        if not runtime_targets:
            raise ModuleNotFoundError(
                "No supported QwenPaw runtime package is installed"
            )

        for target in runtime_targets:
            wrapper_factory = (
                make_runtime_wrapper
                if target.wrapper_kind == "runtime"
                else make_query_handler_wrapper
            )
            wrap_function_wrapper(
                target.module_name,
                f"{target.class_name}.{target.method_name}",
                wrapper_factory(self._handler, target.module_name),
            )
            logger.debug(
                "Instrumented %s.%s.%s",
                target.module_name,
                target.class_name,
                target.method_name,
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        self._handler = None
        for target in get_installed_runtime_targets():
            try:
                runtime_module = import_module(target.module_name)
                runtime_class = getattr(runtime_module, target.class_name)
                unwrap(runtime_class, target.method_name)
                logger.debug(
                    "Uninstrumented %s.%s.%s",
                    target.module_name,
                    target.class_name,
                    target.method_name,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to uninstrument %s.%s.%s: %s",
                    target.module_name,
                    target.class_name,
                    target.method_name,
                    exc,
                )


CoPawInstrumentor = QwenPawInstrumentor
