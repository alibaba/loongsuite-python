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
LoongSuite DSPy instrumentation supporting ``dspy >= 3.0.0``.

Emits the DSPy framework layer of a GenAI trace — ENTRY, CHAIN, AGENT, STEP,
TOOL and RETRIEVER spans. LLM and EMBEDDING spans, and every token metric,
come from ``loongsuite-instrumentation-litellm``: DSPy routes all of its
built-in model and embedding calls through LiteLLM, so instrumenting them here
as well would duplicate spans and double-count tokens.

.. important::

   ``loongsuite-instrumentation-litellm`` must be enabled alongside this
   instrumentation. Without it a trace contains only the framework skeleton:
   no LLM spans and no token usage.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.dspy import DSPyInstrumentor
    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

    DSPyInstrumentor().instrument()
    LiteLLMInstrumentor().instrument()

    # ... use DSPy as normal ...

    DSPyInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from opentelemetry.instrumentation.dspy.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)

__all__ = ["DSPyInstrumentor"]


class DSPyInstrumentor(BaseInstrumentor):
    """An instrumentor for DSPy."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from opentelemetry.instrumentation.dspy.internal.callback import (  # noqa: PLC0415
            OTelDSPyCallback,
        )
        from opentelemetry.instrumentation.dspy.internal.patch import (  # noqa: PLC0415
            instrument_react_step,
        )
        from opentelemetry.instrumentation.dspy.internal.registration import (  # noqa: PLC0415
            register_callback,
        )
        from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
            ExtendedTelemetryHandler,
        )

        tracer_provider = kwargs.get("tracer_provider")

        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
        )

        register_callback(
            OTelDSPyCallback(handler=handler, tracer_provider=tracer_provider)
        )
        instrument_react_step()

    def _uninstrument(self, **kwargs: Any) -> None:
        from opentelemetry.instrumentation.dspy.internal.patch import (  # noqa: PLC0415
            uninstrument_react_step,
        )
        from opentelemetry.instrumentation.dspy.internal.registration import (  # noqa: PLC0415
            unregister_callback,
        )

        uninstrument_react_step()
        unregister_callback()
