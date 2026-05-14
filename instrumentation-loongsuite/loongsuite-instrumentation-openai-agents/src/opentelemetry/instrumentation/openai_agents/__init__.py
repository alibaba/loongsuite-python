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
OpenTelemetry OpenAI Agents SDK Instrumentation
================================================

This package provides automatic instrumentation for the OpenAI Agents SDK,
capturing telemetry data for agent runs, tool executions, LLM generations,
handoffs, and guardrails.

Usage
-----

Basic instrumentation::

    from opentelemetry.instrumentation.openai_agents import (
        OpenAIAgentsInstrumentor,
    )

    OpenAIAgentsInstrumentor().instrument()

    from agents import Agent, Runner

    agent = Agent(name="assistant", instructions="You are helpful.")
    result = Runner.run_sync(agent, "Hello!")

The instrumentation leverages the SDK's built-in ``TracingProcessor``
interface to register an OpenTelemetry bridge, so all native SDK tracing
points are automatically captured without monkey-patching.
"""

import logging
import os
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.package import (
    _instruments,
)
from opentelemetry.instrumentation.openai_agents.version import (
    __version__,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
)

logger = logging.getLogger(__name__)

_ENV_CAPTURE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def _should_capture_content() -> bool:
    val = os.environ.get(_ENV_CAPTURE_CONTENT, "").strip().lower()
    return val not in ("false", "0", "no", "off", "")


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """Instrumentor for the OpenAI Agents SDK.

    Registers an OpenTelemetry-aware ``TracingProcessor`` with the
    SDK's global trace provider so that every agent run, tool call,
    LLM generation, handoff, and guardrail execution is automatically
    exported as an OTel span.
    """

    _handler: Optional[ExtendedTelemetryHandler] = None
    _processor: Optional[Any] = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        OpenAIAgentsInstrumentor._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        capture_content = _should_capture_content()

        from agents.tracing import (  # noqa: PLC0415
            add_trace_processor,
        )

        from opentelemetry.instrumentation.openai_agents._processor import (  # noqa: PLC0415
            OTelTracingProcessor,
        )

        processor = OTelTracingProcessor(
            handler=OpenAIAgentsInstrumentor._handler,
            capture_content=capture_content,
        )
        OpenAIAgentsInstrumentor._processor = processor
        add_trace_processor(processor)

    def _uninstrument(self, **kwargs: Any) -> None:
        processor = OpenAIAgentsInstrumentor._processor
        if processor is None:
            return

        try:
            from agents.tracing.setup import (  # noqa: PLC0415
                get_trace_provider,
            )

            provider = get_trace_provider()
            if hasattr(provider, "_multi_processor"):
                mp = provider._multi_processor
                if hasattr(mp, "_processors"):
                    procs = mp._processors
                    if processor in procs:
                        procs.remove(processor)
        except Exception as e:
            logger.debug("Failed to remove processor: %s", e)

        processor.shutdown()
        OpenAIAgentsInstrumentor._processor = None
        OpenAIAgentsInstrumentor._handler = None


__all__ = [
    "__version__",
    "OpenAIAgentsInstrumentor",
]
