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
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.package import (
    _instruments,
)
from opentelemetry.instrumentation.openai_agents.version import (
    __version__,
)
from opentelemetry.trace import get_tracer
from opentelemetry.util.genai._configuration import is_experimental_mode
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import get_content_capturing_mode

logger = logging.getLogger(__name__)


def _should_capture_content() -> bool:
    return is_experimental_mode() and get_content_capturing_mode() in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """Instrumentor for the OpenAI Agents SDK.

    Registers an OpenTelemetry-aware ``TracingProcessor`` with the
    SDK's global trace provider so that every agent run, tool call,
    LLM generation, handoff, and guardrail execution is automatically
    exported as an OTel span.
    """

    _provider: Optional[Any] = None
    _processor: Optional[Any] = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        # The target SDK is optional until instrumentation is enabled.
        from agents.tracing import setup  # noqa: PLC0415

        from opentelemetry.instrumentation.openai_agents._processor import (  # noqa: PLC0415
            OTelTracingProcessor,
        )

        provider = (
            setup.get_trace_provider()
            if hasattr(setup, "get_trace_provider")
            else setup.GLOBAL_TRACE_PROVIDER
        )
        tracer = get_tracer(__name__, __version__, tracer_provider)
        processor = OTelTracingProcessor(tracer, _should_capture_content())
        provider.register_processor(processor)
        OpenAIAgentsInstrumentor._provider = provider
        OpenAIAgentsInstrumentor._processor = processor

    def _uninstrument(self, **kwargs: Any) -> None:
        processor = OpenAIAgentsInstrumentor._processor
        provider = OpenAIAgentsInstrumentor._provider
        if processor is None or provider is None:
            return

        # The SDK has no public remove API. Its default dispatcher uses a
        # locked tuple on both the oldest and latest supported releases.
        dispatcher = getattr(provider, "_multi_processor", None)
        if dispatcher is not None:
            with dispatcher._lock:
                dispatcher._processors = tuple(
                    item
                    for item in dispatcher._processors
                    if item is not processor
                )
        processor.shutdown()
        OpenAIAgentsInstrumentor._processor = None
        OpenAIAgentsInstrumentor._provider = None


__all__ = [
    "__version__",
    "OpenAIAgentsInstrumentor",
]
