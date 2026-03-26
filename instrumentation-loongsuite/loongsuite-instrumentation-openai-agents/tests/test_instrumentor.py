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

"""Tests for the OpenAIAgentsInstrumentor lifecycle."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tracing.span_data import AgentSpanData
from tests.test_processor import FakeSpan, FakeTrace

from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.openai_agents._processor import (
    OTelTracingProcessor,
)


class TestInstrumentor:
    def test_instrument_registers_processor(self, instrument):
        assert OpenAIAgentsInstrumentor._processor is not None
        assert isinstance(
            OpenAIAgentsInstrumentor._processor,
            OTelTracingProcessor,
        )

    def test_uninstrument_clears_state(self, tracer_provider):
        instrumentor = OpenAIAgentsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        assert OpenAIAgentsInstrumentor._processor is not None

        instrumentor.uninstrument()
        assert OpenAIAgentsInstrumentor._processor is None
        assert OpenAIAgentsInstrumentor._handler is None

    def test_end_to_end_with_instrumentor(self, instrument, span_exporter):
        """Simulate an agent run via the processor
        registered by the instrumentor."""
        processor = OpenAIAgentsInstrumentor._processor

        trace = FakeTrace(name="E2E Test")
        processor.on_trace_start(trace)

        agent_data = AgentSpanData(name="test_agent")
        sdk_span = FakeSpan(
            agent_data,
            span_id="e2e_agent_001",
            trace_id=trace.trace_id,
        )
        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = span_exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "invoke_workflow E2E Test" in span_names
        assert "invoke_agent test_agent" in span_names

    def test_dependencies(self):
        instrumentor = OpenAIAgentsInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert any("openai-agents" in d for d in deps)
