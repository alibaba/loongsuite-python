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

"""Tests for AGENT span (Agent.run_checkpoint)."""

from unittest.mock import MagicMock

import pytest

from opentelemetry.trace import StatusCode


class TestAgentSpan:
    """Verify that Agent.run_checkpoint produces an AGENT span."""

    def test_agent_span_created(self, span_exporter, instrument):
        """Agent.run_checkpoint should create an AGENT span."""
        import slop_code.agent_runner.agent as mod

        agent = mod.Agent(problem_name="file_backup")
        result = agent.run_checkpoint("solve the bug")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans
            if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        assert len(agent_spans) == 1

        span = agent_spans[0]
        assert span.name == "invoke_agent Agent"
        assert span.attributes["gen_ai.system"] == "slop-code"
        assert span.attributes["gen_ai.span.kind"] == "AGENT"
        assert span.attributes["gen_ai.agent.name"] == "Agent"
        assert span.attributes["slop_code.problem.name"] == "file_backup"
        assert span.status.status_code == StatusCode.OK

        assert "gen_ai.input.messages" in span.attributes
        assert "solve the bug" in span.attributes["gen_ai.input.messages"]

        assert "gen_ai.system.instructions" in span.attributes
        assert "coding agent" in span.attributes["gen_ai.system.instructions"]

    def test_agent_span_captures_usage(self, span_exporter, instrument):
        """AGENT span should capture token usage from result."""
        import slop_code.agent_runner.agent as mod

        agent = mod.Agent(problem_name="test_prob")
        agent.run_checkpoint("task")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans
            if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        assert len(agent_spans) == 1
        span = agent_spans[0]

        assert "gen_ai.usage.input_tokens" in span.attributes
        assert "gen_ai.usage.output_tokens" in span.attributes
        assert span.attributes["gen_ai.usage.input_tokens"] == 100
        assert span.attributes["gen_ai.usage.output_tokens"] == 50

    def test_agent_span_error(self, span_exporter, tracer_provider):
        """Exception in Agent.run_checkpoint should produce error span."""
        import slop_code.agent_runner.agent as mod

        from opentelemetry.instrumentation.slop_code import SlopCodeInstrumentor

        class FailingAgent(mod.Agent):
            def run_checkpoint(self, task):
                raise TimeoutError("Agent timeout")

        OriginalAgent = mod.Agent
        mod.Agent = FailingAgent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            agent = mod.Agent(problem_name="test_prob")

            with pytest.raises(TimeoutError, match="Agent timeout"):
                agent.run_checkpoint("task")

            spans = span_exporter.get_finished_spans()
            agent_spans = [
                s for s in spans
                if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
            ]
            assert len(agent_spans) == 1
            span = agent_spans[0]
            assert span.status.status_code == StatusCode.ERROR
            assert span.attributes.get("error.type") == "TimeoutError"
        finally:
            instrumentor.uninstrument()
            mod.Agent = OriginalAgent
