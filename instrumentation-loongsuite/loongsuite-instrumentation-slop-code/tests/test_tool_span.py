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

"""Tests for TOOL span (MiniSWEAgent.execute_action)."""

import pytest

from opentelemetry.trace import StatusCode


class TestToolSpan:
    """Verify that MiniSWEAgent.execute_action produces a TOOL span."""

    def test_tool_span_created(self, span_exporter, instrument):
        """execute_action should create a TOOL span with correct attributes."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        agent = mod.MiniSWEAgent(problem_name="test_prob")
        agent.execute_action({"action": "ls -la", "thought": "List files"})

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert len(tool_spans) == 1

        span = tool_spans[0]
        assert span.name == "execute_tool bash"
        assert span.attributes["gen_ai.system"] == "slop-code"
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == "bash"
        assert span.attributes["gen_ai.tool.type"] == "function"
        assert "gen_ai.tool.call.id" in span.attributes
        assert "ls -la" in span.attributes["gen_ai.tool.call.arguments"]
        assert "gen_ai.tool.call.result" in span.attributes
        assert span.status.status_code == StatusCode.OK

    def test_tool_span_with_string_action(self, span_exporter, instrument):
        """execute_action with a string action should work."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        agent = mod.MiniSWEAgent(problem_name="test_prob")
        agent.execute_action("echo hello")

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert len(tool_spans) == 1
        assert (
            "echo hello"
            in tool_spans[0].attributes["gen_ai.tool.call.arguments"]
        )

    def test_tool_span_error(self, span_exporter, tracer_provider):
        """Exception in execute_action should produce an error TOOL span."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class FailingToolAgent(mod.MiniSWEAgent):
            def execute_action(self, action):
                raise PermissionError("Command not allowed")

        OriginalClass = mod.MiniSWEAgent
        mod.MiniSWEAgent = FailingToolAgent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.MiniSWEAgent(problem_name="test_prob")

            with pytest.raises(PermissionError, match="Command not allowed"):
                agent.execute_action({"action": "rm -rf /"})

            spans = span_exporter.get_finished_spans()
            tool_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "TOOL"
            ]
            assert len(tool_spans) == 1
            span = tool_spans[0]
            assert span.status.status_code == StatusCode.ERROR
            assert span.attributes["error.type"] == "PermissionError"
            assert "gen_ai.tool.call.result" in span.attributes
        finally:
            instrumentor.uninstrument()
            mod.MiniSWEAgent = OriginalClass
