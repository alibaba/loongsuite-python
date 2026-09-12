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

import pytest

from opentelemetry.trace import StatusCode


class TestAgentSpan:
    """Verify that Agent.run_checkpoint produces an AGENT span."""

    def test_agent_span_created(self, span_exporter, instrument):
        """Agent.run_checkpoint should create an AGENT span."""
        import slop_code.agent_runner.agent as mod

        agent = mod.Agent(problem_name="file_backup")
        agent.run_checkpoint("solve the bug")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s
            for s in spans
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
            s
            for s in spans
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

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class FailingAgent(mod.Agent):
            def run_checkpoint(self, task):
                raise TimeoutError("Agent timeout")

        OriginalAgent = mod.Agent
        mod.Agent = FailingAgent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.Agent(problem_name="test_prob")

            with pytest.raises(TimeoutError, match="Agent timeout"):
                agent.run_checkpoint("task")

            spans = span_exporter.get_finished_spans()
            agent_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
            ]
            assert len(agent_spans) == 1
            span = agent_spans[0]
            assert span.status.status_code == StatusCode.ERROR
            assert span.attributes.get("error.type") == "TimeoutError"
        finally:
            instrumentor.uninstrument()
            mod.Agent = OriginalAgent

    def test_agent_span_with_messages_attr(
        self, span_exporter, tracer_provider
    ):
        """Agent with _messages should capture assistant output messages."""
        import slop_code.agent_runner.agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class AgentWithMessages(mod.Agent):
            def __init__(self, problem_name="test"):
                super().__init__(problem_name)
                self.system_template = None
                self.system_prompt = "You are a helpful assistant"
                self._messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant",
                    },
                    {"role": "user", "content": "Fix the bug"},
                    {
                        "role": "assistant",
                        "content": "I found the issue in line 42",
                    },
                ]

        OriginalAgent = mod.Agent
        mod.Agent = AgentWithMessages

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.Agent(problem_name="test_prob")
            agent.run_checkpoint("task")

            spans = span_exporter.get_finished_spans()
            agent_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
            ]
            assert len(agent_spans) == 1
            span = agent_spans[0]

            # Should capture output messages from _messages
            assert "gen_ai.output.messages" in span.attributes
            assert "line 42" in span.attributes["gen_ai.output.messages"]

            # Should use system_prompt as fallback for system instructions
            assert "gen_ai.system.instructions" in span.attributes
            assert (
                "helpful assistant"
                in span.attributes["gen_ai.system.instructions"]
            )
        finally:
            instrumentor.uninstrument()
            mod.Agent = OriginalAgent

    def test_agent_span_with_steps_attr(self, span_exporter, tracer_provider):
        """Agent with _steps should capture assistant output messages from steps."""
        import slop_code.agent_runner.agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class StepRole:
            def __init__(self, value):
                self.value = value

        class Step:
            def __init__(self, role, content):
                self.role = StepRole(role)
                self.content = content

        class AgentWithSteps(mod.Agent):
            def __init__(self, problem_name="test"):
                super().__init__(problem_name)
                self.system_template = None
                self.system_prompt = None
                self._steps = [
                    Step("system", "You are a system agent"),
                    Step("user", "Solve the problem"),
                    Step("assistant", "I will solve it now"),
                ]
                self._messages = []

        OriginalAgent = mod.Agent
        mod.Agent = AgentWithSteps

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.Agent(problem_name="test_prob")
            agent.run_checkpoint("task")

            spans = span_exporter.get_finished_spans()
            agent_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
            ]
            assert len(agent_spans) == 1
            span = agent_spans[0]

            # Should capture output from _steps
            assert "gen_ai.output.messages" in span.attributes
            assert "solve it now" in span.attributes["gen_ai.output.messages"]

            # Should extract system prompt from _steps
            assert "gen_ai.system.instructions" in span.attributes
            assert (
                "system agent" in span.attributes["gen_ai.system.instructions"]
            )
        finally:
            instrumentor.uninstrument()
            mod.Agent = OriginalAgent

    def test_agent_span_system_from_messages(
        self, span_exporter, tracer_provider
    ):
        """Agent with _messages containing system role should extract system prompt."""
        import slop_code.agent_runner.agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class AgentSysMsgs(mod.Agent):
            def __init__(self, problem_name="test"):
                super().__init__(problem_name)
                self.system_template = None
                self.system_prompt = None
                self._steps = []
                self._messages = [
                    {
                        "role": "system",
                        "content": "System context from messages",
                    },
                    {"role": "user", "content": "Help me"},
                ]

        OriginalAgent = mod.Agent
        mod.Agent = AgentSysMsgs

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.Agent(problem_name="test_prob")
            agent.run_checkpoint("task")

            spans = span_exporter.get_finished_spans()
            agent_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
            ]
            assert len(agent_spans) == 1
            span = agent_spans[0]
            assert "gen_ai.system.instructions" in span.attributes
            assert (
                "System context from messages"
                in span.attributes["gen_ai.system.instructions"]
            )
        finally:
            instrumentor.uninstrument()
            mod.Agent = OriginalAgent
