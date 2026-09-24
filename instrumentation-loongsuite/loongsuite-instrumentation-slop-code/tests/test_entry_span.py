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

"""Tests for ENTRY span (run_agent and AgentRunner.run)."""

import pytest

from opentelemetry.trace import StatusCode


class TestEntrySpan:
    """Verify that run_agent produces an ENTRY span."""

    def test_entry_span_created(self, span_exporter, instrument):
        """run_agent should create an ENTRY span with correct attributes."""
        import slop_code.entrypoints.commands.run_agent as mod

        mod.run_agent()

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry_spans) == 1

        span = entry_spans[0]
        assert span.name == "enter_ai_application_system"
        assert span.attributes["gen_ai.system"] == "slop-code"
        assert span.attributes["gen_ai.operation.name"] == "enter"
        assert span.status.status_code == StatusCode.OK

    def test_entry_span_error(self, span_exporter, tracer_provider):
        """run_agent raising an exception should produce an error ENTRY span."""
        import slop_code.entrypoints.commands.run_agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        # Store original and replace with failing function
        original = mod.run_agent

        def failing_run_agent(*args, **kwargs):
            raise RuntimeError("Config error")

        mod.run_agent = failing_run_agent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            with pytest.raises(RuntimeError, match="Config error"):
                mod.run_agent()

            spans = span_exporter.get_finished_spans()
            entry_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "ENTRY"
            ]
            assert len(entry_spans) == 1
            assert entry_spans[0].status.status_code == StatusCode.ERROR
        finally:
            instrumentor.uninstrument()
            mod.run_agent = original

    def test_entry_span_with_problem_names_and_model(
        self, span_exporter, instrument
    ):
        """run_agent with problem_names and model_override kwargs should set input messages."""
        import slop_code.entrypoints.commands.run_agent as mod

        mod.run_agent(
            problem_names=["problem_a", "problem_b"], model_override="gpt-4"
        )

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry_spans) == 1
        span = entry_spans[0]
        assert "gen_ai.input.messages" in span.attributes
        msg = span.attributes["gen_ai.input.messages"]
        assert "problem_a" in msg
        assert "problem_b" in msg
        assert "gpt-4" in msg

    def test_entry_span_with_single_problem_string(
        self, span_exporter, instrument
    ):
        """run_agent with a single problem string should set input messages."""
        import slop_code.entrypoints.commands.run_agent as mod

        mod.run_agent(problem_names="single_problem")

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry_spans) == 1
        assert (
            "single_problem"
            in entry_spans[0].attributes["gen_ai.input.messages"]
        )


class TestRunnerEntrySpan:
    """Verify that AgentRunner.run produces an ENTRY span inside the worker."""

    def test_runner_entry_span_created(self, span_exporter, instrument):
        """AgentRunner.run should create an ENTRY span with problem context."""
        import slop_code.agent_runner.runner as mod

        runner = mod.AgentRunner()
        runner.run()

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry_spans) == 1

        span = entry_spans[0]
        assert span.name == "enter_ai_application_system"
        assert span.attributes["gen_ai.operation.name"] == "enter"
        assert span.attributes["gen_ai.session.id"] == "test_problem"
        assert "gen_ai.input.messages" in span.attributes
        assert (
            "Solve the coding problem"
            in span.attributes["gen_ai.input.messages"]
        )
        assert span.status.status_code == StatusCode.OK

    def test_runner_entry_span_captures_output(
        self, span_exporter, instrument
    ):
        """AgentRunner.run result should be captured as output."""
        import slop_code.agent_runner.runner as mod

        runner = mod.AgentRunner()
        result = runner.run()
        assert result == {"status": "completed"}

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry_spans) == 1
        span = entry_spans[0]
        assert "output.value" in span.attributes
        assert "completed" in span.attributes["output.value"]

    def test_runner_entry_span_error(self, span_exporter, tracer_provider):
        """Exception in AgentRunner.run should produce an error span."""
        import slop_code.agent_runner.runner as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class FailingRunner(mod.AgentRunner):
            def run(self):
                raise RuntimeError("Worker crashed")

        OriginalRunner = mod.AgentRunner
        mod.AgentRunner = FailingRunner

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            runner = mod.AgentRunner()
            with pytest.raises(RuntimeError, match="Worker crashed"):
                runner.run()

            spans = span_exporter.get_finished_spans()
            entry_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "ENTRY"
            ]
            assert len(entry_spans) == 1
            assert entry_spans[0].status.status_code == StatusCode.ERROR
        finally:
            instrumentor.uninstrument()
            mod.AgentRunner = OriginalRunner

    def test_runner_entry_span_fallback_prompt_sources(
        self, span_exporter, tracer_provider
    ):
        """RunnerEntryWrapper should try multiple prompt sources in order."""
        import slop_code.agent_runner.runner as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class RunnerNoPrompt(mod.AgentRunner):
            def __init__(self):
                super().__init__()
                # Remove prompt, add statement instead
                self.run_spec.problem.prompt = None
                self.run_spec.problem.statement = "Fix the following issue"
                self.run_spec.problem.description = None

        OriginalRunner = mod.AgentRunner
        mod.AgentRunner = RunnerNoPrompt

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            runner = mod.AgentRunner()
            runner.run()

            spans = span_exporter.get_finished_spans()
            entry_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "ENTRY"
            ]
            assert len(entry_spans) == 1
            assert "Fix the following issue" in entry_spans[0].attributes.get(
                "gen_ai.input.messages", ""
            )
        finally:
            instrumentor.uninstrument()
            mod.AgentRunner = OriginalRunner

    def test_runner_entry_span_template_fallback(
        self, span_exporter, tracer_provider
    ):
        """RunnerEntryWrapper should fall back to run_spec.template when no problem prompt."""
        import slop_code.agent_runner.runner as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class RunnerTemplateOnly(mod.AgentRunner):
            def __init__(self):
                super().__init__()
                self.run_spec.problem.prompt = None
                self.run_spec.problem.statement = None
                self.run_spec.problem.description = None
                self.run_spec.template = "Template task instructions"

        OriginalRunner = mod.AgentRunner
        mod.AgentRunner = RunnerTemplateOnly

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            runner = mod.AgentRunner()
            runner.run()

            spans = span_exporter.get_finished_spans()
            entry_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "ENTRY"
            ]
            assert len(entry_spans) == 1
            assert "Template task instructions" in entry_spans[
                0
            ].attributes.get("gen_ai.input.messages", "")
        finally:
            instrumentor.uninstrument()
            mod.AgentRunner = OriginalRunner
