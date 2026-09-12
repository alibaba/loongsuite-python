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

"""Tests for STEP span (MiniSWEAgent.agent_step)."""

import pytest

from opentelemetry.trace import StatusCode


class TestStepSpan:
    """Verify that MiniSWEAgent.agent_step produces a STEP span."""

    def test_step_span_created(self, span_exporter, instrument):
        """agent_step should create a STEP span with token attributes."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        agent = mod.MiniSWEAgent(problem_name="test_prob")
        agent.agent_step()
        # The step span stays open until get_observation finishes it
        agent.get_observation()

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step_spans) == 1

        span = step_spans[0]
        assert span.name == "react step"
        assert span.attributes["gen_ai.system"] == "slop-code"
        assert span.attributes["gen_ai.operation.name"] == "react"
        assert span.attributes["gen_ai.react.round"] == 1
        assert span.status.status_code == StatusCode.OK

    def test_step_span_has_token_usage(self, span_exporter, instrument):
        """STEP span should capture token usage from result."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        agent = mod.MiniSWEAgent(problem_name="test_prob")
        agent.agent_step()
        agent.get_observation()

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step_spans) == 1
        span = step_spans[0]

        assert span.attributes["gen_ai.usage.input_tokens"] == 200
        assert span.attributes["gen_ai.usage.output_tokens"] == 80
        assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 50
        assert (
            span.attributes["gen_ai.usage.cache_creation.input_tokens"] == 10
        )

    def test_step_span_increments_round(self, span_exporter, instrument):
        """Multiple agent_step calls should increment the round number."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        agent = mod.MiniSWEAgent(problem_name="test_prob")
        # Simulate steps=2 already completed
        agent.usage.steps = 2
        agent.agent_step()
        agent.get_observation()

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step_spans) == 1
        assert step_spans[0].name == "react step"
        assert step_spans[0].attributes["gen_ai.react.round"] == 3

    def test_step_span_error(self, span_exporter, tracer_provider):
        """Exception in agent_step should produce an error STEP span."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class FailingMiniSWE(mod.MiniSWEAgent):
            def agent_step(self):
                raise RuntimeError("LimitsExceeded")

        OriginalClass = mod.MiniSWEAgent
        mod.MiniSWEAgent = FailingMiniSWE

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.MiniSWEAgent(problem_name="test_prob")

            with pytest.raises(RuntimeError, match="LimitsExceeded"):
                agent.agent_step()

            spans = span_exporter.get_finished_spans()
            step_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "STEP"
            ]
            assert len(step_spans) == 1
            span = step_spans[0]
            assert span.status.status_code == StatusCode.ERROR
            assert span.attributes["gen_ai.react.finish_reason"] == "error"
        finally:
            instrumentor.uninstrument()
            mod.MiniSWEAgent = OriginalClass

    def test_step_span_finish_reason_stop(self, span_exporter, instrument):
        """Successful step should have finish_reason='stop'."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        agent = mod.MiniSWEAgent(problem_name="test_prob")
        agent.agent_step()
        agent.get_observation()

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert step_spans[0].attributes["gen_ai.react.finish_reason"] == "stop"

    def test_step_span_none_result_finishes_immediately(
        self, span_exporter, tracer_provider
    ):
        """agent_step returning None should finish the span immediately with stop."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class NoneStepAgent(mod.MiniSWEAgent):
            def agent_step(self):
                return None

        OriginalClass = mod.MiniSWEAgent
        mod.MiniSWEAgent = NoneStepAgent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.MiniSWEAgent(problem_name="test_prob")
            result = agent.agent_step()
            assert result is None

            spans = span_exporter.get_finished_spans()
            step_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "STEP"
            ]
            assert len(step_spans) == 1
            assert (
                step_spans[0].attributes["gen_ai.react.finish_reason"]
                == "stop"
            )
            assert step_spans[0].status.status_code == StatusCode.OK
        finally:
            instrumentor.uninstrument()
            mod.MiniSWEAgent = OriginalClass

    def test_step_span_output_messages(self, span_exporter, instrument):
        """STEP span should capture output messages from result content."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        agent = mod.MiniSWEAgent(problem_name="test_prob")
        agent.agent_step()
        agent.get_observation()

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step_spans) == 1
        span = step_spans[0]
        assert "gen_ai.output.messages" in span.attributes
        assert "fix this bug" in span.attributes["gen_ai.output.messages"]

    def test_observation_error_finishes_step(
        self, span_exporter, tracer_provider
    ):
        """Exception in get_observation should finish the step span with error."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class ErrorObsAgent(mod.MiniSWEAgent):
            def get_observation(self):
                raise IOError("Environment error")

        OriginalClass = mod.MiniSWEAgent
        mod.MiniSWEAgent = ErrorObsAgent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.MiniSWEAgent(problem_name="test_prob")
            agent.agent_step()

            with pytest.raises(IOError, match="Environment error"):
                agent.get_observation()

            spans = span_exporter.get_finished_spans()
            step_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "STEP"
            ]
            assert len(step_spans) == 1
            assert step_spans[0].status.status_code == StatusCode.ERROR
            assert (
                step_spans[0].attributes["gen_ai.react.finish_reason"]
                == "error"
            )
        finally:
            instrumentor.uninstrument()
            mod.MiniSWEAgent = OriginalClass

    def test_step_span_no_token_usage_in_result(
        self, span_exporter, tracer_provider
    ):
        """Step with result dict but no token_usage should estimate tokens."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class NoTokenAgent(mod.MiniSWEAgent):
            def agent_step(self):
                return {
                    "content": "I will fix the bug.",
                    "step_cost": 0.005,
                }

        OriginalClass = mod.MiniSWEAgent
        mod.MiniSWEAgent = NoTokenAgent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.MiniSWEAgent(problem_name="test_prob")
            agent.agent_step()
            agent.get_observation()

            spans = span_exporter.get_finished_spans()
            step_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "STEP"
            ]
            assert len(step_spans) == 1
            span = step_spans[0]
            # Tokens should be estimated (non-zero)
            assert span.attributes["gen_ai.usage.input_tokens"] > 0
            assert span.attributes["gen_ai.usage.output_tokens"] > 0
        finally:
            instrumentor.uninstrument()
            mod.MiniSWEAgent = OriginalClass

    def test_step_span_empty_content(self, span_exporter, tracer_provider):
        """Step with empty string content should estimate 0 output tokens."""
        import slop_code.agent_runner.agents._miniswe_agent as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class EmptyContentAgent(mod.MiniSWEAgent):
            def agent_step(self):
                return {
                    "content": "",
                    "step_cost": 0.0,
                }

        OriginalClass = mod.MiniSWEAgent
        mod.MiniSWEAgent = EmptyContentAgent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = mod.MiniSWEAgent(problem_name="test_prob")
            agent.agent_step()
            agent.get_observation()

            spans = span_exporter.get_finished_spans()
            step_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "STEP"
            ]
            assert len(step_spans) == 1
            span = step_spans[0]
            # Empty content: _estimate_tokens("") returns 0
            assert span.attributes["gen_ai.usage.output_tokens"] == 0
        finally:
            instrumentor.uninstrument()
            mod.MiniSWEAgent = OriginalClass
