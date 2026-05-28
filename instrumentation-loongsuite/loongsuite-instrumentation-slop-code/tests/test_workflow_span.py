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

"""Tests for CHAIN/workflow span (run_agent_on_problem)."""

from unittest.mock import MagicMock

import pytest

from opentelemetry.trace import StatusCode


class TestWorkflowSpan:
    """Verify that run_agent_on_problem produces a workflow span."""

    def test_workflow_span_created(self, span_exporter, instrument):
        """run_agent_on_problem should create a workflow span."""
        import slop_code.entrypoints.problem_runner.worker as mod

        config = MagicMock()
        config.model_def = MagicMock()
        config.model_def.name = "anthropic/claude-3.5-sonnet"
        config.agent_config = MagicMock()
        config.agent_config.type = "claude_code"
        config.pass_policy = MagicMock()
        config.pass_policy.value = "any"

        mod.run_agent_on_problem(
            MagicMock(),  # problem_config
            "file_backup",  # problem_name
            config,  # config
            MagicMock(),  # progress_queue
            "/tmp/output",  # output_path
        )

        spans = span_exporter.get_finished_spans()
        workflow_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "workflow"
        ]
        assert len(workflow_spans) == 1

        span = workflow_spans[0]
        assert span.name == "chain file_backup"
        assert span.attributes["gen_ai.system"] == "slop-code"
        assert span.attributes["gen_ai.span.kind"] == "CHAIN"
        assert span.attributes["slop_code.problem.name"] == "file_backup"
        assert (
            span.attributes["gen_ai.request.model"]
            == "anthropic/claude-3.5-sonnet"
        )
        assert span.attributes["slop_code.agent.type"] == "claude_code"
        assert span.status.status_code == StatusCode.OK

    def test_workflow_span_error(self, span_exporter, tracer_provider):
        """Exception in run_agent_on_problem should produce error workflow span."""
        import slop_code.entrypoints.problem_runner.worker as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        original = mod.run_agent_on_problem

        def failing_worker(*args, **kwargs):
            raise ValueError("Problem not found")

        mod.run_agent_on_problem = failing_worker

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            with pytest.raises(ValueError, match="Problem not found"):
                mod.run_agent_on_problem(
                    MagicMock(),
                    "missing_problem",
                    MagicMock(),
                    MagicMock(),
                    "/tmp",
                )

            spans = span_exporter.get_finished_spans()
            workflow_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "workflow"
            ]
            assert len(workflow_spans) == 1
            assert workflow_spans[0].status.status_code == StatusCode.ERROR
        finally:
            instrumentor.uninstrument()
            mod.run_agent_on_problem = original

    def test_workflow_span_with_none_config_fields(
        self, span_exporter, instrument
    ):
        """Workflow span should handle None config fields gracefully."""
        import slop_code.entrypoints.problem_runner.worker as mod

        config = MagicMock()
        config.model_def = None
        config.agent_config = None
        config.pass_policy = None

        mod.run_agent_on_problem(
            MagicMock(), "test_problem", config, MagicMock(), "/tmp"
        )

        spans = span_exporter.get_finished_spans()
        workflow_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "workflow"
        ]
        assert len(workflow_spans) == 1
        span = workflow_spans[0]
        assert span.attributes["slop_code.problem.name"] == "test_problem"
        assert "gen_ai.request.model" not in span.attributes

    def test_workflow_span_pass_policy_with_value(
        self, span_exporter, instrument
    ):
        """Workflow span should extract pass_policy.value from enum-like objects."""
        import slop_code.entrypoints.problem_runner.worker as mod

        config = MagicMock()
        config.model_def = None
        config.agent_config = None

        class PolicyEnum:
            def __init__(self):
                self.value = "majority"

        config.pass_policy = PolicyEnum()

        mod.run_agent_on_problem(
            MagicMock(), "policy_test", config, MagicMock(), "/tmp"
        )

        spans = span_exporter.get_finished_spans()
        workflow_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "workflow"
        ]
        assert len(workflow_spans) == 1
        assert (
            workflow_spans[0].attributes["slop_code.pass_policy"] == "majority"
        )

    def test_workflow_span_none_config(self, span_exporter, instrument):
        """Workflow span should handle None config entirely."""
        import slop_code.entrypoints.problem_runner.worker as mod

        mod.run_agent_on_problem(
            MagicMock(), "no_config_problem", None, MagicMock(), "/tmp"
        )

        spans = span_exporter.get_finished_spans()
        workflow_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "workflow"
        ]
        assert len(workflow_spans) == 1
        assert (
            workflow_spans[0].attributes["slop_code.problem.name"]
            == "no_config_problem"
        )
