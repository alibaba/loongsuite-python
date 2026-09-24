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

"""Tests for span hierarchy and parent-child relationships."""

from unittest.mock import MagicMock


class TestSpanHierarchy:
    """Verify parent-child relationships between spans."""

    def test_entry_is_parent_of_workflow(self, span_exporter, instrument):
        """ENTRY span should be parent of workflow span when called inline."""
        import slop_code.entrypoints.commands.run_agent as entry_mod
        import slop_code.entrypoints.problem_runner.worker as worker_mod

        # Patch run_agent to call run_agent_on_problem internally
        original = entry_mod.run_agent.__wrapped__

        def run_with_workflow(*args, **kwargs):
            config = MagicMock()
            config.model_def = None
            config.agent_config = None
            config.pass_policy = None
            return worker_mod.run_agent_on_problem(
                MagicMock(), "test_problem", config, MagicMock(), "/tmp"
            )

        entry_mod.run_agent.__wrapped__ = run_with_workflow

        try:
            entry_mod.run_agent()

            spans = span_exporter.get_finished_spans()
            entry_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "ENTRY"
            ]
            workflow_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "workflow"
            ]

            assert len(entry_spans) == 1
            assert len(workflow_spans) == 1

            entry_span = entry_spans[0]
            workflow_span = workflow_spans[0]

            # workflow should be child of entry
            assert (
                workflow_span.context.trace_id == entry_span.context.trace_id
            )
            assert workflow_span.parent is not None
            assert workflow_span.parent.span_id == entry_span.context.span_id
        finally:
            entry_mod.run_agent.__wrapped__ = original

    def test_workflow_is_parent_of_task(self, span_exporter, instrument):
        """Task spans within a workflow should share the workflow's trace.

        _TaskRunCheckpointWrapper creates ENTRY -> TASK nested spans.
        So the hierarchy is: CHAIN -> ENTRY -> TASK.
        """
        import slop_code.agent_runner.runner as runner_mod
        import slop_code.entrypoints.problem_runner.worker as worker_mod

        original = worker_mod.run_agent_on_problem.__wrapped__

        def workflow_with_task(*args, **kwargs):
            r = runner_mod.AgentRunner()
            checkpoint = MagicMock()
            checkpoint.name = "cp1"
            checkpoint.order = 1
            r._run_checkpoint(checkpoint, "/tmp", True)
            return {"summary": {"state": "completed", "passed_policy": True}}

        worker_mod.run_agent_on_problem.__wrapped__ = workflow_with_task

        try:
            config = MagicMock()
            config.model_def = None
            config.agent_config = None
            config.pass_policy = None
            worker_mod.run_agent_on_problem(
                MagicMock(), "prob1", config, MagicMock(), "/tmp"
            )

            spans = span_exporter.get_finished_spans()
            workflow_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "workflow"
            ]
            task_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "run_task"
            ]
            # _TaskRunCheckpointWrapper creates an inner ENTRY span too
            inner_entry_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "enter"
            ]

            assert len(workflow_spans) == 1
            assert len(task_spans) == 1
            assert len(inner_entry_spans) == 1

            workflow_span = workflow_spans[0]
            inner_entry_span = inner_entry_spans[0]
            task_span = task_spans[0]

            # All spans share the same trace
            assert task_span.context.trace_id == workflow_span.context.trace_id
            assert (
                inner_entry_span.context.trace_id
                == workflow_span.context.trace_id
            )

            # inner_entry is child of workflow
            assert inner_entry_span.parent is not None
            assert (
                inner_entry_span.parent.span_id
                == workflow_span.context.span_id
            )

            # task is child of inner_entry
            assert task_span.parent is not None
            assert task_span.parent.span_id == inner_entry_span.context.span_id
        finally:
            worker_mod.run_agent_on_problem.__wrapped__ = original
