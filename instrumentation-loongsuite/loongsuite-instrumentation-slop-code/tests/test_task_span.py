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

"""Tests for TASK span (AgentRunner._run_checkpoint)."""

from unittest.mock import MagicMock

import pytest

from opentelemetry.trace import StatusCode


class TestTaskSpan:
    """Verify that AgentRunner._run_checkpoint produces a TASK span."""

    def test_task_span_created(self, span_exporter, instrument):
        """_run_checkpoint should create a task span."""
        import slop_code.agent_runner.runner as mod

        runner = mod.AgentRunner()

        checkpoint = MagicMock()
        checkpoint.name = "checkpoint_1"
        checkpoint.order = 1

        runner._run_checkpoint(checkpoint, "/tmp/save", True)

        spans = span_exporter.get_finished_spans()
        task_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "run_task"
        ]
        assert len(task_spans) == 1

        span = task_spans[0]
        assert span.name == "run_task checkpoint_1"
        assert span.attributes["gen_ai.system"] == "slop-code"
        assert span.attributes["gen_ai.span.kind"] == "TASK"
        assert span.attributes["slop_code.checkpoint.name"] == "checkpoint_1"
        assert span.attributes["slop_code.checkpoint.order"] == 1
        assert span.attributes["slop_code.is_first_checkpoint"] is True
        assert span.status.status_code == StatusCode.OK

    def test_task_span_error(self, span_exporter, tracer_provider):
        """Exception in _run_checkpoint should produce an error task span."""
        import slop_code.agent_runner.runner as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        class FailingRunner(mod.AgentRunner):
            def _run_checkpoint(
                self,
                checkpoint,
                checkpoint_save_dir,
                is_first_checkpoint=False,
            ):
                raise RuntimeError("Checkpoint failed")

        # Replace class temporarily
        OriginalRunner = mod.AgentRunner
        mod.AgentRunner = FailingRunner

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            runner = mod.AgentRunner()
            checkpoint = MagicMock()
            checkpoint.name = "bad_checkpoint"
            checkpoint.order = 2

            with pytest.raises(RuntimeError, match="Checkpoint failed"):
                runner._run_checkpoint(checkpoint, "/tmp/save", False)

            spans = span_exporter.get_finished_spans()
            task_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.operation.name") == "run_task"
            ]
            assert len(task_spans) == 1
            assert task_spans[0].status.status_code == StatusCode.ERROR
        finally:
            instrumentor.uninstrument()
            mod.AgentRunner = OriginalRunner

    def test_task_span_not_first_checkpoint(self, span_exporter, instrument):
        """Subsequent checkpoint should have is_first_checkpoint=False."""
        import slop_code.agent_runner.runner as mod

        runner = mod.AgentRunner()

        checkpoint = MagicMock()
        checkpoint.name = "checkpoint_2"
        checkpoint.order = 2

        runner._run_checkpoint(checkpoint, "/tmp/save", False)

        spans = span_exporter.get_finished_spans()
        task_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "run_task"
        ]
        assert len(task_spans) == 1
        assert (
            task_spans[0].attributes["slop_code.is_first_checkpoint"] is False
        )
