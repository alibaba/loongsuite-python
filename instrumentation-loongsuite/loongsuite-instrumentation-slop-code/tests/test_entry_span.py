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

"""Tests for ENTRY span (run_agent)."""

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
            s for s in spans
            if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry_spans) == 1

        span = entry_spans[0]
        assert span.name == "slop-code.enter"
        assert span.attributes["gen_ai.system"] == "slop-code"
        assert span.attributes["gen_ai.operation.name"] == "enter"
        assert span.status.status_code == StatusCode.OK

    def test_entry_span_error(self, span_exporter, tracer_provider):
        """run_agent raising an exception should produce an error ENTRY span."""
        import slop_code.entrypoints.commands.run_agent as mod

        from opentelemetry.instrumentation.slop_code import SlopCodeInstrumentor

        # Store original and replace with failing function
        original = mod.run_agent

        def failing_run_agent(*args, **kwargs):
            raise RuntimeError("Config error")

        mod.run_agent = failing_run_agent

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            with pytest.raises(RuntimeError, match="Config error"):
                mod.run_agent()

            spans = span_exporter.get_finished_spans()
            entry_spans = [
                s for s in spans
                if s.attributes.get("gen_ai.span.kind") == "ENTRY"
            ]
            assert len(entry_spans) == 1
            assert entry_spans[0].status.status_code == StatusCode.ERROR
        finally:
            instrumentor.uninstrument()
            mod.run_agent = original
