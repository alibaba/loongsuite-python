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

"""Tests for LLM span (grade_file_async - Rubric Judge)."""

from unittest.mock import MagicMock

import pytest

from opentelemetry.trace import SpanKind, StatusCode


@pytest.mark.asyncio
class TestLLMSpan:
    """Verify that grade_file_async produces an LLM span."""

    async def test_llm_span_created(self, span_exporter, instrument):
        """grade_file_async should create an LLM span."""
        import slop_code.metrics.rubric.router as mod

        provider = MagicMock()
        provider.value = "openrouter"

        grades, resp = await mod.grade_file_async(
            "prompt_prefix",
            "criteria_text",
            "test.py",
            "anthropic/claude-3.5-sonnet",
            provider,
            0.7,
        )

        spans = span_exporter.get_finished_spans()
        llm_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm_spans) == 1

        span = llm_spans[0]
        assert span.name == "chat anthropic/claude-3.5-sonnet"
        assert span.attributes["gen_ai.system"] == "openrouter"
        assert span.attributes["gen_ai.operation.name"] == "chat"
        assert (
            span.attributes["gen_ai.request.model"]
            == "anthropic/claude-3.5-sonnet"
        )
        assert span.attributes["gen_ai.request.temperature"] == 0.7
        assert span.kind == SpanKind.CLIENT
        assert span.status.status_code == StatusCode.OK

    async def test_llm_span_captures_usage(self, span_exporter, instrument):
        """LLM span should capture token usage from response."""
        import slop_code.metrics.rubric.router as mod

        provider = MagicMock()
        provider.value = "openrouter"

        await mod.grade_file_async(
            "prefix",
            "criteria",
            "file.py",
            "anthropic/claude-3.5-sonnet",
            provider,
            0.5,
        )

        spans = span_exporter.get_finished_spans()
        llm_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm_spans) == 1
        span = llm_spans[0]

        assert span.attributes["gen_ai.usage.input_tokens"] == 500
        assert span.attributes["gen_ai.usage.output_tokens"] == 200
        assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 100
        assert (
            span.attributes["gen_ai.usage.cache_creation.input_tokens"] == 50
        )
        assert span.attributes["gen_ai.response.id"] == "resp-123"

    async def test_llm_span_error(self, span_exporter, tracer_provider):
        """Exception in grade_file_async should produce an error LLM span."""
        import slop_code.metrics.rubric.router as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        original = mod.grade_file_async

        async def failing_grade(*args, **kwargs):
            raise ConnectionError("API unreachable")

        mod.grade_file_async = failing_grade

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        provider = MagicMock()
        provider.value = "bedrock"

        try:
            with pytest.raises(ConnectionError, match="API unreachable"):
                await mod.grade_file_async(
                    "prefix",
                    "criteria",
                    "file.py",
                    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    provider,
                    0.3,
                )

            spans = span_exporter.get_finished_spans()
            llm_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "LLM"
            ]
            assert len(llm_spans) == 1
            assert llm_spans[0].status.status_code == StatusCode.ERROR
            assert llm_spans[0].attributes["gen_ai.system"] == "bedrock"
        finally:
            instrumentor.uninstrument()
            mod.grade_file_async = original

    async def test_llm_span_bedrock_provider(self, span_exporter, instrument):
        """LLM span with bedrock provider should use 'bedrock' as system."""
        import slop_code.metrics.rubric.router as mod

        provider = MagicMock()
        provider.value = "bedrock"

        await mod.grade_file_async(
            "prefix",
            "criteria",
            "file.py",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            provider,
            0.5,
        )

        spans = span_exporter.get_finished_spans()
        llm_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm_spans) == 1
        assert llm_spans[0].attributes["gen_ai.system"] == "bedrock"

    async def test_llm_span_with_choices_output(
        self, span_exporter, tracer_provider
    ):
        """LLM span should capture output choices when available."""
        import slop_code.metrics.rubric.router as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        original = mod.grade_file_async

        async def grade_with_choices(*args, **kwargs):
            grades = [{"score": 9, "reasoning": "Excellent"}]
            response_data = {
                "id": "resp-456",
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": 100,
                },
                "choices": [
                    {"message": {"role": "assistant", "content": "Score: 9"}}
                ],
            }
            return grades, response_data

        mod.grade_file_async = grade_with_choices

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            provider = MagicMock()
            provider.value = "openrouter"

            await mod.grade_file_async(
                "prefix",
                "criteria",
                "file.py",
                "gpt-4",
                provider,
                0.5,
            )

            spans = span_exporter.get_finished_spans()
            llm_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "LLM"
            ]
            assert len(llm_spans) == 1
            span = llm_spans[0]
            assert "gen_ai.output.messages" in span.attributes
            assert "Score: 9" in span.attributes["gen_ai.output.messages"]
        finally:
            instrumentor.uninstrument()
            mod.grade_file_async = original

    async def test_llm_span_usage_not_dict(
        self, span_exporter, tracer_provider
    ):
        """LLM span should handle non-dict usage gracefully."""
        import slop_code.metrics.rubric.router as mod

        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        original = mod.grade_file_async

        async def grade_no_usage(*args, **kwargs):
            grades = [{"score": 5}]
            response_data = {
                "id": "resp-789",
                "usage": "not a dict",
            }
            return grades, response_data

        mod.grade_file_async = grade_no_usage

        instrumentor = SlopCodeInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            provider = MagicMock()
            provider.value = "openrouter"

            await mod.grade_file_async(
                "prefix",
                "criteria",
                "file.py",
                "gpt-4",
                provider,
                0.5,
            )

            spans = span_exporter.get_finished_spans()
            llm_spans = [
                s
                for s in spans
                if s.attributes.get("gen_ai.span.kind") == "LLM"
            ]
            assert len(llm_spans) == 1
            # Should not crash and should not set usage tokens
            assert "gen_ai.usage.input_tokens" not in llm_spans[0].attributes
        finally:
            instrumentor.uninstrument()
            mod.grade_file_async = original
