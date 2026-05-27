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

"""
Unit tests for _wrapper.py
"""

import asyncio
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock

from opentelemetry.instrumentation.agno._wrapper import AgnoModelWrapper


class TestAresponse:
    """Tests for aresponse() method."""

    def test_aresponse_returns_result_not_coroutine(self):
        """aresponse() should await wrapped() on early return path."""
        mock_instance = MagicMock()
        mock_instance.id = "test-model"

        async def run_test():
            wrapper = AgnoModelWrapper(tracer=MagicMock())
            wrapper._enable_genai_capture = lambda: False

            async def mock_wrapped(*args, **kwargs):
                return "expected_result"

            result = await wrapper.aresponse(
                mock_wrapped, mock_instance, (), {"messages": []}
            )
            assert not asyncio.iscoroutine(result), (
                "aresponse() returned coroutine instead of awaited result"
            )

        asyncio.run(run_test())


class TestResponseStream:
    """Tests for response_stream() method."""

    def test_response_stream_calls_wrapped_once(self):
        """response_stream() should call wrapped() exactly once."""
        call_count = [0]
        original_method = AgnoModelWrapper.response_stream

        def patched_method(self, wrapped, instance, args, kwargs):
            original_wrapped = wrapped

            def counting_wrapped(*a, **kw):
                call_count[0] += 1
                return original_wrapped(*a, **kw)

            return original_method(
                self, counting_wrapped, instance, args, kwargs
            )

        AgnoModelWrapper.response_stream = patched_method
        try:
            mock_instance = MagicMock()
            mock_instance.id = "test-model"

            def mock_generator(*args, **kwargs) -> Iterator[str]:
                yield "chunk1"
                yield "chunk2"

            wrapper = AgnoModelWrapper(tracer=MagicMock())
            wrapper._enable_genai_capture = lambda: True
            wrapper._request_attributes_extractor = MagicMock()
            wrapper._request_attributes_extractor.extract.return_value = {}
            wrapper._response_attributes_extractor = MagicMock()
            wrapper._response_attributes_extractor.extract.return_value = {}

            with_span_mock = MagicMock()
            with_span_mock.__enter__ = MagicMock(return_value=with_span_mock)
            with_span_mock.__exit__ = MagicMock(return_value=False)
            with_span_mock.finish_tracing = MagicMock()
            wrapper._start_as_current_span = MagicMock(
                return_value=with_span_mock
            )

            results = list(
                wrapper.response_stream(
                    mock_generator, mock_instance, (), {"messages": []}
                )
            )

            assert call_count[0] == 1, (
                f"wrapped() called {call_count[0]} times, expected 1"
            )
            assert results == ["chunk1", "chunk2"]
        finally:
            AgnoModelWrapper.response_stream = original_method


class TestAresponseStream:
    """Tests for aresponse_stream() method."""

    def test_aresponse_stream_calls_wrapped_once(self):
        """aresponse_stream() should call wrapped() exactly once."""
        call_count = [0]
        original_method = AgnoModelWrapper.aresponse_stream

        async def patched_method(self, wrapped, instance, args, kwargs):
            original_wrapped = wrapped

            def counting_wrapped(*a, **kw):
                call_count[0] += 1
                return original_wrapped(*a, **kw)

            async for item in original_method(
                self, counting_wrapped, instance, args, kwargs
            ):
                yield item

        AgnoModelWrapper.aresponse_stream = patched_method
        try:
            mock_instance = MagicMock()
            mock_instance.id = "test-model"

            async def mock_async_generator(
                *args, **kwargs
            ) -> AsyncIterator[str]:
                yield "async_chunk1"
                yield "async_chunk2"

            async def run_test():
                wrapper = AgnoModelWrapper(tracer=MagicMock())
                wrapper._enable_genai_capture = lambda: True
                wrapper._request_attributes_extractor = MagicMock()
                wrapper._request_attributes_extractor.extract.return_value = {}
                wrapper._response_attributes_extractor = MagicMock()
                wrapper._response_attributes_extractor.extract.return_value = {}

                with_span_mock = MagicMock()
                with_span_mock.__enter__ = MagicMock(
                    return_value=with_span_mock
                )
                with_span_mock.__exit__ = MagicMock(return_value=False)
                with_span_mock.finish_tracing = MagicMock()
                wrapper._start_as_current_span = MagicMock(
                    return_value=with_span_mock
                )

                results = []
                async for chunk in wrapper.aresponse_stream(
                    mock_async_generator, mock_instance, (), {"messages": []}
                ):
                    results.append(chunk)
                return results

            results = asyncio.run(run_test())

            assert call_count[0] == 1, (
                f"wrapped() called {call_count[0]} times, expected 1"
            )
            assert results == ["async_chunk1", "async_chunk2"]
        finally:
            AgnoModelWrapper.aresponse_stream = original_method
