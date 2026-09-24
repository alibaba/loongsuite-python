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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
import unittest.mock
from typing import Any, List, Optional
from unittest.mock import patch

try:
    try:
        from google.genai._interactions.resources.interactions import (
            AsyncInteractionsResource,
            InteractionsResource,
        )
    except ImportError:
        # In version 2.9 of google-genai these were moved.
        from google.genai._gaos.interactions import (
            AsyncInteractions as AsyncInteractionsResource,
        )
        from google.genai._gaos.interactions import (
            Interactions as InteractionsResource,
        )
    _HAS_INTERACTIONS = True
except ImportError:
    _HAS_INTERACTIONS = False

    # Placeholders to prevent compilation errors
    class AsyncInteractionsResource:
        create = None

    class InteractionsResource:
        create = None


from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ..common.base import TestCase as CommonTestCaseBase
from .util import create_mock_completed_event, create_mock_interaction


class TestCase(CommonTestCaseBase):
    def setUp(self) -> None:
        super().setUp()
        if not _HAS_INTERACTIONS:
            raise unittest.SkipTest(
                "Interactions are not supported in this version of google-genai"
            )
        if self.__class__ == TestCase:
            raise unittest.SkipTest("Skipping testcase base.")
        self._create_mock = None
        self._original_create = InteractionsResource.create
        self._original_async_create = AsyncInteractionsResource.create
        self._interactions: List[Any] = []
        self._interaction_index = 0
        self._completed_event_type = "interaction_completed"

    @property
    def mock_create(self) -> unittest.mock.MagicMock:
        if self._create_mock is None:
            self._create_and_install_mocks()
        assert self._create_mock is not None
        return self._create_mock

    def configure_valid_interaction(self, **kwargs: Any) -> None:
        self._create_and_install_mocks()
        interaction = create_mock_interaction(**kwargs)
        self._interactions.append(interaction)

    def configure_exception(self, e: Exception) -> None:
        self._create_and_install_mocks(e)

    def _create_and_install_mocks(self, e: Optional[Exception] = None) -> None:
        if self._create_mock is not None:
            return
        self.reset_client()
        self.reset_instrumentation()
        self._create_mock = self._create_mock_impl(e)
        self._install_mocks()

    def _create_mock_impl(
        self, e: Optional[Exception] = None
    ) -> unittest.mock.MagicMock:
        mock = unittest.mock.MagicMock()

        def _default_impl(*args: Any, **kwargs: Any) -> Any:
            if not self._interactions:
                result = create_mock_interaction()
            else:
                index = self._interaction_index % len(self._interactions)
                result = self._interactions[index]
                self._interaction_index += 1

            if kwargs.get("stream"):
                completed_event = create_mock_completed_event(result)
                completed_event.event_type = self._completed_event_type
                return [completed_event]
            return result

        mock.side_effect = e or _default_impl
        return mock

    def _install_mocks(self) -> None:
        def _sync_create_wrapped(*args: Any, **kwargs: Any) -> Any:
            assert self._create_mock is not None
            return self._create_mock(*args, **kwargs)

        async def _async_create_wrapped(*args: Any, **kwargs: Any) -> Any:
            assert self._create_mock is not None
            res = self._create_mock(*args, **kwargs)
            if kwargs.get("stream"):

                async def _async_generator() -> Any:
                    for item in res:
                        yield item

                return _async_generator()
            return res

        InteractionsResource.create = _sync_create_wrapped
        AsyncInteractionsResource.create = _async_create_wrapped

    def tearDown(self) -> None:
        super().tearDown()
        InteractionsResource.create = self._original_create
        AsyncInteractionsResource.create = self._original_async_create

    # Abstract methods to be overridden by subclasses
    def run_interaction(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def run_streaming_interaction(
        self, *args: Any, **kwargs: Any
    ) -> List[Any]:
        raise NotImplementedError()

    # The actual collapsed test cases:
    def test_instrumentation_does_not_break_core_functionality(self) -> None:
        self.configure_valid_interaction(
            interaction_id="test-id",
            output_text="Yep, it works!",
        )
        response = self.run_interaction(
            model="gemini-2.5-flash", input="Does this work?"
        )
        self.assertEqual(response.id, "test-id")
        self.assertEqual(response.steps[1].content[0].text, "Yep, it works!")

    def test_generates_span(self) -> None:
        self.configure_valid_interaction()
        self.run_interaction(model="gemini-2.5-flash", input="Does this work?")
        self.otel.assert_has_span_named("interactions.create gemini-2.5-flash")

    def test_model_reflected_into_span_name(self) -> None:
        self.configure_valid_interaction()
        self.run_interaction(model="gemini-1.5-flash", input="Does this work?")
        self.otel.assert_has_span_named("interactions.create gemini-1.5-flash")

    def test_generated_span_has_minimal_genai_attributes(self) -> None:
        self.configure_valid_interaction()
        self.run_interaction(model="gemini-2.5-flash", input="Does this work?")
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        self.assertEqual(span.attributes["gen_ai.provider.name"], "gemini")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "interactions.create"
        )
        self.assertEqual(
            span.attributes["server.address"],
            "generativelanguage.googleapis.com",
        )

    def test_span_and_event_still_written_when_response_is_exception(
        self,
    ) -> None:
        self.configure_exception(ValueError("Uh oh!"))
        with self.assertRaises(ValueError):
            self.run_interaction(
                model="gemini-2.5-flash", input="Does this work?"
            )
        self.otel.assert_has_span_named("interactions.create gemini-2.5-flash")
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        self.otel.assert_has_event_named(
            "gen_ai.client.inference.operation.details"
        )
        event = self.otel.get_event_named(
            "gen_ai.client.inference.operation.details"
        )
        self.assertEqual(span.attributes["error.type"], "ValueError")
        self.assertEqual(event.attributes["error.type"], "ValueError")

    def test_generated_span_has_vertex_ai_system_when_configured(self) -> None:
        self.set_use_vertex(True)
        self.configure_valid_interaction()
        self.run_interaction(model="gemini-2.5-flash", input="Does this work?")
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        self.assertEqual(span.attributes["gen_ai.provider.name"], "vertex_ai")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "interactions.create"
        )

    def test_generated_span_counts_tokens(self) -> None:
        self.configure_valid_interaction(
            input_tokens=15,
            output_tokens=25,
        )
        self.run_interaction(model="gemini-2.5-flash", input="Some input")
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 15)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 25)

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_span_attributes_with_content_capture(self) -> None:
        self.configure_valid_interaction(
            input_text="Hello interactions!",
            output_text="Response from interactions!",
        )
        self.run_interaction(
            model="gemini-2.5-flash",
            input="Hello interactions!",
        )
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        self.assertEqual(
            span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
            '[{"role":"user","parts":[{"content":"Hello interactions!","type":"text"}]}]',
        )
        self.assertEqual(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
            '[{"role":"assistant","parts":[{"content":"Response from interactions!","type":"text"}],"finish_reason":"stop"}]',
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT"},
    )
    def test_span_attributes_no_content_capture(self) -> None:
        self.configure_valid_interaction(
            input_text="Hello interactions!",
            output_text="Response from interactions!",
        )
        self.run_interaction(
            model="gemini-2.5-flash",
            input="Hello interactions!",
        )
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        for attribute in (
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        ):
            self.assertNotIn(attribute, span.attributes)

    def test_streaming_generates_span(self) -> None:
        self.configure_valid_interaction(
            interaction_id="stream-id-1",
            model_name="gemini-2.5-flash",
            output_text="Streaming response!",
            input_tokens=5,
            output_tokens=8,
        )
        events = self.run_streaming_interaction(
            model="gemini-2.5-flash",
            input="Streaming test",
            stream=True,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].interaction.id, "stream-id-1")

        self.otel.assert_has_span_named("interactions.create gemini-2.5-flash")
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 5)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 8)
        self.assertGreater(
            span.attributes["gen_ai.response.time_to_first_token"], 0
        )

    def test_streaming_accepts_real_sdk_completed_event_name(self) -> None:
        self._completed_event_type = "interaction.completed"
        self.configure_valid_interaction(
            interaction_id="stream-id-real-sdk",
            model_name="gemini-2.5-flash",
            output_text="Real SDK streaming response!",
            input_tokens=6,
            output_tokens=9,
        )
        events = self.run_streaming_interaction(
            model="gemini-2.5-flash",
            input="Real SDK streaming test",
            stream=True,
        )
        self.assertEqual(events[0].event_type, "interaction.completed")
        span = self.otel.get_span_named("interactions.create gemini-2.5-flash")
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 6)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 9)
        self.assertEqual(
            span.attributes["gen_ai.response.model"], "gemini-2.5-flash"
        )
