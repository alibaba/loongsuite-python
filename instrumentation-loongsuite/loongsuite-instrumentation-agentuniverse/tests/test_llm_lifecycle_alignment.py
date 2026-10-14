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

"""Review-body follow-ups for the LLM layer.

Three points from the review body are pinned here:

* the teardown of a finished LLM call belongs to the shared handler -- its
  ``abandon_llm`` is the documented best-effort cleanup -- so the plugin should
  not detach a token or end a span of its own;
* ``gen_ai.request.model`` names the model the call asks for, while
  ``au.llm.name`` keeps the component name the framework itself reports;
* a span that cannot be created at all (``tracer.start_span`` raising before
  anything is attached) leaves the call running uninstrumented rather than
  half-traced, and the next call of the same trace is unaffected.

The stubs are driven through the real ``@trace_llm`` seam, so what is under
test is the framework's wrapper resolution.
"""

from typing import Any

from agentuniverse.base.annotation.trace import trace_llm
from agentuniverse.llm.llm_output import LLMOutput

from opentelemetry.instrumentation.agentuniverse import _common as common
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from .conftest import StubLLM
from .test_instrumentor import (
    LLM_SPAN,
    capture_on,
    layer_records,
    one_layer_record,
)


class ModellingLLM(StubLLM):
    """A component whose configured model differs from its component name."""

    name = "modelling_llm"
    model_name = "gpt-4o"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any) -> LLMOutput:
        return LLMOutput(text="answer", finish_reason="stop")


class TestHandlerOwnedTeardown:
    def test_finalize_uses_the_handlers_own_cleanup(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        """The plugin hands the teardown to ``abandon_llm``."""
        seen: list = []
        original = ExtendedTelemetryHandler.abandon_llm

        def spy(self: Any, invocation: Any) -> Any:
            seen.append(invocation)
            return original(self, invocation)

        monkeypatch.setattr(ExtendedTelemetryHandler, "abandon_llm", spy)
        with capture_on():
            StubLLM().call(prompt="hello")

        assert seen, "the handler's own cleanup was never used"
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.status"] == "success"


class TestGenAiIdentity:
    def test_request_model_is_the_model_not_the_component_name(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            ModellingLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.name"] == "modelling_llm"
        assert record.gen_ai["gen_ai.request.model"] == "gpt-4o"

    def test_request_model_falls_back_to_the_component_name(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.delattr(StubLLM, "model_name", raising=False)
        with capture_on():
            StubLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.gen_ai["gen_ai.request.model"] == "stub_llm"

    def test_provider_name_is_the_channel(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            ModellingLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.gen_ai["gen_ai.provider.name"] == "test_channel"


class TestSpanCreationFailure:
    def test_a_span_that_cannot_be_created_leaves_the_call_uninstrumented(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        def boom(self: Any, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no span")

        monkeypatch.setattr(common.LayerSpan, "__init__", boom)
        with capture_on():
            result = StubLLM().call(prompt="hello")
        assert result.text, "the business call still answers"
        assert layer_records(loongsuite_harness, LLM_SPAN) == []

    def test_the_next_call_of_the_trace_is_a_root_span(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        def boom(self: Any, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no span")

        with monkeypatch.context() as patch:
            patch.setattr(common.LayerSpan, "__init__", boom)
            with capture_on():
                StubLLM().call(prompt="hello")
        with capture_on():
            StubLLM().call(prompt="hello")

        records = layer_records(loongsuite_harness, LLM_SPAN)
        assert len(records) == 1, [r.name for r in records]
        assert records[0].parent is None
