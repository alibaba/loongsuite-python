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

import asyncio
import os
from typing import Tuple
from unittest.mock import patch

from google.genai import types as genai_types

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger_provider
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.google_genai._compat import TelemetryHandler
from opentelemetry.instrumentation.google_genai._stream import (
    AsyncStreamWrapper,
)
from opentelemetry.instrumentation.google_genai.allowlist_util import AllowList
from opentelemetry.instrumentation.google_genai.generate_content import (
    _create_instrumented_generate_content_stream,
    _wrapped_config_with_tools,
    instrument_generate_content,
    uninstrument_generate_content,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _create_instrumented_interactions_create,
)
from opentelemetry.instrumentation.google_genai.message import _to_part
from opentelemetry.metrics import get_meter_provider
from opentelemetry.trace import SpanKind, get_tracer_provider
from opentelemetry.util.genai.types import Reasoning, Text

from .common.otel_mocker import OTelMocker


def _handler() -> Tuple[TelemetryHandler, OTelMocker]:
    otel = OTelMocker()
    otel.install()
    return (
        TelemetryHandler(
            tracer_provider=get_tracer_provider(),
            meter_provider=get_meter_provider(),
            logger_provider=get_logger_provider(),
        ),
        otel,
    )


def test_standard_instrumentation_suppression_creates_no_span():
    handler, otel = _handler()
    token = context_api.attach(
        context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        with handler.inference(
            provider="gemini",
            request_model="gemini-test",
            operation_name="generate_content",
        ):
            pass
        assert otel.get_finished_spans() == []
    finally:
        context_api.detach(token)
        otel.uninstall()


def test_automatic_function_call_uses_internal_span():
    handler, otel = _handler()
    try:
        with handler.tool("add"):
            pass
        span = otel.get_span_named("execute_tool add")
        assert span.kind is SpanKind.INTERNAL
    finally:
        otel.uninstall()


@patch.dict(
    "os.environ",
    {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    clear=False,
)
def test_content_capture_does_not_require_legacy_stability_flag():
    handler, otel = _handler()
    try:
        invocation = handler.inference(
            provider="gemini",
            request_model="gemini-test",
            operation_name="generate_content",
        )
        invocation.input_messages = []
        invocation.output_messages = []
        invocation.system_instruction = [Text(content="system")]
        invocation.stop()
        span = otel.get_span_named("generate_content gemini-test")
        assert span.attributes["gen_ai.system_instructions"] == (
            '[{"content":"system","type":"text"}]'
        )
    finally:
        otel.uninstall()


def test_google_thought_part_maps_to_reasoning():
    part = _to_part(genai_types.Part(text="internal thought", thought=True), 0)
    assert isinstance(part, Reasoning)
    assert part.content == "internal thought"


def test_wrapping_tools_does_not_mutate_reusable_config():
    handler, otel = _handler()

    def get_temperature(city: str) -> str:
        return city

    config = genai_types.GenerateContentConfig(tools=[get_temperature])
    try:
        first, first_changed = _wrapped_config_with_tools(handler, config)
        second, second_changed = _wrapped_config_with_tools(handler, config)

        assert first_changed and second_changed
        assert config.tools == [get_temperature]
        assert first.tools[0].__wrapped__ is get_temperature
        assert second.tools[0].__wrapped__ is get_temperature
    finally:
        otel.uninstall()


def test_invalid_dict_config_is_passed_through_to_sdk():
    handler, otel = _handler()
    invalid_config = {"temperature": object()}
    try:
        _, has_wrapped_tools = _wrapped_config_with_tools(
            handler, invalid_config
        )
        assert not has_wrapped_tools
    finally:
        otel.uninstall()


def test_sync_stream_creation_failure_finishes_span():
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content_stream(
        handler, AllowList()
    )

    def fail(*args, **kwargs):
        raise ValueError("stream creation failed")

    try:
        try:
            wrapped(
                fail,
                object(),
                (),
                {"model": "gemini-test", "contents": "hello"},
            )
        except ValueError:
            pass
        span = otel.get_span_named("generate_content gemini-test")
        assert span is not None
        assert span.attributes["error.type"] == "ValueError"
    finally:
        otel.uninstall()


def test_interactions_stream_creation_failure_finishes_span():
    handler, otel = _handler()
    wrapped = _create_instrumented_interactions_create(handler)

    def fail(*args, **kwargs):
        raise ValueError("stream creation failed")

    try:
        try:
            wrapped(
                fail,
                object(),
                (),
                {
                    "model": "gemini-test",
                    "input": "hello",
                    "stream": True,
                },
            )
        except ValueError:
            pass
        span = otel.get_span_named("interactions.create gemini-test")
        assert span is not None
        assert span.attributes["error.type"] == "ValueError"
    finally:
        otel.uninstall()


def test_async_stream_aclose_finalizes_google_style_stream():
    finalized = []

    class Wrapper(AsyncStreamWrapper):
        def _process_chunk(self, chunk):
            pass

        def _on_stream_end(self):
            finalized.append("success")

        def _on_stream_error(self, error):
            finalized.append(error)

    async def exercise():
        async def stream():
            yield "chunk"

        underlying = stream()
        wrapper = Wrapper(underlying)
        await wrapper.aclose()

    asyncio.run(exercise())
    assert finalized == ["success"]


@patch.dict(
    os.environ,
    {"OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "false"},
)
def test_instrumentation_preserves_explicit_event_setting():
    handler, otel = _handler()
    snapshot = instrument_generate_content(handler, AllowList())
    try:
        assert os.environ["OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT"] == "false"
    finally:
        uninstrument_generate_content(snapshot)
        otel.uninstall()
