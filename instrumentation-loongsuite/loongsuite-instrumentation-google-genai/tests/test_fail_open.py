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
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import pytest
from google.genai import types

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider
from opentelemetry.instrumentation.google_genai._compat import TelemetryHandler
from opentelemetry.instrumentation.google_genai._stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.instrumentation.google_genai.allowlist_util import AllowList
from opentelemetry.instrumentation.google_genai.embeddings import (
    _CAPTURE_RAW_RESPONSE,
    _RAW_RESPONSE_BODY,
    _complete_embedding_advice,
    _EmbeddingAdviceState,
)
from opentelemetry.instrumentation.google_genai.generate_content import (
    _create_instrumented_generate_content,
    _create_instrumented_generate_content_stream,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    AsyncInteractionsStreamWrapper,
    InteractionsStreamWrapper,
    _apply_interaction_response_attributes,
)
from opentelemetry.instrumentation.google_genai.message import (
    StreamOutputMessageAccumulator,
)
from opentelemetry.instrumentation.google_genai.tool_call_wrapper import (
    _wrap_tool_function,
)
from opentelemetry.metrics import get_meter_provider
from opentelemetry.trace import get_tracer_provider

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


def test_prepare_failure_calls_sdk_once_and_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content(handler, AllowList())
    response = object()
    calls = 0

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("probe prepare failed")

    def sdk_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_genai.generate_content."
        "_wrapped_config_with_tools",
        fail_prepare,
    )
    try:
        actual = wrapped(
            sdk_call,
            object(),
            (),
            {"model": "gemini-test", "contents": "hello"},
        )
        assert actual is response
        assert calls == 1
        assert otel.get_finished_spans() == []
    finally:
        otel.uninstall()


def test_response_mapping_failure_returns_original_and_ends_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content(handler, AllowList())
    response = object()
    calls = 0

    def fail_mapping(*args, **kwargs):
        raise RuntimeError("probe mapping failed")

    def sdk_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_genai.generate_content."
        "_apply_response_attributes",
        fail_mapping,
    )
    try:
        actual = wrapped(
            sdk_call,
            object(),
            (),
            {"model": "gemini-test", "contents": "hello"},
        )
        assert actual is response
        assert calls == 1
        assert len(otel.get_finished_spans()) == 1
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        otel.uninstall()


def test_interaction_usage_missing_optional_fields_is_tolerated() -> None:
    class Usage:
        total_input_tokens = 3
        total_output_tokens = 2

    class Response:
        model = "gemini-test"
        usage = Usage()

    class Invocation:
        response_model_name = None
        input_tokens = None
        output_tokens = None
        thinking_tokens = None
        cache_read_input_tokens = None

    class Handler:
        @staticmethod
        def should_capture_content() -> bool:
            return False

    invocation = Invocation()
    _apply_interaction_response_attributes(Response(), invocation, Handler())

    assert invocation.input_tokens == 3
    assert invocation.output_tokens == 2
    assert invocation.thinking_tokens is None
    assert invocation.cache_read_input_tokens is None


@pytest.mark.parametrize(
    "wrapper_factory",
    [
        lambda invocation, handler: InteractionsStreamWrapper(
            iter(()), invocation, handler
        ),
        lambda invocation, handler: AsyncInteractionsStreamWrapper(
            _empty_async_stream(), invocation, handler
        ),
    ],
)
def test_interaction_stream_mapping_failure_still_stops_invocation(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_factory,
) -> None:
    class Invocation:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    def fail_mapping(*args, **kwargs):
        raise RuntimeError("probe mapping failed")

    invocation = Invocation()
    wrapper = wrapper_factory(invocation, object())
    wrapper._self_last_interaction = object()
    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_genai.interactions."
        "_apply_interaction_response_attributes",
        fail_mapping,
    )

    with pytest.raises(RuntimeError, match="probe mapping failed"):
        wrapper._on_stream_end()

    assert invocation.stopped


def test_embedding_mapping_failure_still_stops_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Invocation:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    def fail_mapping(*args, **kwargs):
        raise RuntimeError("probe mapping failed")

    invocation = Invocation()
    raw_body_token = _RAW_RESPONSE_BODY.set(None)
    capture_token = _CAPTURE_RAW_RESPONSE.set(True)
    state = _EmbeddingAdviceState(
        invocation=invocation,
        raw_body_token=raw_body_token,
        capture_token=capture_token,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_genai.embeddings."
        "_apply_embedding_response_attributes",
        fail_mapping,
    )

    _complete_embedding_advice(state, object())

    assert invocation.stopped
    assert _RAW_RESPONSE_BODY.get() is None
    assert not _CAPTURE_RAW_RESPONSE.get()


async def _empty_async_stream():
    if False:
        yield None


def test_failure_reporter_failure_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content(handler, AllowList())
    original = ValueError("business failure")
    calls = 0

    def sdk_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise original

    def fail_reporter(invocation, *args, **kwargs):
        invocation._owner.abandon_inference(invocation)
        raise RuntimeError("probe failure reporter failed")

    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_genai._compat."
        "InferenceInvocation.fail",
        fail_reporter,
    )
    try:
        with pytest.raises(ValueError) as raised:
            wrapped(
                sdk_call,
                object(),
                (),
                {"model": "gemini-test", "contents": "hello"},
            )
        assert raised.value is original
        assert calls == 1
    finally:
        otel.uninstall()


class _FaultySyncWrapper(SyncStreamWrapper[object]):
    def __init__(
        self,
        stream,
        *,
        chunk_failure: bool = False,
        final_failure: bool = False,
    ) -> None:
        super().__init__(stream)
        self._self_chunk_failure = chunk_failure
        self._self_final_failure = final_failure
        self._self_finalizations = 0

    def _process_chunk(self, chunk: object) -> None:
        if self._self_chunk_failure:
            raise RuntimeError("probe chunk failed")

    def _on_stream_end(self) -> None:
        self._self_finalizations += 1
        if self._self_final_failure:
            raise RuntimeError("probe finalization failed")

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_finalizations += 1
        raise RuntimeError("probe error reporter failed")


def test_stream_probe_failures_preserve_chunks_and_finalize_once() -> None:
    chunks = [object(), object()]
    wrapper = _FaultySyncWrapper(
        iter(chunks),
        chunk_failure=True,
        final_failure=True,
    )
    actual = list(wrapper)
    wrapper.close()
    assert actual[0] is chunks[0]
    assert actual[1] is chunks[1]
    assert wrapper._self_finalizations == 1


def test_stream_error_reporter_preserves_original_exception() -> None:
    original = ValueError("business stream failed")

    def stream():
        yield object()
        raise original

    wrapper = _FaultySyncWrapper(stream())
    iterator = iter(wrapper)
    next(iterator)
    with pytest.raises(ValueError) as raised:
        next(iterator)
    assert raised.value is original
    assert wrapper._self_finalizations == 1


def test_stream_close_failure_does_not_replace_business_outcome() -> None:
    class CloseFailure:
        def __iter__(self):
            return iter(())

        def close(self):
            raise RuntimeError("provider close failed")

    wrapper = _FaultySyncWrapper(CloseFailure())
    wrapper.close()
    assert wrapper._self_finalizations == 1


def test_stream_creation_detaches_before_cross_thread_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content_stream(
        handler,
        AllowList(),
    )
    chunks = [object(), object()]

    def sdk_call(*args, **kwargs):
        return iter(chunks)

    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_genai.generate_content."
        "_apply_response_attributes",
        lambda *args, **kwargs: None,
    )
    try:
        stream = wrapped(
            sdk_call,
            object(),
            (),
            {"model": "gemini-test", "contents": "hello"},
        )
        assert not trace.get_current_span().get_span_context().is_valid
        with ThreadPoolExecutor(max_workers=1) as executor:
            actual = executor.submit(list, stream).result()
        assert actual[0] is chunks[0]
        assert actual[1] is chunks[1]
        assert len(otel.get_finished_spans()) == 1
    finally:
        otel.uninstall()


def test_stream_wrapper_construction_failure_returns_original_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content_stream(
        handler,
        AllowList(),
    )
    raw_stream = iter([object()])

    def sdk_call(*args, **kwargs):
        return raw_stream

    class ConstructionFailure:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("probe wrapper construction failed")

    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_genai.generate_content."
        "GenerateContentStreamWrapper",
        ConstructionFailure,
    )
    try:
        actual = wrapped(
            sdk_call,
            object(),
            (),
            {"model": "gemini-test", "contents": "hello"},
        )
        assert actual is raw_stream
        assert len(otel.get_finished_spans()) == 1
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        otel.uninstall()


def test_stream_accumulator_failure_preserves_chunks_and_ends_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content_stream(
        handler,
        AllowList(),
    )
    chunks = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    index=0,
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="business output")],
                    ),
                )
            ]
        )
    ]

    def sdk_call(*args, **kwargs):
        return iter(chunks)

    def fail_accumulation(self, candidates):
        raise RuntimeError("probe accumulation failed")

    monkeypatch.setattr(
        StreamOutputMessageAccumulator,
        "add_candidates",
        fail_accumulation,
    )
    try:
        actual = list(
            wrapped(
                sdk_call,
                object(),
                (),
                {"model": "gemini-test", "contents": "hello"},
            )
        )
        assert actual[0] is chunks[0]
        assert len(otel.get_finished_spans()) == 1
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        otel.uninstall()


def test_stream_output_mapping_failure_still_ends_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, otel = _handler()
    wrapped = _create_instrumented_generate_content_stream(
        handler,
        AllowList(),
    )
    chunk = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="business output")],
                ),
            )
        ]
    )

    def sdk_call(*args, **kwargs):
        return iter([chunk])

    def fail_output_mapping(self):
        raise RuntimeError("probe output mapping failed")

    monkeypatch.setattr(
        StreamOutputMessageAccumulator,
        "to_output_messages",
        fail_output_mapping,
    )
    try:
        actual = list(
            wrapped(
                sdk_call,
                object(),
                (),
                {"model": "gemini-test", "contents": "hello"},
            )
        )
        assert actual == [chunk]
        assert len(otel.get_finished_spans()) == 1
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        otel.uninstall()


def test_generator_exit_identity_and_cleanup_are_preserved() -> None:
    original = GeneratorExit("consumer closed")

    class GeneratorExitIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise original

        def close(self):
            return None

    wrapper = _FaultySyncWrapper(GeneratorExitIterator())
    with pytest.raises(GeneratorExit) as raised:
        next(wrapper)
    assert raised.value is original
    assert wrapper._self_finalizations == 1


def test_one_faulty_concurrent_stream_does_not_contaminate_sibling() -> None:
    chunks = [[object(), object()], [object(), object()]]
    wrappers = [
        _FaultySyncWrapper(iter(chunks[0]), chunk_failure=True),
        _FaultySyncWrapper(iter(chunks[1])),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(list, wrappers))
    assert actual[0][0] is chunks[0][0]
    assert actual[1][0] is chunks[1][0]
    assert wrappers[0]._self_finalizations == 1
    assert wrappers[1]._self_finalizations == 1


class _FaultyAsyncWrapper(AsyncStreamWrapper[object]):
    def __init__(self, stream) -> None:
        super().__init__(stream)
        self._self_finalizations = 0

    def _process_chunk(self, chunk: object) -> None:
        raise RuntimeError("probe async chunk failed")

    def _on_stream_end(self) -> None:
        self._self_finalizations += 1
        raise RuntimeError("probe async finalization failed")

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_finalizations += 1
        raise RuntimeError("probe async error reporter failed")


def test_async_stream_probe_failures_and_aclose_are_fail_open() -> None:
    chunks = [object(), object()]

    async def exercise():
        async def source():
            for chunk in chunks:
                yield chunk

        wrapper = _FaultyAsyncWrapper(source())
        actual = [chunk async for chunk in wrapper]
        await wrapper.aclose()
        return wrapper, actual

    wrapper, actual = asyncio.run(exercise())
    assert actual[0] is chunks[0]
    assert actual[1] is chunks[1]
    assert wrapper._self_finalizations == 1


def test_async_cancellation_preserves_control_flow_and_finalizes_once() -> (
    None
):
    async def exercise():
        gate = asyncio.Event()

        async def source():
            await gate.wait()
            yield object()

        wrapper = _FaultyAsyncWrapper(source())
        task = asyncio.create_task(wrapper.__anext__())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return wrapper

    wrapper = asyncio.run(exercise())
    assert wrapper._self_finalizations == 1


def test_tool_prepare_and_finalize_failures_do_not_change_business_result() -> (
    None
):
    result = object()
    calls = 0

    class Invocation:
        should_capture_content_on_span = True

        def stop(self):
            raise RuntimeError("probe stop failed")

        def fail(self, error):
            raise RuntimeError("probe fail failed")

    class Handler:
        def __init__(self):
            self.prepare_fails = True

        def tool(self, *args, **kwargs):
            if self.prepare_fails:
                raise RuntimeError("probe prepare failed")
            return Invocation()

        def abandon_tool(self, invocation):
            return None

    handler = Handler()

    def business_tool(value):
        nonlocal calls
        calls += 1
        return result

    wrapped = _wrap_tool_function(business_tool, handler)
    assert wrapped("first") is result
    handler.prepare_fails = False
    assert wrapped("second") is result
    assert calls == 2


def test_tool_failure_reporter_does_not_replace_business_exception() -> None:
    original = ValueError("tool business failure")

    class Invocation:
        should_capture_content_on_span = False

        def fail(self, error):
            raise RuntimeError("probe tool reporter failed")

    class Handler:
        def tool(self, *args, **kwargs):
            return Invocation()

        def abandon_tool(self, invocation):
            return None

    def business_tool():
        raise original

    wrapped = _wrap_tool_function(business_tool, Handler())
    with pytest.raises(ValueError) as raised:
        wrapped()
    assert raised.value is original
