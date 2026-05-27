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

from abc import ABC
from contextlib import contextmanager
from inspect import signature
from logging import getLogger
from os import environ
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    OrderedDict,
    Tuple,
)

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.agno._extractor import (
    AgentRunRequestExtractor,
    AgentRunResponseExtractor,
    FunctionCallRequestExtractor,
    FunctionCallResponseExtractor,
    ModelRequestExtractor,
    ModelResponseExtractor,
)
from opentelemetry.instrumentation.agno._with_span import _WithSpan
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

logger = getLogger(__name__)
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)


def bind_arguments(
    method: Callable[..., Any], *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_arguments = method_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    arguments = bound_arguments.arguments
    arguments = OrderedDict(
        {
            key: value
            for key, value in arguments.items()
            if key != "self" and value is not None and value != {}
        }
    )
    return arguments


class _WithTracer(ABC):
    def __init__(
        self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[Tuple[str, AttributeValue]],
        extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our attributes into
        # two tiers, where the addition of "extra_attributes" is deferred until the end
        # and only after the "attributes" are added.
        try:
            # Check if there is an active span
            parent_span = trace_api.get_current_span()
            if parent_span == trace_api.INVALID_SPAN:
                # No active span, create a new root span
                span = self._tracer.start_span(
                    name=span_name, attributes=dict(attributes)
                )
            else:
                # Active span exists, create a child span
                ctx = trace_api.set_span_in_context(parent_span)
                span = self._tracer.start_span(
                    name=span_name, context=ctx, attributes=dict(attributes)
                )

        except Exception:
            logger.exception("Failed to start span")
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(span=span, extra_attributes=dict(extra_attributes))


class AgnoAgentWrapper(_WithTracer):
    def __init__(self, tracer, *args, **kwargs):
        super().__init__(tracer, *args, **kwargs)
        self._request_attributes_extractor = AgentRunRequestExtractor()
        self._response_attributes_extractor = AgentRunResponseExtractor()

    def _enable_genai_capture(self) -> bool:
        capture_content = environ.get(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
        )
        return capture_content.lower() == "true"

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name="Agent.run",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    response
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
                return response
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name="Agent.run_stream",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            try:
                yield from wrapped(*args, **kwargs)
                response = instance.run_response
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    response
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()

    async def arun(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            response = await wrapped(*args, **kwargs)
            return response
        with self._start_as_current_span(
            span_name="Agent.arun",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    response
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
                return response
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()

    async def arun_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            async for response in wrapped(*args, **kwargs):
                yield response
            return
        with self._start_as_current_span(
            span_name="Agent.arun_stream",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            try:
                async for response in wrapped(*args, **kwargs):
                    yield response
                response = instance.run_response
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    response
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()


class AgnoFunctionCallWrapper(_WithTracer):
    def __init__(self, tracer, *args, **kwargs):
        super().__init__(tracer, *args, **kwargs)
        self._request_attributes_extractor = FunctionCallRequestExtractor()
        self._response_attributes_extractor = FunctionCallResponseExtractor()

    def _enable_genai_capture(self) -> bool:
        capture_content = environ.get(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
        )
        return capture_content.lower() == "true"

    def execute(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        function_name = instance.function.name
        if not self._enable_genai_capture() or instance is None:
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=f"ToolCall.{function_name}",
            attributes=self._request_attributes_extractor.extract(instance),
            extra_attributes=self._request_attributes_extractor.extract(
                instance
            ),
        ) as with_span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    response
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
                return response
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()

    async def aexecute(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        function_name = instance.function.name
        if not self._enable_genai_capture() or instance is None:
            return await wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name=f"ToolCall.{function_name}",
            attributes=self._request_attributes_extractor.extract(instance),
            extra_attributes=self._request_attributes_extractor.extract(
                instance
            ),
        ) as with_span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    response
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
                return response
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()


class AgnoModelWrapper(_WithTracer):
    def __init__(self, tracer, *args, **kwargs):
        super().__init__(tracer, *args, **kwargs)
        self._request_attributes_extractor = ModelRequestExtractor()
        self._response_attributes_extractor = ModelResponseExtractor()

    def _enable_genai_capture(self) -> bool:
        capture_content = environ.get(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
        )
        return capture_content.lower() == "true"

    def response(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name="Model.response",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    [response]
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
                return response
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()

    def response_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name="Model.response_stream",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            responses = []
            try:
                for response in wrapped(*args, **kwargs):
                    responses.append(response)
                    yield response
                resp_attr = self._response_attributes_extractor.extract(
                    responses
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise

    async def aresponse(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            return await wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name="Model.aresponse",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                resp_attr = self._response_attributes_extractor.extract(
                    [response]
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
                return response
            except Exception:
                logger.exception(
                    f"Failed to finalize response of type {type(response)}"
                )
                with_span.finish_tracing()

    async def aresponse_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = bind_arguments(wrapped, *args, **kwargs)
        if not self._enable_genai_capture() or instance is None:
            async for response in wrapped(*args, **kwargs):
                yield response
            return
        with self._start_as_current_span(
            span_name="Model.aresponse_stream",
            attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
            extra_attributes=self._request_attributes_extractor.extract(
                instance, arguments
            ),
        ) as with_span:
            responses = []
            try:
                async for response in wrapped(*args, **kwargs):
                    responses.append(response)
                    yield response
                resp_attr = self._response_attributes_extractor.extract(
                    responses
                )
                with_span.finish_tracing(
                    status=trace_api.Status(
                        status_code=trace_api.StatusCode.OK
                    ),
                    attributes=dict(resp_attr),
                    extra_attributes=dict(resp_attr),
                )
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
