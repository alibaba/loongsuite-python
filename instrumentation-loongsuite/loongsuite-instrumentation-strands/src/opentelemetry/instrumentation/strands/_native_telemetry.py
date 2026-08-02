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

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry.context import Context
from opentelemetry.trace import (
    INVALID_SPAN,
    Link,
    NonRecordingSpan,
    Span,
    SpanKind,
    Tracer,
    get_current_span,
)
from opentelemetry.util.types import Attributes


def _enabled(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() not in {"false", "0", "no", "off"}


class NativeTelemetrySuppression:
    """Disable only Strands SDK-native spans while this instrumentor is active."""

    def __init__(self) -> None:
        self._native: Any = None
        self._original: Any = None
        self._replacement: Any = None
        self._trace_api_bindings: list[tuple[Any, Any, Any]] = []

    def install(self) -> None:
        if not _enabled(
            os.getenv("OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_NATIVE"), True
        ):
            return
        from strands.telemetry.tracer import get_tracer  # noqa: PLC0415

        native = get_tracer()
        original = native.tracer
        replacement = _ContextPreservingNoOpTracer()
        native.tracer = replacement
        from strands.agent import agent as agent_module  # noqa: PLC0415
        from strands.event_loop import event_loop  # noqa: PLC0415
        from strands.tools.executors import _executor  # noqa: PLC0415

        for module in (agent_module, event_loop, _executor):
            original_trace_api = module.trace_api
            proxy = _TraceApiProxy(original_trace_api)
            module.trace_api = proxy
            self._trace_api_bindings.append(
                (module, original_trace_api, proxy)
            )
        self._native = native
        self._original = original
        self._replacement = replacement

    def restore(self) -> None:
        if (
            self._native is not None
            and self._native.tracer is self._replacement
        ):
            self._native.tracer = self._original
        for module, original_trace_api, proxy in self._trace_api_bindings:
            if module.trace_api is proxy:
                module.trace_api = original_trace_api
        self._trace_api_bindings.clear()
        self._native = self._original = self._replacement = None


class _TraceApiProxy:
    """Keep Strands' native no-op spans out of application ContextVars."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    @contextmanager
    def use_span(self, span: Span, **kwargs: Any) -> Iterator[Span]:
        del kwargs
        yield span


class _ContextPreservingNoOpTracer(Tracer):
    """Suppress SDK spans without breaking a caller's active trace context."""

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        span = self.start_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        del end_on_exit
        yield span

    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        del (
            name,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
        )
        parent = get_current_span(context)
        span_context = parent.get_span_context()
        if not span_context.is_valid:
            return INVALID_SPAN
        return NonRecordingSpan(span_context)
