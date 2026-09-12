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

"""Bridge MAF native spans through ``opentelemetry-util-genai`` semantics.

Microsoft Agent Framework already owns correct span lifetime and streaming
cleanup behavior.  This bridge keeps those native spans, but patches MAF's span
creation helpers so util-genai's invocation finish helpers run while the span is
still recording.  That keeps LoongSuite GenAI attributes in the SDK snapshot
seen by exporters instead of relying on post-end SpanProcessor mutation.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import sys
import timeit
import weakref
from time import perf_counter, time_ns
from typing import Any, AsyncGenerator, Callable, Generator, Mapping, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import SpanKind

try:
    from aliyun.sdk.extension.arms.semconv import _SUPPRESS_LLM_SDK_KEY
except ImportError:
    _SUPPRESS_LLM_SDK_KEY = None

try:
    from opentelemetry.util.genai.extended_span_utils import (
        _apply_embedding_finish_attributes,
        _apply_execute_tool_finish_attributes,
        _apply_invoke_agent_finish_attributes,
    )
    from opentelemetry.util.genai.extended_types import (
        EmbeddingInvocation,
        ExecuteToolInvocation,
        InvokeAgentInvocation,
    )
    from opentelemetry.util.genai.span_utils import (
        _apply_llm_finish_attributes,
    )
    from opentelemetry.util.genai.types import LLMInvocation
except ImportError as exc:
    _UTIL_GENAI_IMPORT_ERROR: Optional[ImportError] = exc
    _apply_embedding_finish_attributes = None
    _apply_execute_tool_finish_attributes = None
    _apply_invoke_agent_finish_attributes = None
    _apply_llm_finish_attributes = None
    EmbeddingInvocation = None
    ExecuteToolInvocation = None
    InvokeAgentInvocation = None
    LLMInvocation = None
else:
    _UTIL_GENAI_IMPORT_ERROR = None

from .semantic_conventions import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TTFT,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    MAF_LIVE_SPAN_MARKER,
    GenAIOperation,
    GenAISpanKind,
)
from .span_processor import (
    _FRAMEWORK_PROVIDER_NAME,
    _agent_boundary_input_messages_value,
    _agent_boundary_output_messages_value,
    _attr_value,
    _classify_span,
    _delete_attr,
    _is_exception_event,
    _is_maf_span,
    _mcp_tool_name,
    _normalize_finish_reason_value,
    _normalize_provider,
    _normalized_output_messages_value,
    _ttft_from_events,
)

logger = logging.getLogger(__name__)

_applied = False
_original_get_span: Any = None
_original_start_streaming_span: Any = None
_original_activate_span: Any = None
_original_get_function_span: Any = None
_original_create_mcp_client_span: Any = None
_original_tools_get_function_span: Any = None
_original_tools_get_meter: Any = None
_original_mcp_create_mcp_client_span: Any = None
_original_agent_trace_invocation: Any = None
_original_get_tracer: Any = None
_original_get_meter: Any = None
_legacy_response_stream_originals: dict[str, Any] = {}

_FINALIZED_ATTR = "_loongsuite_util_genai_finalized"
_END_WRAPPED_ATTR = "_loongsuite_util_genai_end_wrapped"
_STREAM_START_ATTR = "_loongsuite_util_genai_stream_start_s"
_STREAM_FIRST_TOKEN_ATTR = "_loongsuite_util_genai_stream_first_token_s"
_MAF_BRIDGE_MARKER = {MAF_LIVE_SPAN_MARKER: _FRAMEWORK_PROVIDER_NAME}


def _mark_maf_live_span(span: OtelSpan | None) -> None:
    if span is None:
        return
    try:
        setattr(span, MAF_LIVE_SPAN_MARKER, _FRAMEWORK_PROVIDER_NAME)
    except Exception:
        pass


def apply_util_genai_bridge(
    *,
    tracer_provider: Any = None,
    meter_provider: Any = None,
) -> None:
    """Patch MAF span helpers so GenAI spans are finalized by util-genai."""
    global _applied
    global _original_create_mcp_client_span
    global _original_activate_span
    global _original_mcp_create_mcp_client_span
    global _original_get_function_span
    global _original_get_span
    global _original_start_streaming_span
    global _original_tools_get_function_span
    global _original_tools_get_meter
    global _original_agent_trace_invocation
    global _original_get_tracer
    global _original_get_meter

    if _applied:
        return
    if _UTIL_GENAI_IMPORT_ERROR is not None:
        logger.warning(
            "MAF util-genai bridge skipped: opentelemetry-util-genai "
            "finish helpers unavailable: %s",
            _UTIL_GENAI_IMPORT_ERROR,
        )
        return

    try:
        import agent_framework.observability as observability  # type: ignore
    except ImportError as exc:
        logger.warning("MAF util-genai bridge skipped: %s", exc)
        return

    _original_get_tracer = getattr(observability, "get_tracer", None)
    _original_get_meter = getattr(observability, "get_meter", None)
    _original_get_span = getattr(observability, "_get_span", None)
    _original_start_streaming_span = getattr(
        observability, "_start_streaming_span", None
    )
    _original_activate_span = getattr(observability, "_activate_span", None)
    _original_get_function_span = getattr(
        observability, "get_function_span", None
    )
    _original_create_mcp_client_span = getattr(
        observability, "create_mcp_client_span", None
    )
    _original_agent_trace_invocation = getattr(
        getattr(observability, "AgentTelemetryLayer", None),
        "_trace_agent_invocation",
        None,
    )

    wrapped_get_span = (
        _wrap_get_span(_original_get_span)
        if _original_get_span is not None
        else None
    )
    wrapped_start_streaming_span = (
        _wrap_start_streaming_span(_original_start_streaming_span)
        if _original_start_streaming_span is not None
        else None
    )
    wrapped_get_function_span = (
        _wrap_get_function_span(_original_get_function_span)
        if _original_get_function_span is not None
        else None
    )
    wrapped_create_mcp_client_span = (
        _wrap_create_mcp_client_span(_original_create_mcp_client_span)
        if _original_create_mcp_client_span is not None
        else None
    )
    wrapped_get_tracer = (
        _wrap_provider_get_tracer(_original_get_tracer, tracer_provider)
        if tracer_provider is not None and _original_get_tracer is not None
        else None
    )
    wrapped_get_meter = (
        _wrap_provider_get_meter(_original_get_meter, meter_provider)
        if meter_provider is not None and _original_get_meter is not None
        else None
    )

    if wrapped_get_tracer is not None:
        observability.get_tracer = wrapped_get_tracer  # type: ignore[attr-defined]
    if wrapped_get_meter is not None:
        observability.get_meter = wrapped_get_meter  # type: ignore[attr-defined]
    if wrapped_get_span is not None:
        observability._get_span = wrapped_get_span  # type: ignore[attr-defined]
    if _original_start_streaming_span is not None:
        observability._start_streaming_span = wrapped_start_streaming_span  # type: ignore[attr-defined]
    if _original_activate_span is not None:
        observability._activate_span = _wrap_activate_span(  # type: ignore[attr-defined]
            _original_activate_span
        )
    if wrapped_get_function_span is not None:
        observability.get_function_span = wrapped_get_function_span  # type: ignore[attr-defined]
    if wrapped_create_mcp_client_span is not None:
        observability.create_mcp_client_span = wrapped_create_mcp_client_span  # type: ignore[attr-defined]

    legacy_response_stream_patched = _patch_legacy_response_stream()
    if legacy_response_stream_patched and _original_agent_trace_invocation:
        agent_cls = getattr(observability, "AgentTelemetryLayer", None)
        if agent_cls is not None:
            agent_cls._trace_agent_invocation = (
                _wrap_legacy_agent_trace_invocation(  # type: ignore[attr-defined]
                    _original_agent_trace_invocation
                )
            )

    try:
        import agent_framework._tools as tools_mod  # type: ignore

        _original_tools_get_function_span = getattr(
            tools_mod, "get_function_span", None
        )
        _original_tools_get_meter = getattr(tools_mod, "get_meter", None)
        if wrapped_get_function_span is not None:
            tools_mod.get_function_span = wrapped_get_function_span  # type: ignore[attr-defined]
        if (
            wrapped_get_meter is not None
            and _original_tools_get_meter is not None
        ):
            tools_mod.get_meter = wrapped_get_meter  # type: ignore[attr-defined]
    except ImportError:
        _original_tools_get_function_span = None
        _original_tools_get_meter = None

    try:
        import agent_framework._mcp as mcp_mod  # type: ignore

        _original_mcp_create_mcp_client_span = getattr(
            mcp_mod, "create_mcp_client_span", None
        )
        if wrapped_create_mcp_client_span is not None:
            mcp_mod.create_mcp_client_span = wrapped_create_mcp_client_span  # type: ignore[attr-defined]
    except ImportError:
        _original_mcp_create_mcp_client_span = None

    _applied = True
    logger.info("MAF util-genai span bridge applied.")


def revert_util_genai_bridge() -> None:
    """Restore MAF span helpers patched by :func:`apply_util_genai_bridge`."""
    global _applied
    global _original_create_mcp_client_span
    global _original_activate_span
    global _original_mcp_create_mcp_client_span
    global _original_get_function_span
    global _original_get_span
    global _original_start_streaming_span
    global _original_tools_get_function_span
    global _original_tools_get_meter
    global _original_agent_trace_invocation
    global _original_get_tracer
    global _original_get_meter

    if not _applied:
        return
    try:
        import agent_framework.observability as observability  # type: ignore

        if _original_get_tracer is not None:
            observability.get_tracer = _original_get_tracer  # type: ignore[attr-defined]
        if _original_get_meter is not None:
            observability.get_meter = _original_get_meter  # type: ignore[attr-defined]
        if _original_get_span is not None:
            observability._get_span = _original_get_span  # type: ignore[attr-defined]
        if _original_start_streaming_span is not None:
            observability._start_streaming_span = (
                _original_start_streaming_span  # type: ignore[attr-defined]
            )
        if _original_activate_span is not None:
            observability._activate_span = _original_activate_span  # type: ignore[attr-defined]
        if _original_get_function_span is not None:
            observability.get_function_span = _original_get_function_span  # type: ignore[attr-defined]
        if _original_create_mcp_client_span is not None:
            observability.create_mcp_client_span = (
                _original_create_mcp_client_span  # type: ignore[attr-defined]
            )
        agent_cls = getattr(observability, "AgentTelemetryLayer", None)
        if (
            agent_cls is not None
            and _original_agent_trace_invocation is not None
        ):
            agent_cls._trace_agent_invocation = (  # type: ignore[attr-defined]
                _original_agent_trace_invocation
            )
    except ImportError:
        pass
    try:
        import agent_framework._tools as tools_mod  # type: ignore

        if _original_tools_get_function_span is not None:
            tools_mod.get_function_span = _original_tools_get_function_span  # type: ignore[attr-defined]
        if _original_tools_get_meter is not None:
            tools_mod.get_meter = _original_tools_get_meter  # type: ignore[attr-defined]
    except ImportError:
        pass
    try:
        import agent_framework._mcp as mcp_mod  # type: ignore

        if _original_mcp_create_mcp_client_span is not None:
            mcp_mod.create_mcp_client_span = (
                _original_mcp_create_mcp_client_span  # type: ignore[attr-defined]
            )
    except ImportError:
        pass

    _applied = False
    _original_get_span = None
    _original_start_streaming_span = None
    _original_activate_span = None
    _original_get_function_span = None
    _original_create_mcp_client_span = None
    _original_tools_get_function_span = None
    _original_tools_get_meter = None
    _original_mcp_create_mcp_client_span = None
    _original_agent_trace_invocation = None
    _original_get_tracer = None
    _original_get_meter = None
    _restore_legacy_response_stream()


def _patch_legacy_response_stream() -> bool:
    """Add per-pull context support to older MAF ``ResponseStream``.

    MAF 1.0 streaming spans were created detached and the stream type had no
    pull-context hook, so child spans created while resolving/iterating the
    stream became roots. MAF 1.10 added ``with_pull_context_manager``; this
    compatibility shim backports only that context propagation surface.
    """
    global _legacy_response_stream_originals

    if _legacy_response_stream_originals:
        return True
    try:
        from agent_framework._types import ResponseStream  # type: ignore
    except ImportError:
        return False
    if hasattr(ResponseStream, "with_pull_context_manager"):
        return False

    originals = {
        "__init__": ResponseStream.__init__,
        "_get_stream": ResponseStream._get_stream,
        "__anext__": ResponseStream.__anext__,
        "_run_cleanup_hooks": ResponseStream._run_cleanup_hooks,
    }
    _legacy_response_stream_originals = originals

    original_init = originals["__init__"]
    original_get_stream = originals["_get_stream"]
    original_anext = originals["__anext__"]
    original_run_cleanup_hooks = originals["_run_cleanup_hooks"]

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._pull_context_manager_factories = []
        self._stream_error = None

    async def _get_stream(self: Any) -> Any:
        if getattr(self, "_stream", None) is not None:
            # The stream is already resolved; child spans are created while
            # pulling updates, which is covered by the patched ``__anext__``.
            return await original_get_stream(self)
        with contextlib.ExitStack() as stack:
            for factory in getattr(
                self, "_pull_context_manager_factories", ()
            ):
                stack.enter_context(factory())
            return await original_get_stream(self)

    async def _anext(self: Any) -> Any:
        with contextlib.ExitStack() as stack:
            for factory in getattr(
                self, "_pull_context_manager_factories", ()
            ):
                stack.enter_context(factory())
            return await original_anext(self)

    async def _run_cleanup_hooks(self: Any) -> Any:
        _, exc, _ = sys.exc_info()
        if exc is not None and not isinstance(exc, StopAsyncIteration):
            self._stream_error = exc
        try:
            with contextlib.ExitStack() as stack:
                for factory in getattr(
                    self, "_pull_context_manager_factories", ()
                ):
                    stack.enter_context(factory())
                return await original_run_cleanup_hooks(self)
        finally:
            if exc is not None:
                self._stream_error = None

    def _with_pull_context_manager(self: Any, cm_factory: Callable[[], Any]):
        self._pull_context_manager_factories.append(cm_factory)
        return self

    ResponseStream.__init__ = _init  # type: ignore[assignment]
    ResponseStream._get_stream = _get_stream  # type: ignore[assignment]
    ResponseStream.__anext__ = _anext  # type: ignore[assignment]
    ResponseStream._run_cleanup_hooks = _run_cleanup_hooks  # type: ignore[assignment]
    ResponseStream.with_pull_context_manager = _with_pull_context_manager  # type: ignore[attr-defined]
    return True


def _restore_legacy_response_stream() -> None:
    global _legacy_response_stream_originals
    if not _legacy_response_stream_originals:
        return
    try:
        from agent_framework._types import ResponseStream  # type: ignore
    except ImportError:
        _legacy_response_stream_originals = {}
        return
    ResponseStream.__init__ = _legacy_response_stream_originals["__init__"]  # type: ignore[assignment]
    ResponseStream._get_stream = _legacy_response_stream_originals[
        "_get_stream"
    ]  # type: ignore[assignment]
    ResponseStream.__anext__ = _legacy_response_stream_originals["__anext__"]  # type: ignore[assignment]
    ResponseStream._run_cleanup_hooks = _legacy_response_stream_originals[  # type: ignore[assignment]
        "_run_cleanup_hooks"
    ]
    try:
        delattr(ResponseStream, "with_pull_context_manager")
    except AttributeError:
        pass
    _legacy_response_stream_originals = {}


def _wrap_legacy_agent_trace_invocation(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    def _trace_agent_invocation(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args or not kwargs.get("stream"):
            return original(self, *args, **kwargs)
        return _legacy_streaming_agent_invocation(self, **kwargs)

    return _trace_agent_invocation


def _legacy_streaming_agent_invocation(self: Any, **kwargs: Any) -> Any:
    import agent_framework.observability as observability  # type: ignore
    from agent_framework._types import ResponseStream  # type: ignore

    execute = kwargs["execute"]
    messages = kwargs.get("messages")
    session = kwargs.get("session")
    merged_options = kwargs.get("merged_options") or {}
    client_kwargs = kwargs.get("client_kwargs")
    if not observability.OBSERVABILITY_SETTINGS.ENABLED:
        return execute()

    provider_name = str(getattr(self, "otel_provider_name", "unknown"))
    merged_client_kwargs = (
        dict(client_kwargs) if client_kwargs is not None else {}
    )
    OtelAttr = observability.OtelAttr
    attributes = observability._get_span_attributes(
        operation_name=OtelAttr.AGENT_INVOKE_OPERATION,
        provider_name=provider_name,
        agent_id=getattr(self, "id", "unknown"),
        agent_name=getattr(self, "name", None)
        or getattr(self, "id", "unknown"),
        agent_description=getattr(self, "description", None),
        thread_id=session.service_session_id if session else None,
        all_options=dict(merged_options),
        **merged_client_kwargs,
    )

    operation = attributes.get(OtelAttr.OPERATION, "operation")
    span_name = attributes.get(OtelAttr.AGENT_NAME, "unknown")
    span = observability.get_tracer().start_span(
        f"{operation} {span_name}",
        kind=otel_trace.SpanKind.INTERNAL,
        attributes=attributes,
    )
    _mark_maf_live_span(span)
    _wrap_span_end(span)

    if (
        observability.OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED
        and messages
        and span.is_recording()
    ):
        observability._capture_messages(
            span=span,
            provider_name=provider_name,
            messages=messages,
            system_instructions=observability._get_instructions_from_options(
                dict(merged_options)
            ),
        )

    span_state = {"closed": False}
    duration_state: dict[str, float] = {}
    start_time = perf_counter()
    inner_response_telemetry_captured_fields: set[str] = set()
    inner_accumulated_usage: dict[str, Any] = {}

    def _close_span() -> None:
        if span_state["closed"]:
            return
        span_state["closed"] = True
        span.end()

    def _record_duration() -> None:
        duration_state["duration"] = perf_counter() - start_time

    try:
        with _activate_live_span(span):
            run_result = execute()
        if isinstance(run_result, ResponseStream):
            result_stream = run_result
        elif inspect.isawaitable(run_result):
            result_stream = ResponseStream.from_awaitable(run_result)
        else:
            raise RuntimeError(
                "Streaming telemetry requires a ResponseStream result."
            )
    except Exception as exception:
        observability.capture_exception(
            span=span, exception=exception, timestamp=time_ns()
        )
        _close_span()
        raise

    async def _finalize_stream() -> None:
        try:
            stream_error = getattr(result_stream, "_stream_error", None)
            if stream_error is not None:
                observability.capture_exception(
                    span=span, exception=stream_error, timestamp=time_ns()
                )
                return
            response = await result_stream.get_final_response()
            response_attributes = observability._get_response_attributes(
                attributes,
                response,
                capture_response_id=(
                    observability.INNER_RESPONSE_ID_CAPTURED_FIELD
                    not in inner_response_telemetry_captured_fields
                ),
                capture_usage=(
                    observability.INNER_USAGE_CAPTURED_FIELD
                    not in inner_response_telemetry_captured_fields
                ),
            )
            observability._apply_accumulated_usage(
                response_attributes,
                inner_response_telemetry_captured_fields,
            )
            observability._capture_response(
                span=span,
                attributes=response_attributes,
                duration=duration_state.get("duration"),
            )
            if (
                observability.OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED
                and getattr(response, "messages", None)
                and span.is_recording()
            ):
                observability._capture_messages(
                    span=span,
                    provider_name=provider_name,
                    messages=response.messages,
                    output=True,
                )
        except Exception as exception:
            observability.capture_exception(
                span=span, exception=exception, timestamp=time_ns()
            )
        finally:
            _close_span()

    @contextlib.contextmanager
    def _inner_telemetry_pull_context() -> Any:
        fields_token = (
            observability.INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.set(
                inner_response_telemetry_captured_fields
            )
        )
        usage_token = observability.INNER_ACCUMULATED_USAGE.set(
            inner_accumulated_usage
        )
        try:
            with _activate_live_span(span):
                yield
        finally:
            observability.INNER_ACCUMULATED_USAGE.reset(usage_token)
            observability.INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.reset(
                fields_token
            )

    wrapped_stream = (
        result_stream.with_cleanup_hook(_record_duration)
        .with_cleanup_hook(_finalize_stream)
        .with_pull_context_manager(_inner_telemetry_pull_context)
    )
    try:
        weakref.finalize(wrapped_stream, _close_span)
    except TypeError:
        logger.debug("MAF ResponseStream is not weak-referenceable")
    return wrapped_stream


def _activate_live_span(span: OtelSpan) -> Any:
    return otel_trace.use_span(
        span=span,
        end_on_exit=False,
        record_exception=False,
        set_status_on_exception=False,
    )


@contextlib.contextmanager
def _activate_non_streaming_span(
    span: OtelSpan,
    attributes: Mapping[Any, Any],
) -> Generator[None, Any, Any]:
    """Keep a MAF-owned non-streaming span current for the wrapped call."""
    context = otel_trace.set_span_in_context(span, otel_context.get_current())
    if (
        _SUPPRESS_LLM_SDK_KEY is not None
        and _mapping_value(attributes, GEN_AI_SPAN_KIND) == GenAISpanKind.LLM
    ):
        context = otel_context.set_value(
            _SUPPRESS_LLM_SDK_KEY,
            True,
            context,
        )
    token = otel_context.attach(context)
    try:
        yield
    finally:
        otel_context.detach(token)


def _wrap_get_span(original: Callable[..., Any]) -> Callable[..., Any]:
    @contextlib.contextmanager
    def _get_span(
        attributes: dict[str, Any],
        span_name_attribute: str,
    ) -> Generator[OtelSpan, Any, Any]:
        bridge_attrs = _prepare_start_attributes(attributes)
        span_cm = _current_span_context(
            original, bridge_attrs, span_name_attribute
        )
        with span_cm as span:
            _mark_maf_live_span(span)
            with _activate_non_streaming_span(span, bridge_attrs):
                try:
                    yield span
                finally:
                    _finalize_with_util_genai(span)

    return _get_span


def _wrap_start_streaming_span(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    def _start_streaming_span(
        attributes: dict[str, Any], span_name_attribute: str
    ) -> OtelSpan:
        bridge_attrs = _prepare_start_attributes(attributes)
        span = _start_streaming_span_with_kind(
            original, bridge_attrs, span_name_attribute
        )
        _mark_maf_live_span(span)
        _mark_stream_start(span)
        _wrap_span_end(span)
        return span

    return _start_streaming_span


def _wrap_activate_span(original: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.isasyncgenfunction(original):

        @contextlib.asynccontextmanager
        async def _async_activate_span(
            span: OtelSpan,
        ) -> AsyncGenerator[None, Any]:
            async with original(span):
                try:
                    yield
                except Exception:
                    raise
                else:
                    _record_first_stream_pull(span)

        return _async_activate_span

    if inspect.iscoroutinefunction(original):

        @contextlib.asynccontextmanager
        async def _awaited_activate_span(
            span: OtelSpan,
        ) -> AsyncGenerator[None, Any]:
            cm = await original(span)
            async with cm:
                try:
                    yield
                except Exception:
                    raise
                else:
                    _record_first_stream_pull(span)

        return _awaited_activate_span

    def _activate_span(span: OtelSpan) -> Any:
        cm = original(span)
        if hasattr(cm, "__aenter__"):
            return _async_activate_context(span, cm)
        return _sync_activate_context(span, cm)

    return _activate_span


@contextlib.contextmanager
def _sync_activate_context(
    span: OtelSpan, cm: Any
) -> Generator[None, Any, Any]:
    with cm:
        try:
            yield
        except Exception:
            raise
        else:
            _record_first_stream_pull(span)


@contextlib.asynccontextmanager
async def _async_activate_context(
    span: OtelSpan, cm: Any
) -> AsyncGenerator[None, Any]:
    async with cm:
        try:
            yield
        except Exception:
            raise
        else:
            _record_first_stream_pull(span)


def _wrap_get_function_span(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @contextlib.contextmanager
    def _get_function_span(
        attributes: dict[str, Any],
    ) -> Generator[OtelSpan, Any, Any]:
        bridge_attrs = _prepare_start_attributes(attributes)
        with original(bridge_attrs) as span:
            _mark_maf_live_span(span)
            try:
                yield span
            finally:
                _finalize_with_util_genai(span)

    return _get_function_span


def _wrap_create_mcp_client_span(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @contextlib.contextmanager
    def _create_mcp_client_span(
        method_name: str,
        target: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[OtelSpan, Any, Any]:
        bridge_attrs = dict(attributes or {})
        if method_name == "tools/call":
            bridge_attrs[GEN_AI_OPERATION_NAME] = GenAIOperation.EXECUTE_TOOL
            bridge_attrs[GEN_AI_SPAN_KIND] = GenAISpanKind.MCP
            if target:
                bridge_attrs.setdefault("gen_ai.tool.name", target)
        with original(method_name, target, bridge_attrs) as span:
            _mark_maf_live_span(span)
            yield span

    return _create_mcp_client_span


def _wrap_provider_get_tracer(
    original: Callable[..., Any], tracer_provider: Any
) -> Callable[..., Any]:
    """Route MAF's global-style tracer accessor to an explicit provider."""
    signature = inspect.signature(original)

    def _get_tracer(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        name = bound.arguments.get(
            "instrumenting_module_name", "agent_framework"
        )
        version = bound.arguments.get("instrumenting_library_version")
        schema_url = bound.arguments.get("schema_url")
        attributes = bound.arguments.get("attributes")
        try:
            return tracer_provider.get_tracer(
                name,
                version,
                schema_url,
                attributes,
            )
        except TypeError:
            return tracer_provider.get_tracer(name, version, schema_url)

    return _get_tracer


def _wrap_provider_get_meter(
    original: Callable[..., Any], meter_provider: Any
) -> Callable[..., Any]:
    """Route MAF's native metric instruments to an explicit provider."""
    signature = inspect.signature(original)

    def _get_meter(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        name = bound.arguments.get("name", "agent_framework")
        version = bound.arguments.get("version")
        schema_url = bound.arguments.get("schema_url")
        attributes = bound.arguments.get("attributes")
        try:
            return meter_provider.get_meter(
                name,
                version,
                schema_url,
                attributes,
            )
        except TypeError:
            return meter_provider.get_meter(name, version, schema_url)

    return _get_meter


def _current_span_context(
    original: Callable[..., Any],
    attributes: dict[str, Any],
    span_name_attribute: str,
) -> Any:
    kind = _otel_start_kind(attributes)
    if kind is None:
        return original(attributes, span_name_attribute)
    return _start_current_span_with_kind(attributes, span_name_attribute, kind)


def _start_streaming_span_with_kind(
    original: Callable[..., Any],
    attributes: dict[str, Any],
    span_name_attribute: str,
) -> OtelSpan:
    kind = _otel_start_kind(attributes)
    if kind is None:
        return original(attributes, span_name_attribute)
    return _start_detached_span_with_kind(
        attributes, span_name_attribute, kind
    )


def _otel_start_kind(attributes: Mapping[Any, Any]) -> SpanKind | None:
    span_kind = _mapping_value(attributes, GEN_AI_SPAN_KIND)
    if span_kind in {GenAISpanKind.LLM, GenAISpanKind.EMBEDDING}:
        return SpanKind.CLIENT
    return None


def _start_current_span_with_kind(
    attributes: dict[str, Any],
    span_name_attribute: str,
    kind: SpanKind,
) -> Any:
    span = _start_detached_span_with_kind(
        attributes, span_name_attribute, kind
    )
    return otel_trace.use_span(
        span=span,
        end_on_exit=True,
        record_exception=False,
        set_status_on_exception=False,
    )


def _start_detached_span_with_kind(
    attributes: dict[str, Any],
    span_name_attribute: str,
    kind: SpanKind,
) -> OtelSpan:
    import agent_framework.observability as observability  # type: ignore

    operation = (
        _mapping_value(attributes, GEN_AI_OPERATION_NAME) or "operation"
    )
    span_name = _mapping_value(attributes, span_name_attribute) or "unknown"
    span = observability.get_tracer().start_span(
        f"{operation} {span_name}", kind=kind
    )
    _mark_maf_live_span(span)
    span.set_attributes(attributes)
    return span


def _wrap_span_end(span: OtelSpan) -> None:
    if getattr(span, _END_WRAPPED_ATTR, False):
        return
    original_end = getattr(span, "end", None)
    if original_end is None:
        return

    def _end(*args: Any, **kwargs: Any) -> Any:
        _finalize_with_util_genai(span)
        return original_end(*args, **kwargs)

    try:
        setattr(span, "end", _end)
        setattr(span, _END_WRAPPED_ATTR, True)
    except Exception as exc:  # pragma: no cover - SDK defensive
        logger.warning(
            "MAF streaming span finalization bridge disabled: %s", exc
        )


def _mark_stream_start(span: OtelSpan) -> None:
    try:
        setattr(span, _STREAM_START_ATTR, timeit.default_timer())
    except Exception:
        pass


def _record_first_stream_pull(span: OtelSpan) -> None:
    # MAF registers ``_activate_span(span)`` through
    # ``ResponseStream.with_pull_context_manager``. That factory is entered and
    # exited once per ``__anext__`` pull, so the first successful exit marks the
    # first streamed update rather than final stream cleanup. The context
    # manager API does not expose the update object, so keep this as an internal
    # fallback marker and let finalization prefer any real TTFT event emitted by
    # the framework/provider before writing the public GenAI attribute.
    if getattr(span, _STREAM_FIRST_TOKEN_ATTR, None) is not None:
        return
    try:
        setattr(span, _STREAM_FIRST_TOKEN_ATTR, timeit.default_timer())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("could not record MAF streaming TTFT: %s", exc)


def _prepare_start_attributes(attributes: Mapping[Any, Any]) -> dict[str, Any]:
    """Seed attributes known before MAF creates the span."""
    bridge_attrs = dict(attributes)
    op_name = _mapping_value(bridge_attrs, GEN_AI_OPERATION_NAME)
    span_name = _span_name_from_attributes(bridge_attrs)
    if not _is_maf_span(
        span_name,
        op_name if isinstance(op_name, str) else None,
        bridge_attrs,
        _MAF_BRIDGE_MARKER,
    ):
        return bridge_attrs
    span_kind, classified_op = _classify_span(
        span_name, op_name if isinstance(op_name, str) else None, bridge_attrs
    )
    if not _mapping_value(bridge_attrs, GEN_AI_OPERATION_NAME):
        bridge_attrs[GEN_AI_OPERATION_NAME] = classified_op
    if not _mapping_value(bridge_attrs, GEN_AI_SPAN_KIND):
        bridge_attrs[GEN_AI_SPAN_KIND] = span_kind
    provider = _normalize_provider(
        _mapping_value(bridge_attrs, GEN_AI_PROVIDER_NAME)
    )
    if provider is not None:
        bridge_attrs[GEN_AI_PROVIDER_NAME] = provider
    return bridge_attrs


def _finalize_with_util_genai(span: OtelSpan) -> None:
    if getattr(span, _FINALIZED_ATTR, False):
        return
    try:
        name = getattr(span, "name", "") or ""
        existing_op = _attr_value(span, GEN_AI_OPERATION_NAME)
        existing_op = existing_op if isinstance(existing_op, str) else None
        if not _is_maf_span(name, existing_op, span):
            return

        span_kind, op_name = _classify_span(name, existing_op, span)
        _set_common_live_attributes(span, span_kind, op_name)

        if span_kind == GenAISpanKind.LLM:
            _apply_llm_finish_attributes(span, _llm_invocation(span, op_name))
            ttft = _ttft_from_live_span(span)
            if ttft is not None and not _attr_value(
                span, GEN_AI_RESPONSE_TTFT
            ):
                span.set_attribute(GEN_AI_RESPONSE_TTFT, ttft)
        elif (
            span_kind == GenAISpanKind.AGENT
            and op_name == GenAIOperation.INVOKE_AGENT
        ):
            _apply_invoke_agent_finish_attributes(
                span, _invoke_agent_invocation(span)
            )
        elif span_kind == GenAISpanKind.TOOL:
            _apply_execute_tool_finish_attributes(
                span, _execute_tool_invocation(span)
            )
        elif span_kind == GenAISpanKind.EMBEDDING:
            _apply_embedding_finish_attributes(
                span, _embedding_invocation(span)
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("MAF util-genai bridge finalize failed: %s", exc)
    finally:
        try:
            setattr(span, _FINALIZED_ATTR, True)
        except Exception:
            pass


def _set_common_live_attributes(
    span: OtelSpan, span_kind: str, op_name: str
) -> None:
    if not _attr_value(span, GEN_AI_SPAN_KIND):
        span.set_attribute(GEN_AI_SPAN_KIND, span_kind)
    current_op = _attr_value(span, GEN_AI_OPERATION_NAME)
    if not current_op or (
        current_op != op_name and span_kind == GenAISpanKind.AGENT
    ):
        span.set_attribute(GEN_AI_OPERATION_NAME, op_name)
    if span_kind == GenAISpanKind.MCP and not _attr_value(
        span, "gen_ai.tool.name"
    ):
        tool_name = _mcp_tool_name(getattr(span, "name", "") or "", span)
        if tool_name:
            span.set_attribute("gen_ai.tool.name", tool_name)
    provider = _normalize_provider(_attr_value(span, GEN_AI_PROVIDER_NAME))
    if provider is not None:
        span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
    elif span_kind == GenAISpanKind.AGENT and op_name in {
        GenAIOperation.CREATE_AGENT,
        GenAIOperation.INVOKE_AGENT,
    }:
        span.set_attribute(GEN_AI_PROVIDER_NAME, _FRAMEWORK_PROVIDER_NAME)
    finish_reasons = _finish_reasons(span)
    if finish_reasons:
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
    agent_boundary = (
        span_kind == GenAISpanKind.AGENT
        and op_name == GenAIOperation.INVOKE_AGENT
    )
    if agent_boundary:
        handled_input, input_messages = _agent_boundary_input_messages_value(
            _attr_value(span, "gen_ai.input.messages")
        )
        if handled_input:
            _set_or_delete_attr(span, "gen_ai.input.messages", input_messages)
        handled_output, output_messages = (
            _agent_boundary_output_messages_value(
                _attr_value(span, "gen_ai.output.messages")
            )
        )
        if handled_output:
            _set_or_delete_attr(
                span, "gen_ai.output.messages", output_messages
            )
    else:
        output_messages = _normalized_output_messages_value(
            _attr_value(span, "gen_ai.output.messages")
        )
        if output_messages is not None:
            span.set_attribute("gen_ai.output.messages", output_messages)


def _set_or_delete_attr(span: OtelSpan, key: str, value: Any | None) -> None:
    if value is None:
        _delete_attr(span, key)
    else:
        span.set_attribute(key, value)


def _llm_invocation(span: OtelSpan, op_name: str) -> LLMInvocation:
    return LLMInvocation(
        request_model=_string_attr(span, GEN_AI_REQUEST_MODEL)
        or _model_from_span_name(span, "chat ")
        or "unknown",
        operation_name=op_name or GenAIOperation.CHAT,
        provider=_string_attr(span, GEN_AI_PROVIDER_NAME),
        response_model_name=_string_attr(span, GEN_AI_RESPONSE_MODEL),
        finish_reasons=_finish_reasons(span),
        input_tokens=_int_attr(span, GEN_AI_USAGE_INPUT_TOKENS),
        output_tokens=_int_attr(span, GEN_AI_USAGE_OUTPUT_TOKENS),
        temperature=_float_attr(span, "gen_ai.request.temperature"),
        top_p=_float_attr(span, "gen_ai.request.top_p"),
        frequency_penalty=_float_attr(
            span, "gen_ai.request.frequency_penalty"
        ),
        presence_penalty=_float_attr(span, "gen_ai.request.presence_penalty"),
        max_tokens=_int_attr(span, "gen_ai.request.max_tokens"),
        stop_sequences=_string_list_attr(
            span, "gen_ai.request.stop_sequences"
        ),
        seed=_int_attr(span, "gen_ai.request.seed"),
        conversation_id=_string_attr(span, "gen_ai.conversation.id"),
        choice_count=_int_attr(span, "gen_ai.request.choice.count"),
    )


def _invoke_agent_invocation(span: OtelSpan) -> InvokeAgentInvocation:
    return InvokeAgentInvocation(
        provider=_string_attr(span, GEN_AI_PROVIDER_NAME) or "",
        agent_name=_string_attr(span, "gen_ai.agent.name")
        or _model_from_span_name(span, "invoke_agent "),
        agent_id=_string_attr(span, "gen_ai.agent.id"),
        agent_description=_string_attr(span, "gen_ai.agent.description"),
        conversation_id=_string_attr(span, "gen_ai.conversation.id"),
        request_model=_string_attr(span, GEN_AI_REQUEST_MODEL),
        response_model_name=_string_attr(span, GEN_AI_RESPONSE_MODEL),
        finish_reasons=_finish_reasons(span),
        input_tokens=_int_attr(span, GEN_AI_USAGE_INPUT_TOKENS),
        output_tokens=_int_attr(span, GEN_AI_USAGE_OUTPUT_TOKENS),
        temperature=_float_attr(span, "gen_ai.request.temperature"),
        top_p=_float_attr(span, "gen_ai.request.top_p"),
        frequency_penalty=_float_attr(
            span, "gen_ai.request.frequency_penalty"
        ),
        presence_penalty=_float_attr(span, "gen_ai.request.presence_penalty"),
        max_tokens=_int_attr(span, "gen_ai.request.max_tokens"),
        stop_sequences=_string_list_attr(
            span, "gen_ai.request.stop_sequences"
        ),
        seed=_int_attr(span, "gen_ai.request.seed"),
        choice_count=_int_attr(span, "gen_ai.request.choice.count"),
    )


def _execute_tool_invocation(span: OtelSpan) -> ExecuteToolInvocation:
    return ExecuteToolInvocation(
        tool_name=_string_attr(span, "gen_ai.tool.name")
        or _model_from_span_name(span, "execute_tool ")
        or "unknown",
        provider=_string_attr(span, GEN_AI_PROVIDER_NAME),
        tool_call_id=_string_attr(span, "gen_ai.tool.call.id"),
        tool_description=_string_attr(span, "gen_ai.tool.description"),
        tool_type=_string_attr(span, "gen_ai.tool.type"),
    )


def _embedding_invocation(span: OtelSpan) -> EmbeddingInvocation:
    return EmbeddingInvocation(
        request_model=_string_attr(span, GEN_AI_REQUEST_MODEL)
        or _model_from_span_name(span, "embeddings ")
        or "unknown",
        provider=_string_attr(span, GEN_AI_PROVIDER_NAME),
        response_model_name=_string_attr(span, GEN_AI_RESPONSE_MODEL),
        input_tokens=_int_attr(span, GEN_AI_USAGE_INPUT_TOKENS),
    )


def _mapping_value(attributes: Mapping[Any, Any], key: str) -> Any:
    if key in attributes:
        return attributes[key]
    for attr_key, value in attributes.items():
        if str(attr_key) == key:
            return value
    return None


def _span_name_from_attributes(attributes: Mapping[Any, Any]) -> str:
    op = _mapping_value(attributes, GEN_AI_OPERATION_NAME)
    if op == GenAIOperation.CHAT:
        model = _mapping_value(attributes, GEN_AI_REQUEST_MODEL) or "unknown"
        return f"chat {model}"
    if op == GenAIOperation.EMBEDDINGS:
        model = _mapping_value(attributes, GEN_AI_REQUEST_MODEL) or "unknown"
        return f"embeddings {model}"
    if op == GenAIOperation.INVOKE_AGENT:
        name = _mapping_value(attributes, "gen_ai.agent.name") or "unknown"
        return f"invoke_agent {name}"
    if op == GenAIOperation.EXECUTE_TOOL:
        name = _mapping_value(attributes, "gen_ai.tool.name") or "unknown"
        return f"execute_tool {name}"
    method = _mapping_value(attributes, "mcp.method.name")
    if method:
        return str(method)
    return str(op or "")


def _string_attr(span: Any, key: str) -> Optional[str]:
    value = _attr_value(span, key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return value if isinstance(value, str) else str(value)


def _int_attr(span: Any, key: str) -> Optional[int]:
    value = _attr_value(span, key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_attr(span: Any, key: str) -> Optional[float]:
    value = _attr_value(span, key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list_attr(span: Any, key: str) -> Optional[list[str]]:
    value = _attr_value(span, key)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    return None


def _finish_reasons(span: Any) -> Optional[list[str]]:
    value = _attr_value(span, GEN_AI_RESPONSE_FINISH_REASONS)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return [value]
        if isinstance(parsed, list) and all(
            isinstance(item, str) for item in parsed
        ):
            return [_normalize_finish_reason_value(item) for item in parsed]
        return [_normalize_finish_reason_value(value)]
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return [_normalize_finish_reason_value(item) for item in value]
    return None


def _model_from_span_name(span: Any, prefix: str) -> Optional[str]:
    name = getattr(span, "name", "") or ""
    if not isinstance(name, str) or not name.startswith(prefix):
        return None
    value = name[len(prefix) :].strip()
    return value or None


def _ttft_from_live_span(span: OtelSpan) -> Optional[int]:
    ttft = _ttft_from_events(span)
    if ttft is not None:
        return ttft
    status = getattr(span, "status", None)
    if getattr(status, "status_code", None) == otel_trace.StatusCode.ERROR:
        return None
    events = getattr(span, "events", None) or getattr(span, "_events", None)
    if events is not None and any(_is_exception_event(ev) for ev in events):
        return None
    start_s = getattr(span, _STREAM_START_ATTR, None)
    first_s = getattr(span, _STREAM_FIRST_TOKEN_ATTR, None)
    if start_s is not None and first_s is not None:
        try:
            return int(
                max(float(first_s) - float(start_s), 0.0) * 1_000_000_000
            )
        except (TypeError, ValueError):
            return None
    events = getattr(span, "_events", None)
    if events is None:
        return None

    class _ReadableLike:
        pass

    readable = _ReadableLike()
    readable.events = events
    readable.start_time = getattr(span, "start_time", None)
    return _ttft_from_events(readable)


__all__ = [
    "apply_util_genai_bridge",
    "revert_util_genai_bridge",
]
