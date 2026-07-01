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
import timeit
from typing import Any, AsyncGenerator, Callable, Generator, Mapping, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import SpanKind

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
    GenAIOperation,
    GenAISpanKind,
)
from .span_processor import (
    _FRAMEWORK_PROVIDER_NAME,
    _attr_value,
    _classify_span,
    _is_exception_event,
    _is_maf_span,
    _mcp_tool_name,
    _normalize_provider,
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
_original_mcp_create_mcp_client_span: Any = None

_FINALIZED_ATTR = "_loongsuite_util_genai_finalized"
_END_WRAPPED_ATTR = "_loongsuite_util_genai_end_wrapped"
_STREAM_START_ATTR = "_loongsuite_util_genai_stream_start_s"
_STREAM_FIRST_TOKEN_ATTR = "_loongsuite_util_genai_stream_first_token_s"


def apply_util_genai_bridge() -> None:
    """Patch MAF span helpers so GenAI spans are finalized by util-genai."""
    global _applied
    global _original_create_mcp_client_span
    global _original_activate_span
    global _original_mcp_create_mcp_client_span
    global _original_get_function_span
    global _original_get_span
    global _original_start_streaming_span
    global _original_tools_get_function_span

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

    try:
        import agent_framework._tools as tools_mod  # type: ignore

        _original_tools_get_function_span = getattr(
            tools_mod, "get_function_span", None
        )
        if wrapped_get_function_span is not None:
            tools_mod.get_function_span = wrapped_get_function_span  # type: ignore[attr-defined]
    except ImportError:
        _original_tools_get_function_span = None

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

    if not _applied:
        return
    try:
        import agent_framework.observability as observability  # type: ignore

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
    except ImportError:
        pass
    try:
        import agent_framework._tools as tools_mod  # type: ignore

        if _original_tools_get_function_span is not None:
            tools_mod.get_function_span = _original_tools_get_function_span  # type: ignore[attr-defined]
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
    _original_mcp_create_mcp_client_span = None


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
            yield span

    return _create_mcp_client_span


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
        span_name, op_name if isinstance(op_name, str) else None, bridge_attrs
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
            return parsed
        return [value]
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
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
