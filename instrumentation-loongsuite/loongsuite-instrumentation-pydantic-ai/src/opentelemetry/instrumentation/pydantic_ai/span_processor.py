# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from opentelemetry import metrics
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace import Span, StatusCode
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GenAiSpanKindValues,
)

logger = logging.getLogger(__name__)

GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

FRAMEWORK_NAME = "pydantic-ai"
SLOW_CALL_THRESHOLD_SECONDS = 1.0
MAX_PROVIDER_TRACE_IDS = 2048

OPERATION_TO_SPAN_KIND = {
    "create_agent": GenAiSpanKindValues.AGENT.value,
    "invoke_agent": GenAiSpanKindValues.AGENT.value,
    "chat": GenAiSpanKindValues.LLM.value,
    "generate_content": GenAiSpanKindValues.LLM.value,
    "text_completion": GenAiSpanKindValues.LLM.value,
    "execute_tool": GenAiSpanKindValues.TOOL.value,
    "embeddings": GenAiSpanKindValues.EMBEDDING.value,
    "react": GenAiSpanKindValues.STEP.value,
    "enter": GenAiSpanKindValues.ENTRY.value,
    "retrieval": GenAiSpanKindValues.RETRIEVER.value,
}


def infer_span_kind(attributes: dict[str, Any]) -> str | None:
    operation_name = attributes.get(GEN_AI_OPERATION_NAME)
    if isinstance(operation_name, str):
        mapped = OPERATION_TO_SPAN_KIND.get(operation_name)
        if mapped:
            return mapped
    return None


def normalize_genai_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(attributes)
    if GEN_AI_OPERATION_NAME in normalized:
        normalized.setdefault(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
    if GEN_AI_SPAN_KIND not in normalized:
        span_kind = infer_span_kind(normalized)
        if span_kind is not None:
            normalized[GEN_AI_SPAN_KIND] = span_kind

    input_tokens = _to_int(normalized.get(GEN_AI_USAGE_INPUT_TOKENS))
    output_tokens = _to_int(normalized.get(GEN_AI_USAGE_OUTPUT_TOKENS))
    if (
        GEN_AI_USAGE_TOTAL_TOKENS not in normalized
        and input_tokens is not None
        and output_tokens is not None
    ):
        normalized[GEN_AI_USAGE_TOTAL_TOKENS] = input_tokens + output_tokens
    return normalized


class LoongSuiteSpanProcessor(SpanProcessor):
    """Normalize Pydantic AI spans to LoongSuite GenAI semantics."""

    def __init__(
        self,
        *,
        meter_provider: metrics.MeterProvider | None = None,
        slow_call_threshold_seconds: float = SLOW_CALL_THRESHOLD_SECONDS,
    ) -> None:
        self._enabled = True
        self._slow_call_threshold_seconds = slow_call_threshold_seconds
        self._provider_llm_trace_ids: set[int] = set()
        self._provider_llm_trace_order: deque[int] = deque(
            maxlen=MAX_PROVIDER_TRACE_IDS
        )
        self._meter = (
            meter_provider.get_meter(__name__)
            if meter_provider is not None
            else metrics.get_meter(__name__)
        )
        self._genai_calls_count = self._meter.create_counter("genai_calls_count")
        self._genai_calls_error_count = self._meter.create_counter(
            "genai_calls_error_count"
        )
        self._genai_calls_slow_count = self._meter.create_counter(
            "genai_calls_slow_count"
        )
        self._genai_calls_duration = self._meter.create_histogram(
            "genai_calls_duration_seconds",
            unit="s",
        )
        self._genai_usage_tokens = self._meter.create_counter(
            "genai_llm_usage_tokens",
            unit="{token}",
        )
        self._genai_first_token = self._meter.create_histogram(
            "genai_llm_first_token_seconds",
            unit="s",
        )
        self._arms_request_metrics: dict[str, dict[str, Any]] = {}

    def on_start(
        self,
        span: Span,
        parent_context: Context | None = None,
    ) -> None:
        if not self._enabled or not span.is_recording():
            return
        attributes = getattr(span, "attributes", None)
        if attributes is None:
            return
        if not _is_pydantic_ai_span(span, dict(attributes)):
            return
        normalized = normalize_genai_attributes(dict(attributes))
        for key, value in normalized.items():
            if attributes.get(key) != value:
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        if not self._enabled:
            return
        current_attributes = dict(span.attributes or {})
        if _is_provider_llm_span(span, current_attributes):
            self._remember_provider_llm_trace(span)
            return
        if not _is_pydantic_ai_span(span, current_attributes):
            return
        attributes = normalize_genai_attributes(current_attributes)
        self._replace_readable_span_attributes(span, attributes)
        self._record_metrics(span, attributes)

    def shutdown(self) -> None:
        self._enabled = False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def disable(self) -> None:
        self._enabled = False

    def _replace_readable_span_attributes(
        self,
        span: ReadableSpan,
        attributes: dict[str, Any],
    ) -> None:
        try:
            span._attributes = attributes  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to normalize pydantic-ai span: %s", exc)

    def _record_metrics(
        self,
        span: ReadableSpan,
        attributes: dict[str, Any],
    ) -> None:
        span_kind = attributes.get(GEN_AI_SPAN_KIND)
        if span_kind not in {
            GenAiSpanKindValues.AGENT.value,
            GenAiSpanKindValues.LLM.value,
            GenAiSpanKindValues.EMBEDDING.value,
            GenAiSpanKindValues.TOOL.value,
            GenAiSpanKindValues.STEP.value,
        }:
            return
        if (
            span_kind == GenAiSpanKindValues.LLM.value
            and self._has_provider_llm_span(span)
        ):
            return

        metric_attributes = {
            "modelName": _model_name(attributes),
            "spanKind": span_kind,
        }
        call_type = _call_type(attributes)
        duration_seconds = _duration_seconds(span)
        self._genai_calls_count.add(1, metric_attributes)
        self._record_arms_request_count(call_type)
        if duration_seconds is not None:
            self._genai_calls_duration.record(
                duration_seconds,
                metric_attributes,
            )
            self._record_arms_request_duration(call_type, duration_seconds)
            if duration_seconds > self._slow_call_threshold_seconds:
                self._genai_calls_slow_count.add(1, metric_attributes)
                self._record_arms_request_slow_count(call_type)

        if span.status.status_code == StatusCode.ERROR:
            self._genai_calls_error_count.add(1, metric_attributes)
            self._record_arms_request_error_count(call_type)

        if span_kind == GenAiSpanKindValues.LLM.value:
            self._record_llm_metrics(attributes, metric_attributes)

    def _record_llm_metrics(
        self,
        attributes: dict[str, Any],
        metric_attributes: dict[str, Any],
    ) -> None:
        input_tokens = _to_int(attributes.get(GEN_AI_USAGE_INPUT_TOKENS))
        output_tokens = _to_int(attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS))
        if input_tokens is not None:
            self._genai_usage_tokens.add(
                input_tokens,
                {**metric_attributes, "usageType": "input"},
            )
        if output_tokens is not None:
            self._genai_usage_tokens.add(
                output_tokens,
                {**metric_attributes, "usageType": "output"},
            )

        ttft_ns = _to_int(attributes.get(GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN))
        if ttft_ns is not None:
            self._genai_first_token.record(
                ttft_ns / 1_000_000_000,
                metric_attributes,
            )

    def _record_arms_request_count(self, call_type: str) -> None:
        self._arms_metrics(call_type)["count"].add(1)

    def _record_arms_request_error_count(self, call_type: str) -> None:
        self._arms_metrics(call_type)["error_count"].add(1)

    def _record_arms_request_slow_count(self, call_type: str) -> None:
        self._arms_metrics(call_type)["slow_count"].add(1)

    def _record_arms_request_duration(
        self,
        call_type: str,
        duration_seconds: float,
    ) -> None:
        self._arms_metrics(call_type)["duration"].record(duration_seconds)

    def _arms_metrics(self, call_type: str) -> dict[str, Any]:
        metrics_by_type = self._arms_request_metrics.get(call_type)
        if metrics_by_type is None:
            metrics_by_type = {
                "count": self._meter.create_counter(
                    f"arms_{call_type}_requests_count"
                ),
                "error_count": self._meter.create_counter(
                    f"arms_{call_type}_requests_error_count"
                ),
                "duration": self._meter.create_histogram(
                    f"arms_{call_type}_requests_seconds",
                    unit="s",
                ),
                "slow_count": self._meter.create_counter(
                    f"arms_{call_type}_requests_slow_count"
                ),
            }
            self._arms_request_metrics[call_type] = metrics_by_type
        return metrics_by_type

    def _remember_provider_llm_trace(self, span: ReadableSpan) -> None:
        trace_id = _trace_id(span)
        if trace_id is None or trace_id in self._provider_llm_trace_ids:
            return
        if (
            len(self._provider_llm_trace_order)
            == self._provider_llm_trace_order.maxlen
        ):
            old_trace_id = self._provider_llm_trace_order.popleft()
            self._provider_llm_trace_ids.discard(old_trace_id)
        self._provider_llm_trace_ids.add(trace_id)
        self._provider_llm_trace_order.append(trace_id)

    def _has_provider_llm_span(self, span: ReadableSpan) -> bool:
        trace_id = _trace_id(span)
        return trace_id is not None and trace_id in self._provider_llm_trace_ids


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration_seconds(span: ReadableSpan) -> float | None:
    if span.start_time is None or span.end_time is None:
        return None
    return (span.end_time - span.start_time) / 1_000_000_000


def _model_name(attributes: dict[str, Any]) -> str:
    for key in (GEN_AI_RESPONSE_MODEL, GEN_AI_REQUEST_MODEL, "model_name"):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _call_type(attributes: dict[str, Any]) -> str:
    operation_name = attributes.get(GEN_AI_OPERATION_NAME)
    if operation_name == "execute_tool":
        return "tool"
    if isinstance(operation_name, str) and operation_name:
        return operation_name
    span_kind = attributes.get(GEN_AI_SPAN_KIND)
    if isinstance(span_kind, str) and span_kind:
        return span_kind.lower()
    return "unknown"


def _trace_id(span: Any) -> int | None:
    context = getattr(span, "context", None)
    trace_id = getattr(context, "trace_id", None)
    if isinstance(trace_id, int):
        return trace_id
    return None


def _scope_name(span: Any) -> str | None:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return scope_name if isinstance(scope_name, str) else None


def _is_provider_llm_span(span: Any, attributes: dict[str, Any]) -> bool:
    if _is_pydantic_ai_span(span, attributes):
        return False
    operation_name = attributes.get(GEN_AI_OPERATION_NAME)
    span_kind = attributes.get(GEN_AI_SPAN_KIND)
    return operation_name in {
        "chat",
        "generate_content",
        "text_completion",
    } or span_kind == GenAiSpanKindValues.LLM.value


def _is_pydantic_ai_span(span: Any, attributes: dict[str, Any]) -> bool:
    if attributes.get(GEN_AI_FRAMEWORK) == FRAMEWORK_NAME:
        return True
    scope_name = _scope_name(span)
    if scope_name in {
        "pydantic-ai",
        "opentelemetry.instrumentation.pydantic_ai.capability",
    }:
        return True
    return any(key.startswith("pydantic_ai.") for key in attributes)
