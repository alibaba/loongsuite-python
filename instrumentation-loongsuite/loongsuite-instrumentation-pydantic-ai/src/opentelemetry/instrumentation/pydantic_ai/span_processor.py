# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import logging
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
        meter = (
            meter_provider.get_meter(__name__)
            if meter_provider is not None
            else metrics.get_meter(__name__)
        )
        self._genai_calls_count = meter.create_counter("genai_calls_count")
        self._genai_calls_error_count = meter.create_counter(
            "genai_calls_error_count"
        )
        self._genai_calls_slow_count = meter.create_counter(
            "genai_calls_slow_count"
        )
        self._genai_calls_duration = meter.create_histogram(
            "genai_calls_duration_seconds",
            unit="s",
        )
        self._genai_usage_tokens = meter.create_counter(
            "genai_llm_usage_tokens",
            unit="{token}",
        )
        self._genai_first_token = meter.create_histogram(
            "genai_llm_first_token_seconds",
            unit="s",
        )

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

        metric_attributes = {
            "modelName": _model_name(attributes),
            "spanKind": span_kind,
        }
        duration_seconds = _duration_seconds(span)
        self._genai_calls_count.add(1, metric_attributes)
        if duration_seconds is not None:
            self._genai_calls_duration.record(
                duration_seconds,
                metric_attributes,
            )
            if duration_seconds > self._slow_call_threshold_seconds:
                self._genai_calls_slow_count.add(1, metric_attributes)

        if span.status.status_code == StatusCode.ERROR:
            self._genai_calls_error_count.add(1, metric_attributes)

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


def _is_pydantic_ai_span(span: Any, attributes: dict[str, Any]) -> bool:
    if attributes.get(GEN_AI_FRAMEWORK) == FRAMEWORK_NAME:
        return True
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    if scope_name in {
        "pydantic-ai",
        "opentelemetry.instrumentation.pydantic_ai.capability",
    }:
        return True
    return any(key.startswith("pydantic_ai.") for key in attributes)
