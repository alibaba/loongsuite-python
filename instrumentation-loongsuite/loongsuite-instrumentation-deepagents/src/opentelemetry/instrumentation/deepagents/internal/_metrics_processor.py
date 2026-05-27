# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SpanProcessor-backed GenAI metrics for deepagents deployments."""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import metrics
from opentelemetry.instrumentation.deepagents.version import __version__
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.trace import StatusCode

from ._attributes import (
    DEFAULT_SLOW_THRESHOLDS_SECONDS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
    GEN_AI_SPAN_KIND,
    GENAI_SPAN_KINDS,
    METRIC_CALLS_COUNT,
    METRIC_CALLS_DURATION_SECONDS,
    METRIC_CALLS_ERROR_COUNT,
    METRIC_CALLS_SLOW_COUNT,
    METRIC_LLM_FIRST_TOKEN_SECONDS,
    METRIC_LLM_USAGE_TOKENS,
    SPAN_KIND_LLM,
    USAGE_TOKEN_ATTRIBUTES,
)

_logger = logging.getLogger(__name__)
_processors_by_provider_id: dict[int, "DeepAgentMetricsSpanProcessor"] = {}


class DeepAgentMetricsSpanProcessor(SpanProcessor):
    """Record LoongSuite GenAI metrics from completed GenAI spans."""

    def __init__(self, meter_provider: Any = None) -> None:
        super().__init__()
        meter = metrics.get_meter(
            __name__,
            __version__,
            meter_provider=meter_provider,
        )
        self._calls_count = meter.create_counter(METRIC_CALLS_COUNT)
        self._calls_duration = meter.create_histogram(
            METRIC_CALLS_DURATION_SECONDS,
            unit="s",
        )
        self._calls_error_count = meter.create_counter(METRIC_CALLS_ERROR_COUNT)
        self._calls_slow_count = meter.create_counter(METRIC_CALLS_SLOW_COUNT)
        self._llm_first_token = meter.create_histogram(
            METRIC_LLM_FIRST_TOKEN_SECONDS,
            unit="s",
        )
        self._llm_usage_tokens = meter.create_counter(METRIC_LLM_USAGE_TOKENS)
        self._thresholds = _load_slow_thresholds()
        self._enabled = True

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        del span, parent_context

    def on_end(self, span: ReadableSpan) -> None:
        if not self._enabled:
            return
        attributes = span.attributes or {}
        span_kind = attributes.get(GEN_AI_SPAN_KIND)
        if span_kind not in GENAI_SPAN_KINDS:
            return

        labels = {
            "spanKind": str(span_kind),
            "modelName": _model_name(attributes),
        }
        duration = _duration_seconds(span)

        self._calls_count.add(1, labels)
        if duration is not None:
            self._calls_duration.record(duration, labels)
            threshold = self._thresholds.get(str(span_kind))
            if threshold is not None and duration > threshold:
                self._calls_slow_count.add(1, labels)
        if getattr(span.status, "status_code", None) == StatusCode.ERROR:
            self._calls_error_count.add(1, labels)

        if span_kind != SPAN_KIND_LLM:
            return

        ttft_ns = _to_float(attributes.get(GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN))
        if ttft_ns is not None:
            self._llm_first_token.record(ttft_ns / 1_000_000_000, labels)

        for usage_type, attribute_name in USAGE_TOKEN_ATTRIBUTES.items():
            token_count = _to_float(attributes.get(attribute_name))
            if token_count is None:
                continue
            usage_labels = {
                "spanKind": SPAN_KIND_LLM,
                "modelName": labels["modelName"],
                "usageType": usage_type,
            }
            self._llm_usage_tokens.add(token_count, usage_labels)

    def shutdown(self) -> None:
        self._enabled = False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True


def install_metrics_processor(
    *,
    tracer_provider: Any,
    meter_provider: Any = None,
) -> None:
    if not isinstance(tracer_provider, TracerProvider):
        _logger.warning(
            "deepagents metrics require an SDK TracerProvider; metrics skipped."
        )
        return
    if meter_provider is None:
        _logger.warning(
            "deepagents metrics meter_provider was not supplied; using the "
            "global MeterProvider."
        )
        meter_provider = metrics.get_meter_provider()
    if not isinstance(meter_provider, SDKMeterProvider):
        _logger.warning(
            "deepagents metrics MeterProvider is %s, not an SDK MeterProvider; "
            "metrics may be no-op.",
            type(meter_provider).__name__,
        )
    provider_id = id(tracer_provider)
    if provider_id in _processors_by_provider_id:
        return
    processor = DeepAgentMetricsSpanProcessor(meter_provider=meter_provider)
    tracer_provider.add_span_processor(processor)
    _processors_by_provider_id[provider_id] = processor


def shutdown_metrics_processors() -> None:
    for processor in list(_processors_by_provider_id.values()):
        processor.shutdown()
    _processors_by_provider_id.clear()


def _duration_seconds(span: ReadableSpan) -> float | None:
    if span.start_time is None or span.end_time is None:
        return None
    return max((span.end_time - span.start_time) / 1_000_000_000, 0.0)


def _model_name(attributes: Any) -> str:
    return str(
        attributes.get(GEN_AI_REQUEST_MODEL)
        or attributes.get(GEN_AI_RESPONSE_MODEL)
        or ""
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_slow_thresholds() -> dict[str, float]:
    thresholds = dict(DEFAULT_SLOW_THRESHOLDS_SECONDS)
    for span_kind, default in DEFAULT_SLOW_THRESHOLDS_SECONDS.items():
        env_name = f"LOONGSUITE_DEEPAGENTS_SLOW_THRESHOLD_{span_kind}_MS"
        raw_value = os.getenv(env_name)
        if raw_value is None:
            thresholds[span_kind] = default
            continue
        try:
            thresholds[span_kind] = float(raw_value) / 1000.0
        except ValueError:
            _logger.warning(
                "Invalid %s=%r; using default %.3fs",
                env_name,
                raw_value,
                default,
            )
            thresholds[span_kind] = default
    return thresholds
