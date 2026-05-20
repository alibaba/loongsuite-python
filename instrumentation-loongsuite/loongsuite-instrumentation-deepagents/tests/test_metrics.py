# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import logging

from opentelemetry.instrumentation.deepagents import DeepAgentsInstrumentor
from opentelemetry.instrumentation.deepagents.internal._metrics_processor import (
    DeepAgentMetricsSpanProcessor,
    install_metrics_processor,
    shutdown_metrics_processors,
)


def _metric_names(metric_reader):
    metric_reader.collect()
    metrics_data = metric_reader.get_metrics_data()
    names = set()
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                names.add(metric.name)
    return names


def test_metrics_processor_records_genai_call_and_llm_usage(
    tracer_provider,
    meter_provider,
    metric_reader,
):
    tracer_provider.add_span_processor(
        DeepAgentMetricsSpanProcessor(meter_provider=meter_provider)
    )
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("chat qwen3.6-plus") as span:
        span.set_attribute("gen_ai.span.kind", "LLM")
        span.set_attribute("gen_ai.request.model", "qwen3.6-plus")
        span.set_attribute("gen_ai.usage.input_tokens", 11)
        span.set_attribute("gen_ai.usage.output_tokens", 7)
        span.set_attribute("gen_ai.usage.total_tokens", 18)
        span.set_attribute("gen_ai.response.time_to_first_token", 250_000_000)

    names = _metric_names(metric_reader)
    assert "genai_calls_count" in names
    assert "genai_calls_duration_seconds" in names
    assert "genai_llm_usage_tokens" in names
    assert "genai_llm_first_token_seconds" in names


def test_instrumentor_warns_when_meter_provider_defaults_to_global(
    tracer_provider,
    caplog,
):
    instrumentor = DeepAgentsInstrumentor()
    caplog.set_level(logging.WARNING)

    instrumentor._instrument(tracer_provider=tracer_provider)
    try:
        assert "meter_provider was not supplied" in caplog.text
        assert "metrics may be no-op" in caplog.text
    finally:
        instrumentor._uninstrument()


def test_metrics_processor_warns_when_meter_provider_is_missing(
    tracer_provider,
    caplog,
):
    caplog.set_level(logging.WARNING)

    install_metrics_processor(tracer_provider=tracer_provider)
    try:
        assert "deepagents metrics meter_provider was not supplied" in caplog.text
        assert "metrics may be no-op" in caplog.text
    finally:
        shutdown_metrics_processors()


def test_metrics_processor_accepts_sdk_meter_provider_without_warning(
    tracer_provider,
    meter_provider,
    caplog,
):
    caplog.set_level(logging.WARNING)

    install_metrics_processor(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    try:
        assert "meter_provider was not supplied" not in caplog.text
        assert "metrics may be no-op" not in caplog.text
    finally:
        shutdown_metrics_processors()
