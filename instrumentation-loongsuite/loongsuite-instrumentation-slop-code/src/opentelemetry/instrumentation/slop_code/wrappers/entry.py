# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""ENTRY span wrappers for slop-code benchmark runs."""

import json
import logging

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.slop_code.utils import (
    SYSTEM_NAME,
    genai_messages,
    safe_get,
    set_optional_attr,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import gen_ai_extended_attributes

logger = logging.getLogger(__name__)


class _EntryWrapper:
    """Wrapper for the top-level CLI run_agent command."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        with self._tracer.start_as_current_span(
            name="enter_ai_application_system",
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: "enter",
                gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.ENTRY.value,
                "gen_ai.framework": SYSTEM_NAME,
            },
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise


class _RunnerEntryWrapper:
    """Create an ENTRY span inside the worker process so child spans share it."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        problem = safe_get(safe_get(instance, "run_spec"), "problem")
        problem_name = safe_get(problem, "name", "unknown")
        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "enter",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.ENTRY.value,
            "gen_ai.framework": SYSTEM_NAME,
            "gen_ai.session.id": str(problem_name),
        }
        # Capture the benchmark problem prompt as the application input when available.
        task = safe_get(problem, "prompt") or safe_get(problem, "statement") or safe_get(problem, "description")
        if task is not None:
            attrs["gen_ai.input.messages"] = genai_messages([{"role": "user", "content": str(task)}])

        with self._tracer.start_as_current_span(
            name="enter_ai_application_system",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                if result is not None:
                    set_optional_attr(span, "output.value", json.dumps(result, ensure_ascii=False, default=str)[:1024])
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
