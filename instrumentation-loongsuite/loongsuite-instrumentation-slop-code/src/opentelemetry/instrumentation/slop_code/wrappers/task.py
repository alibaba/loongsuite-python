# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""ENTRY + TASK span wrapper for AgentRunner._run_checkpoint."""

import logging

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.slop_code.utils import (
    SYSTEM_NAME,
    safe_get,
    set_optional_attr,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)

logger = logging.getLogger(__name__)


class _TaskRunCheckpointWrapper:
    """Create an ENTRY span and a child TASK span for each benchmark checkpoint."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        checkpoint = args[0] if args else kwargs.get("checkpoint")
        is_first_checkpoint = (
            args[2]
            if len(args) > 2
            else kwargs.get("is_first_checkpoint", False)
        )
        checkpoint_name = safe_get(checkpoint, "name", "unknown")
        checkpoint_order = safe_get(checkpoint, "order")
        problem = safe_get(safe_get(instance, "run_spec"), "problem")
        problem_name = safe_get(problem, "name", checkpoint_name)

        entry_attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "enter",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: "ENTRY",
            "gen_ai.framework": SYSTEM_NAME,
            "gen_ai.session.id": str(problem_name),
        }
        task_attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "run_task",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: "TASK",
            "gen_ai.framework": SYSTEM_NAME,
            "input.value": str(checkpoint_name),
            "input.mime_type": "text/plain",
            "slop_code.checkpoint.name": str(checkpoint_name),
            "slop_code.is_first_checkpoint": bool(is_first_checkpoint),
        }
        if checkpoint_order is not None:
            task_attrs["slop_code.checkpoint.order"] = checkpoint_order

        with self._tracer.start_as_current_span(
            name="enter_ai_application_system",
            kind=SpanKind.INTERNAL,
            attributes=entry_attrs,
        ) as entry_span:
            with self._tracer.start_as_current_span(
                name=f"run_task {checkpoint_name}",
                kind=SpanKind.INTERNAL,
                attributes=task_attrs,
            ) as task_span:
                try:
                    result = wrapped(*args, **kwargs)
                    if result is not None:
                        set_optional_attr(
                            task_span,
                            "slop_code.had_error",
                            safe_get(result, "had_error"),
                        )
                        set_optional_attr(
                            task_span,
                            "slop_code.passed_policy",
                            safe_get(result, "passed_policy"),
                        )
                        set_optional_attr(
                            task_span, "output.value", str(result)
                        )
                        set_optional_attr(
                            task_span, "output.mime_type", "text/plain"
                        )
                        set_optional_attr(
                            entry_span, "output.value", str(result)
                        )
                    task_span.set_status(Status(StatusCode.OK))
                    entry_span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    task_span.record_exception(exc)
                    task_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    entry_span.record_exception(exc)
                    entry_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
