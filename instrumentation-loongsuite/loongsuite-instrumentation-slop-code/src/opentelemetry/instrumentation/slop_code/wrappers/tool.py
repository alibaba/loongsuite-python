# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""TOOL span wrapper for MiniSWEAgent.execute_action."""

import json
import logging
from uuid import uuid4

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.slop_code.utils import (
    SYSTEM_NAME,
    truncate_text,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)

logger = logging.getLogger(__name__)


def _json_attr(value) -> str:
    return truncate_text(json.dumps(value, ensure_ascii=False, default=str))


class _ToolExecuteActionWrapper:
    """Wrap shell/tool execution performed by the benchmark agent."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        action = args[0] if args else kwargs.get("action", {})
        command = (
            action.get("action") if isinstance(action, dict) else str(action)
        )
        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "execute_tool",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: "TOOL",
            "gen_ai.framework": SYSTEM_NAME,
            "gen_ai.tool.call.id": str(uuid4()),
            "gen_ai.tool.name": "bash",
            "gen_ai.tool.type": "function",
            "gen_ai.tool.description": "Execute a shell command in the benchmark environment",
            "gen_ai.tool.call.arguments": _json_attr({"command": command}),
        }
        with self._tracer.start_as_current_span(
            name="execute_tool bash",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute(
                    "gen_ai.tool.call.result", _json_attr(result)
                )
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_attribute(
                    "gen_ai.tool.call.result",
                    _json_attr(
                        {"error": str(exc), "error.type": type(exc).__name__}
                    ),
                )
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute("error.type", type(exc).__name__)
                raise
