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

"""Wrap ``task_tool`` as a TASK span.

``task_tool`` itself is a LangChain ``@tool``: the LoongSuite langchain
instrumentation already produces a ``gen_ai.tool.name="task"`` TOOL Span via
its ``LoongsuiteTracer.on_tool_start/end``. DeerFlow adds an outer TASK span
describing the "dispatch to subagent" semantic; the langchain TOOL Span
becomes the child.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.deer_flow.utils import (
    DEER_FLOW_COMPONENT,
    DEER_FLOW_OPERATION,
    DEER_FLOW_TASK_NAME,
    _safe_call,
    _should_capture_content,
)
from opentelemetry import context as otel_context
from opentelemetry.trace import SpanKind, Status, StatusCode, get_tracer, set_span_in_context
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

logger = logging.getLogger(__name__)

_MODULE = "deerflow.tools.builtins.task_tool"
_ATTR = "task_tool"
OP_NAME = "task_tool"
SPAN_NAME_PREFIX = "run_task"


class _TaskToolWrapper:
    def __init__(self, handler: ExtendedTelemetryHandler | None = None):
        self._handler = handler
        self._tracer = get_tracer(
            "opentelemetry.instrumentation.deer_flow",
        )

    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        # ``task_tool`` is a LangChain ``@tool`` with signature
        # ``task_tool(runtime, description, prompt, subagent_type, tool_call_id)``,
        # where ``runtime`` and ``tool_call_id`` are injected. The actual call
        # may pass ``description`` either as arg[0] or arg[1] depending on
        # whether LangChain forwards the injected ``runtime`` positionally.
        description = kwargs.get("description") or (
            args[1] if len(args) > 1 else (args[0] if args else "")
        )
        prompt = kwargs.get("prompt") or (
            args[2] if len(args) > 2 else (args[1] if len(args) > 1 else "")
        )
        subagent_type = kwargs.get("subagent_type") or (
            args[3] if len(args) > 3 else (args[2] if len(args) > 2 else "unknown")
        )

        task_name = f"subagent.invoke:{subagent_type}"
        span_name = f"{SPAN_NAME_PREFIX} {task_name}"

        span = self._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute(GEN_AI_SPAN_KIND, "TASK")
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
        span.set_attribute(DEER_FLOW_OPERATION, OP_NAME)
        span.set_attribute(DEER_FLOW_COMPONENT, "task_tool")
        span.set_attribute(DEER_FLOW_TASK_NAME, task_name)
        span.set_attribute(GenAI.GEN_AI_AGENT_NAME, f"subagent:{subagent_type}")

        # Attach the TASK span to the OTel Context so the langchain TOOL span
        # emitted inside ``wrapped`` becomes a child of this TASK span (see
        # execute.md §3.4.3).
        token = otel_context.attach(set_span_in_context(span))

        if _should_capture_content():
            try:
                span.set_attribute(
                    "input.value",
                    json.dumps(
                        {
                            "description": description,
                            "prompt": prompt,
                            "subagent_type": subagent_type,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            except Exception:
                pass

        result: Any = None
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            if _should_capture_content() and result is not None:
                _safe_call(
                    "set_output_value",
                    lambda r: span.set_attribute("output.value", str(r)),
                    result,
                )
            otel_context.detach(token)
            span.end()
        return result


def instrument(handler: ExtendedTelemetryHandler) -> list[tuple[str, str]]:
    # ``handler`` is accepted for API symmetry with the other patch modules;
    # the TASK span is created manually because ``ExtendedTelemetryHandler``
    # does not expose ``start_task`` / ``stop_task``.
    del handler
    wrap_function_wrapper(_MODULE, _ATTR, _TaskToolWrapper())
    return [(_MODULE, _ATTR)]
