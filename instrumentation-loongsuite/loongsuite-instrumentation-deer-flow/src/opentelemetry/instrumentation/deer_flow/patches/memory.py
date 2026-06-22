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

"""Wrap DeerFlow Memory APIs as TASK spans.

* ``FileMemoryStorage.load`` / ``save`` (sync) — ``gen_ai.task.name =
  memory.load`` / ``memory.save``
* ``MemoryUpdater.aupdate_memory`` (async) — ``gen_ai.task.name = memory.update``

Memory content contains PII and is gated by a two-level switch:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` (the standard gen-ai
content capture flag) AND ``OTEL_DEER_FLOW_CAPTURE_MEMORY_CONTENT`` (deer-flow
specific, default false). Both must be true for ``input.value`` / ``output.value``
to be emitted.
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
    _should_capture_memory_content,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, get_tracer
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

logger = logging.getLogger(__name__)

_STORAGE_MODULE = "deerflow.agents.memory.storage"
_STORAGE_CLASS = "FileMemoryStorage"
_UPDATER_MODULE = "deerflow.agents.memory.updater"
_UPDATER_CLASS = "MemoryUpdater"


def _capture_memory() -> bool:
    return _should_capture_content() and _should_capture_memory_content()


def _serialize(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


class _MemoryLoadSaveWrapper:
    """Wrap ``FileMemoryStorage.load`` / ``save`` as TASK spans."""

    def __init__(self, tracer: Any):
        self._tracer = tracer

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        method_name = getattr(wrapped, "__name__", "memory_op")
        task_name = f"memory.{method_name}"
        span_name = f"run_task {task_name}"
        span = self._tracer.start_span(
            name=span_name, kind=SpanKind.INTERNAL
        )
        span.set_attribute(GEN_AI_SPAN_KIND, "TASK")
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
        span.set_attribute(DEER_FLOW_OPERATION, f"memory.{method_name}")
        span.set_attribute(DEER_FLOW_COMPONENT, "memory")
        span.set_attribute(DEER_FLOW_TASK_NAME, task_name)

        agent_name = kwargs.get("agent_name") if kwargs else None
        if agent_name is None and args:
            agent_name = args[0]
        if agent_name is not None:
            span.set_attribute("gen_ai.agent.name", str(agent_name))

        user_id = kwargs.get("user_id") if kwargs else None
        if user_id is not None:
            span.set_attribute("gen_ai.user.id", str(user_id))

        if _capture_memory():
            _safe_call(
                "set_input_value",
                lambda: span.set_attribute(
                    "input.value",
                    _serialize({"args": list(args), "kwargs": dict(kwargs)})
                    or "{}",
                ),
            )

        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            if _capture_memory():
                _safe_call(
                    "set_output_value",
                    lambda: span.set_attribute(
                        "output.value", _serialize(result) or ""
                    ),
                )
            span.end()
        return result


class _MemoryUpdaterWrapper:
    """Wrap ``MemoryUpdater.aupdate_memory`` as a TASK span."""

    def __init__(self, tracer: Any):
        self._tracer = tracer

    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        method_name = "update"
        task_name = f"memory.{method_name}"
        span_name = f"run_task {task_name}"
        span = self._tracer.start_span(
            name=span_name, kind=SpanKind.INTERNAL
        )
        span.set_attribute(GEN_AI_SPAN_KIND, "TASK")
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
        span.set_attribute(DEER_FLOW_OPERATION, "memory.update")
        span.set_attribute(DEER_FLOW_COMPONENT, "memory")
        span.set_attribute(DEER_FLOW_TASK_NAME, task_name)

        thread_id = kwargs.get("thread_id") if kwargs else None
        if thread_id is None and len(args) >= 2:
            thread_id = args[1]
        if thread_id is not None:
            span.set_attribute("gen_ai.session.id", str(thread_id))

        agent_name = kwargs.get("agent_name") if kwargs else None
        if agent_name is None and len(args) >= 3:
            agent_name = args[2]
        if agent_name is not None:
            span.set_attribute("gen_ai.agent.name", str(agent_name))

        if _capture_memory():
            _safe_call(
                "set_input_value",
                lambda: span.set_attribute(
                    "input.value",
                    _serialize({"thread_id": thread_id, "agent_name": agent_name})
                    or "{}",
                ),
            )

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            if _capture_memory():
                _safe_call(
                    "set_output_value",
                    lambda: span.set_attribute(
                        "output.value", _serialize(result) or ""
                    ),
                )
            span.end()
        return result


def instrument(handler: ExtendedTelemetryHandler) -> list[tuple[str, str]]:
    del handler  # TASK spans are created manually; handler not used here.
    tracer = get_tracer("opentelemetry.instrumentation.deer_flow")

    targets: list[tuple[str, str]] = []
    for method_name in ("load", "save"):
        attr = f"{_STORAGE_CLASS}.{method_name}"
        try:
            wrap_function_wrapper(
                _STORAGE_MODULE, attr, _MemoryLoadSaveWrapper(tracer)
            )
            targets.append((_STORAGE_MODULE, attr))
        except Exception as exc:
            logger.debug(
                "DeerFlow: could not wrap %s.%s: %s",
                _STORAGE_MODULE,
                attr,
                exc,
            )

    attr = f"{_UPDATER_CLASS}.aupdate_memory"
    try:
        wrap_function_wrapper(_UPDATER_MODULE, attr, _MemoryUpdaterWrapper(tracer))
        targets.append((_UPDATER_MODULE, attr))
    except Exception as exc:
        logger.debug(
            "DeerFlow: could not wrap %s.%s: %s",
            _UPDATER_MODULE,
            attr,
            exc,
        )

    return targets
