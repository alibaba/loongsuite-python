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

"""Wrap ``deerflow.runtime.runs.worker.run_agent`` as an ENTRY span."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.deer_flow.utils import (
    DEER_FLOW_COMPONENT,
    DEER_FLOW_OPERATION,
    DEER_FLOW_PROVIDER,
    _call_arg,
    _extract_human_content,
    _non_empty_string,
    _safe_call,
    _should_capture_content,
)
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.types import Error, InputMessage, MessagePart, Text

logger = logging.getLogger(__name__)

# Recursion guard mirrors crewai's ``_ENTRY_DEPTH`` pattern so that nested
# ``run_agent`` calls (e.g. when task_tool triggers a subagent that itself
# enters run_agent) do not produce stacked ENTRY spans.
_ENTRY_DEPTH: ContextVar[int] = ContextVar("deer_flow_entry_depth", default=0)

_MODULE = "deerflow.runtime.runs.worker"
_ATTR = "run_agent"
OP_NAME = "run_agent"


def _record_thread_id(record: Any) -> str | None:
    return _non_empty_string(getattr(record, "thread_id", None))


def _record_user_id() -> str | None:
    try:
        from deerflow.runtime.user_context import get_effective_user_id
    except Exception:
        return None
    return _safe_call("get_effective_user_id", get_effective_user_id)


def _build_entry_invocation(record: Any, graph_input: Any) -> EntryInvocation | None:
    attributes: dict[str, Any] = {
        DEER_FLOW_OPERATION: OP_NAME,
        DEER_FLOW_COMPONENT: "entry",
    }
    session_id = _record_thread_id(record)
    user_id = _record_user_id()
    input_messages: list[InputMessage] = []
    if _should_capture_content():
        content = _extract_human_content(graph_input)
        if content:
            input_messages.append(
                InputMessage(role="user", parts=[Text(content=content)])
            )
    return EntryInvocation(
        session_id=session_id,
        user_id=user_id,
        input_messages=input_messages,
        attributes=attributes,
    )


class _RunAgentWrapper:
    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        if _ENTRY_DEPTH.get() > 0:
            return await wrapped(*args, **kwargs)

        record = _call_arg(args, kwargs, 2, "record")
        graph_input = kwargs.get("graph_input") or _call_arg(args, kwargs, 5, "graph_input")
        invocation = _safe_call(
            "build_entry_invocation", _build_entry_invocation, record, graph_input
        )
        if invocation is None:
            return await wrapped(*args, **kwargs)

        token: Token[int] = _ENTRY_DEPTH.set(_ENTRY_DEPTH.get() + 1)
        started = _safe_call("start_entry", self._handler.start_entry, invocation)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            if started:
                _safe_call(
                    "fail_entry",
                    self._handler.fail_entry,
                    invocation,
                    Error(message=str(exc) or type(exc).__name__, type=type(exc)),
                )
                span = getattr(invocation, "span", None)
                if span is not None and span.is_recording():
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
            _ENTRY_DEPTH.reset(token)
            raise

        if started:
            _safe_call("stop_entry", self._handler.stop_entry, invocation)
        _ENTRY_DEPTH.reset(token)
        return result


def instrument(handler: ExtendedTelemetryHandler) -> list[tuple[str, str]]:
    wrap_function_wrapper(_MODULE, _ATTR, _RunAgentWrapper(handler))
    return [(_MODULE, _ATTR)]
