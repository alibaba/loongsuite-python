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

"""Wrap ``SubagentExecutor._aexecute`` as an AGENT (subagent) span.

DeerFlow ``executor.py:749/845`` uses ``contextvars.copy_context()`` to
propagate the parent context into the subagent's isolated event loop, so the
OTel Context (including the parent span) follows automatically — no manual
cross-loop injection is needed here.
"""

from __future__ import annotations

import logging
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.deer_flow.utils import (
    DEER_FLOW_COMPONENT,
    DEER_FLOW_OPERATION,
    DEER_FLOW_PROVIDER,
    _normalize_subagent_name,
    _safe_call,
    _should_capture_content,
    _snapshot_token_records,
)
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    MessagePart,
    Text,
)

logger = logging.getLogger(__name__)

_MODULE = "deerflow.subagents.executor"
_ATTR = "SubagentExecutor._aexecute"
OP_NAME = "subagent.invoke"


def _config_name(instance: Any) -> str | None:
    config = getattr(instance, "config", None)
    if config is None:
        return None
    return getattr(config, "name", None)


def _build_invocation(instance: Any, task: str) -> InvokeAgentInvocation | None:
    name = _config_name(instance) or "subagent"
    normalized = _normalize_subagent_name(name)
    attributes: dict[str, Any] = {
        DEER_FLOW_OPERATION: OP_NAME,
        DEER_FLOW_COMPONENT: "subagent",
    }
    input_messages: list[InputMessage] = []
    if _should_capture_content() and task:
        input_messages.append(
            InputMessage(role="user", parts=[Text(content=task)])
        )
    return InvokeAgentInvocation(
        provider=DEER_FLOW_PROVIDER,
        agent_name=name,
        agent_id=f"subagent:{normalized}",
        conversation_id=getattr(instance, "thread_id", None),
        agent_description=task if _should_capture_content() else None,
        input_messages=input_messages,
        attributes=attributes,
    )


class _SubagentAExecuteWrapper:
    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        task = args[0] if args else kwargs.get("task", "")
        invocation = _safe_call(
            "build_subagent_invocation", _build_invocation, instance, task
        )
        if invocation is None:
            return await wrapped(*args, **kwargs)

        started = _safe_call(
            "start_invoke_agent", self._handler.start_invoke_agent, invocation
        )
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            if started:
                _safe_call(
                    "fail_invoke_agent",
                    self._handler.fail_invoke_agent,
                    invocation,
                    Error(message=str(exc) or type(exc).__name__, type=type(exc)),
                )
                span = getattr(invocation, "span", None)
                if span is not None and span.is_recording():
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        # Pull token usage from the executor's token collector (if any) and
        # attach it to the invocation so the metrics recorder can emit
        # ``genai_llm_usage_tokens``.
        collector = getattr(instance, "_token_collector", None)
        if collector is None:
            # DeerFlow sets up ``SubagentTokenCollector`` locally in
            # ``_aexecute``; we cannot reach it from the wrapper. Try the
            # public attribute name as a fallback.
            collector = getattr(instance, "token_collector", None)
        if collector is not None:
            usage = _snapshot_token_records(collector)
            if usage:
                if "input_tokens" in usage:
                    invocation.input_tokens = usage["input_tokens"]
                if "output_tokens" in usage:
                    invocation.output_tokens = usage["output_tokens"]

        if started:
            _safe_call(
                "stop_invoke_agent", self._handler.stop_invoke_agent, invocation
            )
        return result


def instrument(handler: ExtendedTelemetryHandler) -> list[tuple[str, str]]:
    wrap_function_wrapper(_MODULE, _ATTR, _SubagentAExecuteWrapper(handler))
    return [(_MODULE, _ATTR)]
