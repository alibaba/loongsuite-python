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

execute.md §3 (R1) requires explicit OTel Context replay across the isolated
subagent event loop. Source verification (``executor.py:235-245``) shows
``_submit_to_isolated_loop_in_context`` does ``context.run(lambda:
asyncio.run_coroutine_threadsafe(coro_factory(), loop))`` — the ``context.run``
only wraps the *synchronous* submission; the coroutine body executes on the
isolated loop thread without inheriting the parent OTel Context. As a result,
the subagent AGENT span created inside ``_aexecute`` would become an orphan
instead of parenting to the TASK span emitted by ``task_tool``.

Fix: capture ``otel_context.get_current()`` at ``execute_async`` entry (where
the TASK span is still active in the caller's context) and stash it on the
executor instance; ``_aexecute`` then ``context.attach``-es that snapshot at
the start of its coroutine body before creating the AGENT span.
"""

from __future__ import annotations

import logging
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry import context as otel_context
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
    Text,
)

logger = logging.getLogger(__name__)

_MODULE = "deerflow.subagents.executor"
_EXECUTE_ASYNC_ATTR = "SubagentExecutor.execute_async"
_AEXECUTE_ATTR = "SubagentExecutor._aexecute"
OP_NAME = "subagent.invoke"

# Instance attribute used to ferry the parent OTel Context from
# ``execute_async`` (caller thread, TASK span active) to ``_aexecute``
# (isolated loop thread, default context). The leading underscore matches
# DeerFlow's private-attribute convention; wrapt does not touch it.
_OTEL_PARENT_CTX_ATTR = "_otel_parent_context"


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


class _ExecuteAsyncContextCaptureWrapper:
    """Capture the caller's OTel Context onto the executor instance.

    ``execute_async`` is called from ``task_tool`` (lead agent event loop)
    while the TASK span is still attached. The captured snapshot is replayed
    by ``_SubagentAExecuteWrapper`` inside the isolated subagent loop so the
    subagent AGENT span parents to the TASK span (execute.md §3, R1 fix).
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        try:
            setattr(instance, _OTEL_PARENT_CTX_ATTR, otel_context.get_current())
        except Exception as exc:
            logger.debug(
                "DeerFlow: failed to capture parent OTel context on %s: %s",
                type(instance).__name__,
                exc,
            )
        return wrapped(*args, **kwargs)


class _SubagentAExecuteWrapper:
    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        # Replay the parent OTel Context captured at ``execute_async`` entry.
        # ``_aexecute`` runs on the isolated subagent loop thread, where the
        # default context has no parent span; without this attach the AGENT
        # span emitted below would become an orphan (execute.md R1).
        parent_ctx = getattr(instance, _OTEL_PARENT_CTX_ATTR, None)
        replay_token = (
            otel_context.attach(parent_ctx) if parent_ctx is not None else None
        )

        task = args[0] if args else kwargs.get("task", "")
        invocation = _safe_call(
            "build_subagent_invocation", _build_invocation, instance, task
        )
        if invocation is None:
            try:
                return await wrapped(*args, **kwargs)
            finally:
                if replay_token is not None:
                    otel_context.detach(replay_token)

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
        finally:
            if replay_token is not None:
                otel_context.detach(replay_token)

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
    targets: list[tuple[str, str]] = []
    try:
        wrap_function_wrapper(
            _MODULE, _EXECUTE_ASYNC_ATTR, _ExecuteAsyncContextCaptureWrapper()
        )
        targets.append((_MODULE, _EXECUTE_ASYNC_ATTR))
    except Exception as exc:
        logger.debug(
            "DeerFlow: could not wrap %s.%s: %s",
            _MODULE,
            _EXECUTE_ASYNC_ATTR,
            exc,
        )
    wrap_function_wrapper(_MODULE, _AEXECUTE_ATTR, _SubagentAExecuteWrapper(handler))
    targets.append((_MODULE, _AEXECUTE_ATTR))
    return targets
