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

"""wrapt hooks for DefaultAgent.run / DefaultAgent.step (ARMS / util-genai semantics)."""

from __future__ import annotations

import logging
from typing import Any, Callable

from opentelemetry import context as context_api
from opentelemetry.instrumentation.minisweagent.config import (
    ENTRY_SPAN_ACTIVE,
    OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN,
)
from opentelemetry.instrumentation.minisweagent.internal.conversation import (
    build_invoke_agent_payload,
)
from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)


def _task_preview(task: str) -> str:
    if not task:
        return ""
    m = OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN
    if len(task) <= m:
        return task
    return task[: m - 3] + "..."


def _request_model_from_agent(instance: Any) -> str | None:
    model = getattr(instance, "model", None)
    if model is None:
        return None
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    mn = getattr(cfg, "model_name", None)
    if mn is not None:
        return str(mn)
    return None


def _populate_invoke_from_agent(inv: Any, instance: Any) -> None:
    try:
        payload = build_invoke_agent_payload(instance)
    except Exception:
        logger.debug("invoke_agent telemetry payload failed", exc_info=True)
        return
    inv.system_instruction = payload["system_instruction"]
    inv.input_messages = payload["input_messages"]
    inv.output_messages = payload["output_messages"]
    inv.tool_definitions = payload["tool_definitions"]


class DefaultAgentRunWrapper:
    """AGENT invoke_agent span with conversation + system_instruction + bash tool defs."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):  # noqa: ARG002 — API compatibility
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        from opentelemetry.util.genai.extended_handler import (
            get_extended_telemetry_handler,  # noqa: PLC0415
        )
        from opentelemetry.util.genai.extended_types import (  # noqa: PLC0415
            EntryInvocation,
            InvokeAgentInvocation,
        )
        from opentelemetry.util.genai.types import (  # noqa: PLC0415
            Error as GenAIError,
        )
        from opentelemetry.util.genai.types import InputMessage, Text

        task = args[0] if args else kwargs.get("task", "") or ""
        agent_name = (
            f"{instance.__class__.__module__}.{instance.__class__.__name__}"
        )

        han = get_extended_telemetry_handler()

        need_entry = not ENTRY_SPAN_ACTIVE.get()
        entry_inv = None
        entry_token = None
        if need_entry:
            entry_inv = EntryInvocation()
            if task:
                entry_inv.input_messages = [
                    InputMessage(role="user", parts=[Text(content=str(task))]),
                ]
            entry_token = ENTRY_SPAN_ACTIVE.set(True)
            han.start_entry(entry_inv, context=context_api.get_current())

        inv = InvokeAgentInvocation(
            provider="minisweagent", agent_name=agent_name
        )
        inv.request_model = _request_model_from_agent(instance)
        inv.attributes.setdefault("gen_ai.framework", "minisweagent")
        pv = _task_preview(str(task))
        if pv:
            inv.attributes["minisweagent.task.preview"] = pv

        instance._otel_msw_round = 0  # noqa: SLF001
        han.start_invoke_agent(inv, context=context_api.get_current())
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            try:
                _populate_invoke_from_agent(inv, instance)
            except Exception:
                logger.debug(
                    "populate invoke_agent on error failed", exc_info=True
                )
            if isinstance(exc, Exception):
                han.fail_invoke_agent(
                    inv, GenAIError(message=str(exc), type=type(exc))
                )
                if entry_inv is not None:
                    han.fail_entry(
                        entry_inv, GenAIError(message=str(exc), type=type(exc))
                    )
            else:
                han.stop_invoke_agent(inv)
                if entry_inv is not None:
                    han.stop_entry(entry_inv)
            if entry_token is not None:
                ENTRY_SPAN_ACTIVE.reset(entry_token)
            raise

        try:
            _populate_invoke_from_agent(inv, instance)
            if isinstance(result, dict):
                es = result.get("exit_status")
                if es is not None:
                    inv.attributes["minisweagent.exit_status"] = str(es)
                sub = result.get("submission")
                if sub is not None:
                    inv.attributes["minisweagent.submission.preview"] = (
                        _task_preview(str(sub))
                    )
        finally:
            han.stop_invoke_agent(inv)
            if entry_inv is not None:
                han.stop_entry(entry_inv)
            if entry_token is not None:
                ENTRY_SPAN_ACTIVE.reset(entry_token)
        return result


class DefaultAgentStepWrapper:
    """ReAct STEP span (gen_ai.span.kind=STEP, operation.name=react)."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):  # noqa: ARG002
        self._tracer = tracer

    @staticmethod
    def _limits_exceeded(instance: Any) -> bool:
        config = getattr(instance, "config", None)
        if config is None:
            return False
        step_limit = getattr(config, "step_limit", 0) or 0
        n_calls = getattr(instance, "n_calls", 0) or 0
        if 0 < step_limit <= n_calls:
            return True
        cost_limit = getattr(config, "cost_limit", 0) or 0
        cost = getattr(instance, "cost", 0) or 0
        if 0 < cost_limit <= cost:
            return True
        return False

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        from minisweagent.exceptions import InterruptAgentFlow  # noqa: PLC0415

        from opentelemetry.util.genai.extended_handler import (
            get_extended_telemetry_handler,  # noqa: PLC0415
        )
        from opentelemetry.util.genai.extended_types import (
            ReactStepInvocation,  # noqa: PLC0415
        )
        from opentelemetry.util.genai.types import (
            Error as GenAIError,  # noqa: PLC0415
        )

        if self._limits_exceeded(instance):
            return wrapped(*args, **kwargs)

        r = int(getattr(instance, "_otel_msw_round", 0) or 0) + 1
        instance._otel_msw_round = r  # noqa: SLF001

        han = get_extended_telemetry_handler()
        inv = ReactStepInvocation(round=r)
        han.start_react_step(inv, context=context_api.get_current())
        try:
            result = wrapped(*args, **kwargs)
        except InterruptAgentFlow as flow_exc:
            inv.finish_reason = type(flow_exc).__qualname__
            han.stop_react_step(inv)
            raise
        except BaseException as exc:
            inv.finish_reason = type(exc).__qualname__
            if isinstance(exc, Exception):
                han.fail_react_step(
                    inv, GenAIError(message=str(exc), type=type(exc))
                )
            else:
                han.stop_react_step(inv)
            raise
        else:
            han.stop_react_step(inv)
            return result
