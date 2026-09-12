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

"""Wrapt wrappers used by Hermes instrumentation."""

from __future__ import annotations

import contextvars
import timeit
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import Error

from .helpers import (
    agent_output_messages,
    apply_skill_attributes,
    create_agent_invocation,
    create_entry_invocation,
    create_llm_invocation,
    create_tool_invocation,
    provider_name,
    push_state,
    reset_state,
    resolve_entry_platform,
    start_step,
    state,
    step_finish_reason,
    update_llm_invocation_from_response,
)

_ACTIVE_TOOL_NAMES = contextvars.ContextVar(
    "opentelemetry_hermes_active_tool_names",
    default=(),
)
_GENAI_SPAN_KINDS = {"ENTRY", "AGENT", "STEP", "LLM", "TOOL"}
_GENAI_SPAN_NAME_PREFIXES = (
    "enter_ai_application_system",
    "invoke_agent",
    "execute_tool",
    "react step",
    "chat ",
    "generate_content ",
    "text_completion ",
    "embedding ",
    "embeddings ",
)


def _resolve_handler(
    primary: Any | None = None,
    *,
    handler: ExtendedTelemetryHandler | None = None,
) -> ExtendedTelemetryHandler:
    if handler is not None:
        return handler
    if isinstance(primary, ExtendedTelemetryHandler):
        return primary
    return ExtendedTelemetryHandler()


def _bind_handler(instance: Any, handler: ExtendedTelemetryHandler) -> None:
    if instance is None:
        return
    state(instance)["handler"] = handler


def _current_span_is_genai_operation() -> bool:
    current_span = trace_api.get_current_span()
    span_name = str(getattr(current_span, "name", None) or "")
    if span_name.startswith(_GENAI_SPAN_NAME_PREFIXES):
        return True
    attributes = getattr(current_span, "attributes", None)
    if isinstance(attributes, Mapping):
        return attributes.get("gen_ai.span.kind") in _GENAI_SPAN_KINDS
    return False


def _safely_fail_invocation(fail_callback, invocation, error: Error) -> None:
    # Telemetry cleanup must not mask the original application exception.
    with suppress(Exception):
        fail_callback(invocation, error)


def _entry_ttft_ns(entry_invocation, first_token_monotonic_s):
    if (
        entry_invocation is None
        or entry_invocation.monotonic_start_s is None
        or first_token_monotonic_s is None
    ):
        return None
    return int(
        max(first_token_monotonic_s - entry_invocation.monotonic_start_s, 0)
        * 1_000_000_000
    )


def finish_step(
    instance: Any,
    finish_reason: str,
    *,
    exc: Exception | None = None,
) -> None:
    current_state = state(instance)
    invocation = current_state.get("current_step_invocation")
    handler = current_state.get("handler")
    if invocation is None or handler is None:
        return

    invocation.finish_reason = finish_reason
    current_state["last_step_finish_reason"] = finish_reason
    if exc is None:
        handler.stop_react_step(invocation)
    else:
        handler.fail_react_step(
            invocation,
            Error(message=str(exc), type=type(exc)),
        )

    current_state["current_step_invocation"] = None
    current_state["pending_step_finish_reason"] = None


class RunConversationWrapper:
    def __init__(
        self, primary, *, handler: ExtendedTelemetryHandler | None = None
    ):
        self._handler = _resolve_handler(primary, handler=handler)

    def __call__(self, wrapped, instance, args, kwargs):
        _bind_handler(instance, self._handler)
        user_message = args[0] if args else kwargs.get("user_message", "")

        state_token = push_state(instance)
        current_state = state(instance)
        entry_invocation = None
        # EntryInvocation has no source field yet; resolve the Hermes source
        # here so entry creation follows Hermes's own session source default.
        # TODO(loongsuite-hermes): pass entry_platform into EntryInvocation if
        # opentelemetry-util-genai adds a session source field.
        entry_platform = resolve_entry_platform(instance)
        if (
            entry_platform
            and not _current_span_is_genai_operation()
            and not _ACTIVE_TOOL_NAMES.get()
        ):
            entry_invocation = create_entry_invocation(
                instance, str(user_message or "")
            )
            current_state["entry_invocation"] = entry_invocation
            self._handler.start_entry(entry_invocation)

        invocation = create_agent_invocation(instance, str(user_message or ""))
        current_state["agent_invocation"] = invocation
        self._handler.start_invoke_agent(invocation)

        try:
            result = wrapped(*args, **kwargs)
            agent_finish_reason = (
                current_state.get("pending_step_finish_reason")
                or current_state.get("last_step_finish_reason")
                or "stop"
            )
            if current_state["current_step_invocation"] is not None:
                finish_step(instance, agent_finish_reason)

            output_messages = agent_output_messages(result)
            invocation.output_messages = output_messages
            invocation.finish_reasons = [agent_finish_reason]
            if entry_invocation is not None:
                entry_invocation.output_messages = output_messages

            if current_state["last_response_model"]:
                invocation.response_model_name = current_state[
                    "last_response_model"
                ]
            if current_state["last_response_id"]:
                invocation.response_id = current_state["last_response_id"]
            if current_state["input_tokens"] > 0:
                invocation.input_tokens = current_state["input_tokens"]
            if current_state["output_tokens"] > 0:
                invocation.output_tokens = current_state["output_tokens"]
            if current_state["first_token_monotonic_s"] is not None:
                invocation.monotonic_first_token_s = current_state[
                    "first_token_monotonic_s"
                ]
                if entry_invocation is not None:
                    entry_invocation.response_time_to_first_token = (
                        _entry_ttft_ns(
                            entry_invocation,
                            current_state["first_token_monotonic_s"],
                        )
                    )

            self._handler.stop_invoke_agent(invocation)
            if entry_invocation is not None:
                self._handler.stop_entry(entry_invocation)
            return result
        except Exception as exc:
            error = Error(message=str(exc), type=type(exc))
            with suppress(Exception):
                finish_step(instance, "error", exc=exc)
            _safely_fail_invocation(
                self._handler.fail_invoke_agent,
                invocation,
                error,
            )
            if entry_invocation is not None:
                _safely_fail_invocation(
                    self._handler.fail_entry,
                    entry_invocation,
                    error,
                )
            raise
        finally:
            reset_state(state_token)


class LLMCallWrapper:
    def __init__(
        self,
        primary,
        metrics=None,
        streaming: bool = False,
        *,
        handler: ExtendedTelemetryHandler | None = None,
    ):
        self._handler = _resolve_handler(primary, handler=handler)
        self._metrics = metrics
        self._streaming = streaming

    def _record_success_metrics(
        self,
        *,
        provider: str,
        model: str,
        started_at: float,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.record_llm_call(provider=provider, model=model)
        self._metrics.record_llm_duration(
            provider=provider,
            model=model,
            duration_seconds=max(timeit.default_timer() - started_at, 0.0),
        )
        self._metrics.record_llm_tokens(
            provider=provider,
            model=model,
            token_type="input",
            value=input_tokens,
        )
        self._metrics.record_llm_tokens(
            provider=provider,
            model=model,
            token_type="output",
            value=output_tokens,
        )
        self._metrics.record_llm_tokens(
            provider=provider,
            model=model,
            token_type="total",
            value=total_tokens,
        )

    def _record_error_metrics(
        self,
        *,
        provider: str,
        model: str,
        started_at: float,
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.record_llm_error(provider=provider, model=model)
        self._metrics.record_llm_duration(
            provider=provider,
            model=model,
            duration_seconds=max(timeit.default_timer() - started_at, 0.0),
        )

    def __call__(self, wrapped, instance, args, kwargs):
        _bind_handler(instance, self._handler)
        current_state = state(instance)
        if current_state["active_llm_depth"] > 0:
            return wrapped(*args, **kwargs)

        start_step(self._handler, instance, finish_step)
        current_state["active_llm_depth"] += 1

        api_kwargs = args[0] if args else kwargs.get("api_kwargs", {})
        invocation = create_llm_invocation(instance, api_kwargs)
        provider = invocation.provider or provider_name(instance)
        model = invocation.request_model or ""
        self._handler.start_llm(invocation)
        started_at = timeit.default_timer()

        try:
            if self._streaming:
                original_first_delta = kwargs.get("on_first_delta")

                def _wrapped_first_delta():
                    now = timeit.default_timer()
                    invocation.monotonic_first_token_s = now
                    if current_state["first_token_monotonic_s"] is None:
                        current_state["first_token_monotonic_s"] = now
                    if original_first_delta:
                        original_first_delta()

                kwargs["on_first_delta"] = _wrapped_first_delta

            response = wrapped(*args, **kwargs)
            input_tokens, output_tokens, total_tokens = (
                update_llm_invocation_from_response(
                    invocation, instance, response
                )
            )

            current_state["last_response_model"] = (
                invocation.response_model_name or invocation.request_model
            )
            current_state["last_response_id"] = invocation.response_id
            current_state["input_tokens"] += input_tokens
            current_state["output_tokens"] += output_tokens
            current_state["total_tokens"] += total_tokens

            if (
                current_state["first_token_monotonic_s"] is None
                and invocation.monotonic_first_token_s is not None
            ):
                current_state["first_token_monotonic_s"] = (
                    invocation.monotonic_first_token_s
                )

            normalized_step_reason = step_finish_reason(instance, response)
            current_state["pending_step_finish_reason"] = (
                normalized_step_reason
            )

            self._record_success_metrics(
                provider=provider,
                model=model,
                started_at=started_at,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            self._handler.stop_llm(invocation)
            if normalized_step_reason != "tool_calls":
                finish_step(instance, normalized_step_reason)
            return response
        except Exception as exc:
            self._record_error_metrics(
                provider=provider,
                model=model,
                started_at=started_at,
            )
            self._handler.fail_llm(
                invocation,
                Error(message=str(exc), type=type(exc)),
            )
            finish_step(instance, "error", exc=exc)
            raise
        finally:
            current_state["active_llm_depth"] = max(
                0, current_state["active_llm_depth"] - 1
            )


class ToolCallWrapper:
    def __init__(
        self, primary, *, handler: ExtendedTelemetryHandler | None = None
    ):
        self._handler = _resolve_handler(primary, handler=handler)

    def _call_with_tool_span(
        self,
        wrapped,
        instance,
        args,
        kwargs,
        *,
        function_name,
        function_args,
        tool_call_id,
    ):
        _bind_handler(instance, self._handler)
        active_tool_names = _ACTIVE_TOOL_NAMES.get()
        if function_name in active_tool_names:
            return wrapped(*args, **kwargs)

        invocation = create_tool_invocation(
            str(function_name or ""),
            arguments=function_args,
            tool_call_id=tool_call_id,
        )
        token = _ACTIVE_TOOL_NAMES.set(
            active_tool_names + (str(function_name),)
        )
        self._handler.start_execute_tool(invocation)
        try:
            result = wrapped(*args, **kwargs)
            invocation.tool_call_result = result
            apply_skill_attributes(
                invocation,
                str(function_name or ""),
                arguments=function_args,
                result=result,
            )
            self._handler.stop_execute_tool(invocation)
            return result
        except Exception as exc:
            self._handler.fail_execute_tool(
                invocation,
                Error(message=str(exc), type=type(exc)),
            )
            raise
        finally:
            _ACTIVE_TOOL_NAMES.reset(token)

    def __call__(self, wrapped, instance, args, kwargs):
        function_name = args[0] if args else kwargs.get("function_name", "")
        function_args = (
            args[1] if len(args) > 1 else kwargs.get("function_args")
        )
        tool_call_id = args[3] if len(args) > 3 else kwargs.get("tool_call_id")
        return self._call_with_tool_span(
            wrapped,
            instance,
            args,
            kwargs,
            function_name=function_name,
            function_args=function_args,
            tool_call_id=tool_call_id,
        )


class ToolDispatchWrapper(ToolCallWrapper):
    def __call__(self, wrapped, instance, args, kwargs):
        function_name = args[0] if args else kwargs.get("function_name", "")
        function_args = (
            args[1] if len(args) > 1 else kwargs.get("function_args")
        )
        tool_call_id = args[3] if len(args) > 3 else kwargs.get("tool_call_id")
        return self._call_with_tool_span(
            wrapped,
            instance,
            args,
            kwargs,
            function_name=function_name,
            function_args=function_args,
            tool_call_id=tool_call_id,
        )


class ToolBatchWrapper:
    def __init__(
        self, primary=None, *, handler: ExtendedTelemetryHandler | None = None
    ):
        self._handler = _resolve_handler(primary, handler=handler)

    def __call__(self, wrapped, instance, args, kwargs):
        _bind_handler(instance, self._handler)
        try:
            result = wrapped(*args, **kwargs)
            finish_step(instance, "tool_calls")
            return result
        except Exception as exc:
            finish_step(instance, "error", exc=exc)
            raise


class ToolExecutionWrapper:
    def __init__(
        self,
        primary,
        tool_name: str,
        *,
        handler: ExtendedTelemetryHandler | None = None,
    ):
        self._handler = _resolve_handler(primary, handler=handler)
        self._tool_name = tool_name

    def _build_arguments(self, args, kwargs):
        if self._tool_name == "memory":
            return {
                "action": kwargs.get("action")
                if "action" in kwargs
                else (args[0] if len(args) > 0 else None),
                "target": kwargs.get("target")
                if "target" in kwargs
                else (args[1] if len(args) > 1 else None),
                "content": kwargs.get("content")
                if "content" in kwargs
                else (args[2] if len(args) > 2 else None),
                "old_text": kwargs.get("old_text")
                if "old_text" in kwargs
                else (args[3] if len(args) > 3 else None),
            }
        if self._tool_name == "todo":
            return {
                "todos": kwargs.get("todos")
                if "todos" in kwargs
                else (args[0] if len(args) > 0 else None),
                "merge": kwargs.get("merge")
                if "merge" in kwargs
                else (args[1] if len(args) > 1 else None),
            }
        if self._tool_name == "session_search":
            return {
                "query": kwargs.get("query")
                if "query" in kwargs
                else (args[0] if len(args) > 0 else None),
                "role_filter": kwargs.get("role_filter")
                if "role_filter" in kwargs
                else (args[1] if len(args) > 1 else None),
                "limit": kwargs.get("limit")
                if "limit" in kwargs
                else (args[2] if len(args) > 2 else None),
            }
        if self._tool_name == "delegate_task":
            return {
                "goal": kwargs.get("goal")
                if "goal" in kwargs
                else (args[0] if len(args) > 0 else None),
                "context": kwargs.get("context")
                if "context" in kwargs
                else (args[1] if len(args) > 1 else None),
                "toolsets": kwargs.get("toolsets")
                if "toolsets" in kwargs
                else (args[2] if len(args) > 2 else None),
                "tasks": kwargs.get("tasks")
                if "tasks" in kwargs
                else (args[3] if len(args) > 3 else None),
                "max_iterations": kwargs.get("max_iterations")
                if "max_iterations" in kwargs
                else (args[4] if len(args) > 4 else None),
            }
        return None

    def __call__(self, wrapped, instance, args, kwargs):
        _bind_handler(instance, self._handler)
        active_tool_names = _ACTIVE_TOOL_NAMES.get()
        if self._tool_name in active_tool_names:
            return wrapped(*args, **kwargs)

        invocation = create_tool_invocation(
            self._tool_name,
            arguments=self._build_arguments(args, kwargs),
        )
        token = _ACTIVE_TOOL_NAMES.set(active_tool_names + (self._tool_name,))
        self._handler.start_execute_tool(invocation)
        try:
            result = wrapped(*args, **kwargs)
            invocation.tool_call_result = result
            apply_skill_attributes(
                invocation,
                self._tool_name,
                arguments=invocation.tool_call_arguments,
                result=result,
            )
            self._handler.stop_execute_tool(invocation)
            return result
        except Exception as exc:
            self._handler.fail_execute_tool(
                invocation,
                Error(message=str(exc), type=type(exc)),
            )
            raise
        finally:
            _ACTIVE_TOOL_NAMES.reset(token)
