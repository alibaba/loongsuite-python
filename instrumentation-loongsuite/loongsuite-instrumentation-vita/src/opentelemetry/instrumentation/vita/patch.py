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

"""Patch functions for VitaBench instrumentation.

Wraps key vitabench methods to generate OpenTelemetry spans:
- run_task() -> ENTRY spans
- Orchestrator.run() -> CHAIN spans
- Orchestrator.step() -> STEP spans (react)
- LLMAgent.generate_next_message() -> AGENT spans
- generate() -> LLM spans
- Environment.get_response() -> TOOL spans
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from typing import Optional

from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

from .utils import (
    _MAX_CONTENT_LEN,
    _convert_vita_assistant_to_output,
    _convert_vita_messages_to_input,
    _get_tool_definitions,
    _infer_provider,
)

logger = logging.getLogger(__name__)

# ContextVars for ReAct step tracking
_react_step_invocation: ContextVar[Optional[ReactStepInvocation]] = ContextVar(
    "vita_react_step_invocation", default=None
)
_react_step_counter: ContextVar[int] = ContextVar(
    "vita_react_step_counter", default=0
)

# Reentrancy guard for AGENT span (LLMSoloAgent extends LLMAgent)
_in_agent_invoke: ContextVar[bool] = ContextVar(
    "vita_in_agent_invoke", default=False
)


def _close_active_react_step(handler: ExtendedTelemetryHandler) -> None:
    """Close the currently active react_step span, if any."""
    prev = _react_step_invocation.get()
    if prev is not None:
        try:
            handler.stop_react_step(prev)
        except Exception as e:
            logger.debug(f"Failed to close react step: {e}")
        _react_step_invocation.set(None)


# ==================== Hook #1: run_task -> ENTRY ====================


def wrap_run_task(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for vita.run.run_task to create ENTRY span."""
    task = args[1] if len(args) > 1 else kwargs.get("task")
    args[0] if args else kwargs.get("domain")

    invocation = EntryInvocation(
        session_id=str(uuid.uuid4()),
        user_id=None,
    )
    invocation.attributes["gen_ai.framework"] = "vitabench"

    if task and hasattr(task, "instructions") and task.instructions:
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[
                    Text(content=str(task.instructions)[:_MAX_CONTENT_LEN])
                ],
            )
        ]

    handler.start_entry(invocation)
    try:
        result = wrapped(*args, **kwargs)

        if result:
            output_parts = []
            if (
                hasattr(result, "termination_reason")
                and result.termination_reason
            ):
                output_parts.append(
                    Text(content=f"termination: {result.termination_reason}")
                )
            if hasattr(result, "reward_info") and result.reward_info:
                reward = getattr(result.reward_info, "reward", None)
                if reward is not None:
                    output_parts.append(Text(content=f"reward: {reward}"))
            if output_parts:
                invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=output_parts,
                        finish_reason="stop",
                    )
                ]

        handler.stop_entry(invocation)
        return result
    except Exception as e:
        handler.fail_entry(invocation, Error(message=str(e), type=type(e)))
        raise


# ==================== Hook #2: Orchestrator.run -> CHAIN ====================


def wrap_orchestrator_run(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for Orchestrator.run to create CHAIN span."""
    task = getattr(instance, "task", None)
    domain = getattr(instance, "domain", "unknown")
    span_name = f"workflow {domain}"

    input_text = ""
    if task and hasattr(task, "instructions") and task.instructions:
        input_text = str(task.instructions)[:_MAX_CONTENT_LEN]

    tracer = handler._tracer

    # Reset step counter for this orchestrator run
    counter_token = _react_step_counter.set(0)
    step_token = _react_step_invocation.set(None)

    with tracer.start_as_current_span(
        name=span_name,
        kind=SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "workflow",
            "gen_ai.system": "vitabench",
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: "CHAIN",
            "gen_ai.framework": "vitabench",
        },
    ) as span:
        if input_text:
            span.set_attribute("input.value", input_text)

        try:
            result = wrapped(*args, **kwargs)

            # Close any remaining open step span
            _close_active_react_step(handler)

            if (
                result
                and hasattr(result, "termination_reason")
                and result.termination_reason
            ):
                span.set_attribute(
                    "output.value", str(result.termination_reason)
                )

            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            # Close any remaining open step span
            _close_active_react_step(handler)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            _react_step_counter.reset(counter_token)
            _react_step_invocation.reset(step_token)


# ==================== Hook #3: Orchestrator.step -> STEP ====================


def wrap_orchestrator_step(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for Orchestrator.step to create STEP span on AGENT turns."""
    to_role = getattr(instance, "to_role", None)

    # Import Role enum dynamically to avoid import-time dependency
    _Role = None
    try:
        from vita.orchestrator.orchestrator import Role

        _Role = Role
    except ImportError:
        pass

    is_agent_turn = False
    if _Role is not None:
        is_agent_turn = to_role == _Role.AGENT
    else:
        is_agent_turn = str(to_role) == "Role.AGENT" or str(to_role) == "agent"

    if is_agent_turn:
        # Close previous STEP span (deferred close strategy)
        _close_active_react_step(handler)

        step_num = _react_step_counter.get() + 1
        _react_step_counter.set(step_num)

        step_inv = ReactStepInvocation(round=step_num)
        handler.start_react_step(step_inv)
        _react_step_invocation.set(step_inv)

    try:
        result = wrapped(*args, **kwargs)

        if is_agent_turn:
            current_step = _react_step_invocation.get()
            if current_step:
                done = getattr(instance, "done", False)
                if done:
                    term_reason = getattr(instance, "termination_reason", None)
                    if term_reason:
                        current_step.finish_reason = (
                            term_reason.value
                            if hasattr(term_reason, "value")
                            else str(term_reason)
                        )
                    else:
                        current_step.finish_reason = "agent_stop"
                else:
                    message = getattr(instance, "message", None)
                    if (
                        message
                        and hasattr(message, "is_tool_call")
                        and message.is_tool_call()
                    ):
                        current_step.finish_reason = "tool_call"
                    else:
                        current_step.finish_reason = "assistant_text"

        return result
    except Exception as e:
        current_step = _react_step_invocation.get()
        if current_step:
            current_step.finish_reason = "error"
            handler.fail_react_step(
                current_step, Error(message=str(e), type=type(e))
            )
            _react_step_invocation.set(None)
        raise


# ==================== Hook #4: generate_next_message -> AGENT ====================


def wrap_generate_next_message(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for LLMAgent.generate_next_message / LLMSoloAgent.generate_next_message."""
    # Reentrancy guard
    if _in_agent_invoke.get():
        return wrapped(*args, **kwargs)
    token = _in_agent_invoke.set(True)

    try:
        agent_name = instance.__class__.__name__
        model = getattr(instance, "llm", None)

        invocation = InvokeAgentInvocation(
            provider="vitabench",
            agent_name=agent_name,
            request_model=model,
        )

        # input_messages
        message = args[0] if args else kwargs.get("message")
        state = args[1] if len(args) > 1 else kwargs.get("state")
        if message:
            invocation.input_messages = _convert_vita_messages_to_input(
                [message]
            )

        # system_instruction
        if (
            state
            and hasattr(state, "system_messages")
            and state.system_messages
        ):
            invocation.system_instruction = [
                Text(content=str(sm.content)[:_MAX_CONTENT_LEN])
                for sm in state.system_messages
                if sm and getattr(sm, "content", None)
            ]

        # tool_definitions
        tools = getattr(instance, "tools", None)
        tool_defs = _get_tool_definitions(tools)
        if tool_defs:
            invocation.tool_definitions = tool_defs

        handler.start_invoke_agent(invocation)

        try:
            result = wrapped(*args, **kwargs)
            assistant_msg, _ = result

            # output_messages
            invocation.output_messages = _convert_vita_assistant_to_output(
                assistant_msg
            )

            # token usage
            usage = getattr(assistant_msg, "usage", None)
            if usage and isinstance(usage, dict):
                invocation.input_tokens = usage.get("prompt_tokens")
                invocation.output_tokens = usage.get("completion_tokens")

            handler.stop_invoke_agent(invocation)
            return result
        except Exception as e:
            handler.fail_invoke_agent(
                invocation, Error(message=str(e), type=type(e))
            )
            raise
    finally:
        _in_agent_invoke.reset(token)


# ==================== Hook #5: generate -> LLM ====================


def wrap_generate(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for vita.utils.llm_utils.generate to create LLM span."""
    model = args[0] if args else kwargs.get("model", "unknown")
    messages = args[1] if len(args) > 1 else kwargs.get("messages", [])
    tools = args[2] if len(args) > 2 else kwargs.get("tools")
    temperature = kwargs.get("temperature")

    invocation = LLMInvocation(
        request_model=model or "unknown",
        provider=_infer_provider(model or ""),
        temperature=temperature,
    )
    invocation.max_tokens = kwargs.get("max_tokens")

    # input_messages
    invocation.input_messages = _convert_vita_messages_to_input(messages)

    # tool_definitions
    tool_defs = _get_tool_definitions(tools)
    if tool_defs:
        invocation.tool_definitions = tool_defs

    handler.start_llm(invocation)

    try:
        result = wrapped(*args, **kwargs)

        if result:
            # output_messages
            invocation.output_messages = _convert_vita_assistant_to_output(
                result
            )

            # response_model_name
            invocation.response_model_name = model

            # finish_reasons
            if getattr(result, "tool_calls", None):
                invocation.finish_reasons = ["tool_calls"]
            else:
                invocation.finish_reasons = ["stop"]

            # token usage
            usage = getattr(result, "usage", None)
            if usage and isinstance(usage, dict):
                invocation.input_tokens = usage.get("prompt_tokens")
                invocation.output_tokens = usage.get("completion_tokens")

        handler.stop_llm(invocation)
        return result
    except Exception as e:
        handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
        raise


# ==================== Hook #6: Environment.get_response -> TOOL ====================


def wrap_get_response(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for Environment.get_response to create TOOL span."""
    message = args[0] if args else kwargs.get("message")

    tool_name = getattr(message, "name", "unknown") if message else "unknown"
    tool_call_id = getattr(message, "id", None) if message else None

    invocation = ExecuteToolInvocation(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        provider="vitabench",
    )

    # tool_call_arguments
    if message and hasattr(message, "arguments") and message.arguments:
        try:
            invocation.tool_call_arguments = json.dumps(
                message.arguments, ensure_ascii=False, default=str
            )[:_MAX_CONTENT_LEN]
        except Exception:
            invocation.tool_call_arguments = str(message.arguments)[
                :_MAX_CONTENT_LEN
            ]

    handler.start_execute_tool(invocation)

    try:
        result = wrapped(*args, **kwargs)

        # tool_call_result
        if result and getattr(result, "content", None):
            invocation.tool_call_result = str(result.content)[
                :_MAX_CONTENT_LEN
            ]

        # Check if tool reported an error
        if result and getattr(result, "error", False):
            handler.fail_execute_tool(
                invocation,
                Error(
                    message=f"Tool error: {getattr(result, 'content', '')}",
                    type=RuntimeError,
                ),
            )
        else:
            handler.stop_execute_tool(invocation)

        return result
    except Exception as e:
        handler.fail_execute_tool(
            invocation, Error(message=str(e), type=type(e))
        )
        raise
