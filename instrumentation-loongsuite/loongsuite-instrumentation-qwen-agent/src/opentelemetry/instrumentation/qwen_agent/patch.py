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

"""Patch functions for Qwen-Agent instrumentation.

Wraps key qwen-agent methods to generate OpenTelemetry spans:
- Agent.run() -> invoke_agent spans
  (Agent.run_nonstream() is NOT wrapped separately; it calls self.run()
   internally, so a single invoke_agent span is produced by this wrapper.)
- BaseChatModel.chat() -> LLM spans
- Agent._call_tool() -> execute_tool spans
- Agent._call_llm() -> react step spans (only for ReAct agents with tools)
"""

from __future__ import annotations

import logging
import timeit
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import Error

from .utils import (
    _apply_usage_to_llm_invocation,
    _convert_qwen_agent_final_output_messages,
    _convert_qwen_messages_to_output_messages,
    _create_agent_invocation,
    _create_llm_invocation,
    _create_tool_invocation,
)

logger = logging.getLogger(__name__)

# ContextVar guards for ReAct step tracking.
# _react_mode: True when the current agent run has tools (ReAct-capable).
# _react_step_invocation: the currently active (open) ReactStepInvocation.
# _react_step_counter: 1-based step counter within the current agent run.
_react_mode: ContextVar[bool] = ContextVar("qwen_react_mode", default=False)
_react_step_invocation: ContextVar[Optional[ReactStepInvocation]] = ContextVar(
    "qwen_react_step_invocation", default=None
)
_react_step_counter: ContextVar[int] = ContextVar(
    "qwen_react_step_counter", default=0
)
_active_agent_invocations: ContextVar[tuple[Any, ...]] = ContextVar(
    "qwen_active_agent_invocations", default=()
)

# Reentrancy guards to prevent duplicate spans when Agent/BaseChatModel
# are abstract classes and subclass calls super() (Proxy/Wrapper scenarios).
_agent_run_instance_stack: ContextVar[tuple[int, ...]] = ContextVar(
    "_qwen_agent_run_instance_stack", default=()
)
_in_chat: ContextVar[bool] = ContextVar("_qwen_in_chat", default=False)
_in_call_tool: ContextVar[bool] = ContextVar(
    "_qwen_in_call_tool", default=False
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


def _accumulate_llm_usage_on_active_agents(invocation: Any) -> None:
    """Roll up child LLM token usage onto active invoke_agent spans.

    The rollup is intentionally transitive: a parent agent records the total
    nested LLM cost of its run, so consumers should not sum agent spans to
    calculate global token usage.
    """
    active_agents = _active_agent_invocations.get()
    if not active_agents:
        return

    for active_agent in active_agents:
        if getattr(invocation, "input_tokens", None) is not None:
            active_agent.input_tokens = (active_agent.input_tokens or 0) + (
                invocation.input_tokens or 0
            )
        if getattr(invocation, "output_tokens", None) is not None:
            active_agent.output_tokens = (active_agent.output_tokens or 0) + (
                invocation.output_tokens or 0
            )
        if (
            getattr(invocation, "usage_cache_read_input_tokens", None)
            is not None
        ):
            active_agent.usage_cache_read_input_tokens = (
                active_agent.usage_cache_read_input_tokens or 0
            ) + (invocation.usage_cache_read_input_tokens or 0)
        if (
            getattr(invocation, "usage_cache_creation_input_tokens", None)
            is not None
        ):
            active_agent.usage_cache_creation_input_tokens = (
                active_agent.usage_cache_creation_input_tokens or 0
            ) + (invocation.usage_cache_creation_input_tokens or 0)


def wrap_agent_run(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for Agent.run() to create invoke_agent spans.

    Agent.run() is a generator that yields List[Message].
    We wrap it to create an agent span covering the full execution.

    Also sets up the ReAct mode guard: if the agent has tools in its
    function_map, _react_mode is set to True so that wrap_agent_call_llm
    will create react_step spans for each ReAct iteration.
    """
    # Reentrancy guard: prevent duplicate spans in Proxy/Wrapper scenarios
    # where a subclass calls super().run().
    run_stack = _agent_run_instance_stack.get()
    instance_id = id(instance)
    if instance_id in run_stack:
        yield from wrapped(*args, **kwargs)
        return
    run_token = _agent_run_instance_stack.set(run_stack + (instance_id,))

    messages = args[0] if args else kwargs.get("messages", [])

    try:
        invocation = _create_agent_invocation(instance, messages)
    except Exception as e:
        logger.debug(f"Failed to create agent invocation: {e}")
        _agent_run_instance_stack.reset(run_token)
        yield from wrapped(*args, **kwargs)
        return

    # Set up ReAct mode guard: only agents with tools get react_step spans.
    is_react = bool(getattr(instance, "function_map", {}))
    mode_token = _react_mode.set(is_react)
    counter_token = _react_step_counter.set(0)
    step_token = _react_step_invocation.set(None)
    active_agent_token = _active_agent_invocations.set(
        _active_agent_invocations.get() + (invocation,)
    )

    handler.start_invoke_agent(invocation)

    try:
        last_response = None
        for response in wrapped(*args, **kwargs):
            last_response = response
            yield response

        # Extract output from last yielded response
        if last_response:
            invocation.output_messages = (
                _convert_qwen_agent_final_output_messages(last_response)
            )

        # Close the last react_step span before closing invoke_agent.
        _close_active_react_step(handler)

        handler.stop_invoke_agent(invocation)

    except GeneratorExit as e:
        # Generator was closed early (e.g., consumer stopped iterating).
        # Ensure any open react_step and the invoke_agent span are finalized.
        _close_active_react_step(handler)
        handler.fail_invoke_agent(
            invocation, Error(message=str(e), type=type(e))
        )
        raise
    except Exception as e:
        # Close any open react_step on error path too.
        _close_active_react_step(handler)
        handler.fail_invoke_agent(
            invocation, Error(message=str(e), type=type(e))
        )
        raise
    finally:
        # Restore ContextVars to pre-run state.
        _react_step_counter.reset(counter_token)
        _react_step_invocation.reset(step_token)
        _react_mode.reset(mode_token)
        _active_agent_invocations.reset(active_agent_token)
        _agent_run_instance_stack.reset(run_token)


def wrap_chat_model_chat(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for BaseChatModel.chat() to create LLM spans.

    chat() can return:
    - List[Message] (non-stream)
    - Iterator[List[Message]] (stream)
    """
    # Reentrancy guard: prevent duplicate spans in Proxy/Wrapper scenarios
    # where a subclass calls super().chat().
    if _in_chat.get():
        return wrapped(*args, **kwargs)
    chat_token = _in_chat.set(True)

    try:
        messages = args[0] if args else kwargs.get("messages", [])
        functions = (
            kwargs.get("functions")
            if len(args) < 2
            else (args[1] if len(args) > 1 else None)
        )
        stream = kwargs.get("stream", True)
        extra_generate_cfg = kwargs.get("extra_generate_cfg")

        try:
            invocation = _create_llm_invocation(
                instance, messages, functions, stream, extra_generate_cfg
            )
        except Exception as e:
            logger.debug(f"Failed to create LLM invocation: {e}")
            return wrapped(*args, **kwargs)

        handler.start_llm(invocation)

        try:
            result = wrapped(*args, **kwargs)

            if (
                stream
                and hasattr(result, "__iter__")
                and not isinstance(result, list)
            ):
                # Streaming: wrap the iterator
                return _wrap_streaming_llm_response(
                    result, invocation, handler
                )
            else:
                # Non-streaming: result is List[Message]
                if result:
                    _apply_usage_to_llm_invocation(invocation, result)
                    invocation.output_messages = (
                        _convert_qwen_messages_to_output_messages(result)
                    )
                    invocation.response_model_name = invocation.request_model
                    invocation.finish_reasons = ["stop"]

                    # Check for function calls in output
                    for msg in result:
                        fc = (
                            msg.function_call
                            if hasattr(msg, "function_call")
                            else msg.get("function_call")
                            if isinstance(msg, dict)
                            else None
                        )
                        if fc:
                            invocation.finish_reasons = ["tool_calls"]
                            break

                _accumulate_llm_usage_on_active_agents(invocation)
                handler.stop_llm(invocation)
                return result

        except Exception as e:
            handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
            raise
    finally:
        _in_chat.reset(chat_token)


def _wrap_streaming_llm_response(
    response_iter: Iterator, invocation: Any, handler: ExtendedTelemetryHandler
) -> Iterator:
    """Wrap a streaming LLM response iterator to capture output on completion."""
    try:
        last_response = None
        first_token = True
        for response in response_iter:
            if first_token:
                invocation.monotonic_first_token_s = timeit.default_timer()
                first_token = False
            _apply_usage_to_llm_invocation(invocation, response)
            last_response = response
            yield response

        if last_response:
            invocation.output_messages = (
                _convert_qwen_messages_to_output_messages(last_response)
            )
            invocation.response_model_name = invocation.request_model
            invocation.finish_reasons = ["stop"]

            # Check for function calls
            for msg in last_response:
                fc = (
                    msg.function_call
                    if hasattr(msg, "function_call")
                    else msg.get("function_call")
                    if isinstance(msg, dict)
                    else None
                )
                if fc:
                    invocation.finish_reasons = ["tool_calls"]
                    break

        _accumulate_llm_usage_on_active_agents(invocation)
        handler.stop_llm(invocation)

    except GeneratorExit as e:
        # Stream was closed early (e.g., consumer stopped iterating).
        # Ensure the LLM span is finalized.
        handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
        raise
    except Exception as e:
        handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
        raise


def wrap_agent_call_llm(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for Agent._call_llm() to create react_step spans.

    Only creates react_step spans when _react_mode is True (i.e. the
    current agent has tools in its function_map).  This ensures that
    simple agents (no tools) are completely unaffected.

    Each call to _call_llm corresponds to one iteration of the ReAct
    while-loop.  The react_step span is NOT closed here — it stays
    open so that subsequent _call_tool invocations become children of
    this react_step.  The span is closed either:
    - by the next wrap_agent_call_llm call (start of next iteration), or
    - by wrap_agent_run when the agent run finishes.
    """
    if not _react_mode.get():
        # Not a ReAct agent — transparent pass-through.
        return wrapped(*args, **kwargs)

    # Close the previous react_step (if any) before starting a new one.
    _close_active_react_step(handler)

    # Increment step counter (1-based).
    step_num = _react_step_counter.get() + 1
    _react_step_counter.set(step_num)

    step_invocation = ReactStepInvocation(round=step_num)

    try:
        handler.start_react_step(step_invocation)
    except Exception as e:
        logger.debug(f"Failed to start react step: {e}")
        return wrapped(*args, **kwargs)

    _react_step_invocation.set(step_invocation)

    # Call original _call_llm — its return value is a generator (or list).
    # The chat span created inside will be a child of this react_step
    # because start_react_step attached it to the current context.
    return wrapped(*args, **kwargs)


def wrap_agent_call_tool(
    wrapped, instance, args, kwargs, handler: ExtendedTelemetryHandler
):
    """Wrapper for Agent._call_tool() to create execute_tool spans.

    _call_tool(tool_name, tool_args, **kwargs) -> str | List[ContentItem]
    """
    # Reentrancy guard: prevent duplicate spans in Proxy/Wrapper scenarios
    # where a subclass calls super()._call_tool().
    if _in_call_tool.get():
        return wrapped(*args, **kwargs)
    tool_guard_token = _in_call_tool.set(True)

    try:
        tool_name = (
            args[0] if args else kwargs.get("tool_name", "unknown_tool")
        )
        tool_args = args[1] if len(args) > 1 else kwargs.get("tool_args", "{}")

        # Get tool instance for description
        tool_instance = None
        if hasattr(instance, "function_map"):
            tool_instance = instance.function_map.get(tool_name)

        try:
            invocation = _create_tool_invocation(
                tool_name, tool_args, tool_instance
            )
        except Exception as e:
            logger.debug(f"Failed to create tool invocation: {e}")
            return wrapped(*args, **kwargs)

        handler.start_execute_tool(invocation)

        try:
            result = wrapped(*args, **kwargs)

            # Set tool result
            if isinstance(result, str):
                invocation.tool_call_result = result
            elif isinstance(result, list):
                # List[ContentItem] - serialize to string
                invocation.tool_call_result = str(result)
            else:
                invocation.tool_call_result = str(result) if result else None

            handler.stop_execute_tool(invocation)
            return result

        except Exception as e:
            handler.fail_execute_tool(
                invocation, Error(message=str(e), type=type(e))
            )
            raise
    finally:
        _in_call_tool.reset(tool_guard_token)
