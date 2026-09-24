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

"""Patch functions for WideSearch instrumentation.

Wraps key WideSearch methods to generate OpenTelemetry spans:
- run_single_query -> ENTRY span
- Runner.run -> AGENT span (async generator)
- Runner._step -> STEP span
- Runner._invoke_tool_call -> TOOL spans (one per tool_call)
- create_sub_agents_wrap -> TASK span (on returned closure)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar

from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.trace.status import Status
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import ReactStepInvocation
from opentelemetry.util.genai.types import Error

from .utils import (
    _create_agent_invocation,
    _create_entry_invocation,
    _create_tool_invocation,
    _extract_output_messages,
    _step_to_output_messages,
)

logger = logging.getLogger(__name__)

_step_counter: ContextVar[int] = ContextVar("ws_step_counter", default=0)
_in_run_single_query: ContextVar[bool] = ContextVar("ws_in_rsq", default=False)


async def wrap_run_single_query(
    wrapped, instance, args, kwargs, *, handler: ExtendedTelemetryHandler
):
    """H1: ENTRY span for run_single_query."""
    if _in_run_single_query.get():
        return await wrapped(*args, **kwargs)
    token = _in_run_single_query.set(True)

    query = args[0] if args else kwargs.get("query", "")
    system_prompt = kwargs.get("system_prompt") or ""
    tools_desc_kw = kwargs.get("tools_desc")
    try:
        invocation = _create_entry_invocation(
            query,
            system_prompt=system_prompt or None,
            tools_desc=(
                tools_desc_kw if isinstance(tools_desc_kw, list) else None
            ),
        )
    except Exception as e:
        logger.debug(f"Failed to create entry invocation: {e}")
        _in_run_single_query.reset(token)
        return await wrapped(*args, **kwargs)

    handler.start_entry(invocation)

    try:
        result = await wrapped(*args, **kwargs)
        invocation.output_messages = _extract_output_messages(result)
        handler.stop_entry(invocation)
        return result
    except Exception as e:
        handler.fail_entry(invocation, Error(message=str(e), type=type(e)))
        raise
    finally:
        _in_run_single_query.reset(token)


async def wrap_runner_run(
    wrapped, instance, args, kwargs, *, handler: ExtendedTelemetryHandler
):
    """H2: AGENT span for Runner.run (async generator)."""
    starting_agent = args[0] if args else kwargs.get("starting_agent")
    user_input = args[1] if len(args) > 1 else kwargs.get("user_input", "")
    memory = args[2] if len(args) > 2 else kwargs.get("memory")
    system_prompt = getattr(memory, "system_instructions", None)

    try:
        invocation = _create_agent_invocation(
            starting_agent, user_input, system_prompt=system_prompt
        )
    except Exception as e:
        logger.debug(f"Failed to create agent invocation: {e}")
        async for step in wrapped(*args, **kwargs):
            yield step
        return

    counter_token = _step_counter.set(0)
    handler.start_invoke_agent(invocation)

    try:
        last_step = None
        async for step in wrapped(*args, **kwargs):
            last_step = step
            yield step

        if last_step:
            invocation.output_messages = _step_to_output_messages(last_step)
        handler.stop_invoke_agent(invocation)
    except GeneratorExit:
        handler.fail_invoke_agent(
            invocation, Error(message="GeneratorExit", type=GeneratorExit)
        )
        raise
    except Exception as e:
        handler.fail_invoke_agent(
            invocation, Error(message=str(e), type=type(e))
        )
        raise
    finally:
        _step_counter.reset(counter_token)


async def wrap_runner_step(
    wrapped, instance, args, kwargs, *, handler: ExtendedTelemetryHandler
):
    """H3: STEP span for Runner._step."""
    step_num = _step_counter.get() + 1
    _step_counter.set(step_num)

    invocation = ReactStepInvocation(round=step_num)
    invocation.attributes["gen_ai.framework"] = "widesearch"

    try:
        handler.start_react_step(invocation)
    except Exception as e:
        logger.debug(f"Failed to start react step: {e}")
        return await wrapped(*args, **kwargs)

    try:
        result = await wrapped(*args, **kwargs)

        from src.agent.memory import ActionStepError, StepStatus

        if isinstance(result, ActionStepError):
            invocation.finish_reason = "error"
            handler.fail_react_step(
                invocation,
                Error(message=result.message, type=type(result)),
            )
        else:
            if result.step_status == StepStatus.FINISHED:
                invocation.finish_reason = "finished"
            elif result.error_marker is not None:
                invocation.finish_reason = "error"
            else:
                invocation.finish_reason = "continue"
            handler.stop_react_step(invocation)

        return result
    except Exception as e:
        invocation.finish_reason = "error"
        handler.fail_react_step(
            invocation, Error(message=str(e), type=type(e))
        )
        raise


async def wrap_invoke_tool_call(
    wrapped, instance, args, kwargs, *, handler: ExtendedTelemetryHandler
):
    """H4: TOOL span for each tool_call inside Runner._invoke_tool_call."""
    agent = args[0] if args else kwargs.get("agent")
    model_response = args[1] if len(args) > 1 else kwargs.get("model_response")

    if not model_response.outputs:
        return await wrapped(*args, **kwargs)

    resp = model_response.outputs[0]
    if not resp.tool_calls:
        return await wrapped(*args, **kwargs)

    from src.agent.schema import ErrorMarker, ToolCallResult

    async def _call_with_span(tool_call):
        try:
            invocation = _create_tool_invocation(tool_call, agent)
        except Exception as e:
            logger.debug(f"Failed to create tool invocation: {e}")
            return await _call_original(tool_call, agent)

        handler.start_execute_tool(invocation)

        tool_name = tool_call.tool_name
        tool = agent.get_tool_by_name(tool_name)
        if tool is None:
            invocation.tool_call_result = f"Tool {tool_name} not found"
            handler.fail_execute_tool(
                invocation,
                Error(
                    message=f"Tool {tool_name} not found",
                    type=ValueError,
                ),
            )
            return ToolCallResult(
                tool_call_id=tool_call.tool_call_id,
                error_marker=ErrorMarker(
                    message=f"Tool {tool_name} not found"
                ),
            )

        arguments = tool_call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        try:
            response = await tool(**arguments)
        except Exception as e:
            invocation.tool_call_result = str(e)
            handler.fail_execute_tool(
                invocation, Error(message=str(e), type=type(e))
            )
            return ToolCallResult(
                tool_call_id=tool_call.tool_call_id,
                error_marker=ErrorMarker(message=str(e)),
            )

        error_marker = (
            ErrorMarker(message=response.error) if response.error else None
        )
        system_error_marker = (
            ErrorMarker(message=response.system_error)
            if response.system_error
            else None
        )

        result_content = response.data
        invocation.tool_call_result = result_content

        if error_marker or system_error_marker:
            msg = (error_marker or system_error_marker)["message"]
            handler.fail_execute_tool(
                invocation, Error(message=msg, type=RuntimeError)
            )
        else:
            handler.stop_execute_tool(invocation)

        return ToolCallResult(
            tool_call_id=tool_call.tool_call_id,
            content=result_content,
            error_marker=error_marker,
            system_error_marker=system_error_marker,
            extra=response.extra if response.extra else {},
        )

    async def _call_original(tool_call, agent):
        """Fallback: execute tool without span."""
        tool_name = tool_call.tool_name
        tool = agent.get_tool_by_name(tool_name)
        if tool is None:
            return ToolCallResult(
                tool_call_id=tool_call.tool_call_id,
                error_marker=ErrorMarker(
                    message=f"Tool {tool_name} not found"
                ),
            )
        arguments = tool_call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        try:
            response = await tool(**arguments)
        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call.tool_call_id,
                error_marker=ErrorMarker(message=str(e)),
            )
        return ToolCallResult(
            tool_call_id=tool_call.tool_call_id,
            content=response.data,
            error_marker=(
                ErrorMarker(message=response.error) if response.error else None
            ),
            system_error_marker=(
                ErrorMarker(message=response.system_error)
                if response.system_error
                else None
            ),
            extra=response.extra if response.extra else {},
        )

    tasks = [_call_with_span(tc) for tc in resp.tool_calls]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def wrap_create_sub_agents_factory(
    wrapped, instance, args, kwargs, *, handler: ExtendedTelemetryHandler
):
    """H5: TASK span wrapping the closure returned by create_sub_agents_wrap."""
    original_closure = wrapped(*args, **kwargs)

    async def closure_with_task_span(sub_agents):
        tracer = handler._tracer
        span_name = "run_task create_sub_agents"

        with tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("gen_ai.span.kind", "TASK")
            span.set_attribute("gen_ai.operation.name", "run_task")
            span.set_attribute("gen_ai.framework", "widesearch")

            try:
                safe_input = json.dumps(
                    [
                        {
                            "index": sa.get("index"),
                            "prompt": sa.get("prompt", "")[:200],
                        }
                        for sa in sub_agents
                    ],
                    ensure_ascii=False,
                )
                span.set_attribute("input.value", safe_input)
            except Exception:
                pass

            try:
                result = await original_closure(sub_agents)

                if result and hasattr(result, "data") and result.data:
                    output_str = (
                        result.data
                        if isinstance(result.data, str)
                        else json.dumps(result.data, ensure_ascii=False)
                    )
                    if len(output_str) > 4096:
                        output_str = output_str[:4096] + "...(truncated)"
                    span.set_attribute("output.value", output_str)

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return closure_with_task_span
