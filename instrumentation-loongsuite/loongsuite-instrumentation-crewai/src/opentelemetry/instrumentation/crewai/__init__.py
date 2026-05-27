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

"""
OpenTelemetry CrewAI Instrumentation (Optimized)

"""

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api

# Import hook system for decoupled extensions
from opentelemetry.instrumentation.crewai.package import _instruments
from opentelemetry.instrumentation.crewai.utils import (
    OP_NAME_AGENT,
    OP_NAME_CREW,
    OP_NAME_TASK,
    OP_NAME_TOOL,
    GenAIHookHelper,
    extract_agent_inputs,
    extract_tool_inputs,
    to_input_message,
    to_output_message,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)

try:
    import crewai.agent
    import crewai.crew
    import crewai.flow.flow
    import crewai.task
    import crewai.tools.tool_usage

    _CREWAI_LOADED = True
except (ImportError, Exception):
    _CREWAI_LOADED = False

logger = logging.getLogger(__name__)


class CrewAIInstrumentor(BaseInstrumentor):
    """
    An instrumentor for CrewAI framework.

    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument CrewAI framework."""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__,
            "",
            tracer_provider=tracer_provider,
        )

        genai_helper = GenAIHookHelper()

        # Wrap Crew.kickoff (CHAIN span)
        try:
            wrap_function_wrapper(
                module="crewai.crew",
                name="Crew.kickoff",
                wrapper=_CrewKickoffWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Crew.kickoff: {e}")

        # Wrap Flow.kickoff_async (CHAIN span)
        try:
            wrap_function_wrapper(
                module="crewai.flow.flow",
                name="Flow.kickoff_async",
                wrapper=_FlowKickoffAsyncWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.debug(f"Could not wrap Flow.kickoff_async: {e}")

        # Wrap Agent.execute_task (AGENT span)
        try:
            wrap_function_wrapper(
                module="crewai.agent",
                name="Agent.execute_task",
                wrapper=_AgentExecuteTaskWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Agent.execute_task: {e}")

        # Wrap Task.execute_sync (TASK span)
        try:
            wrap_function_wrapper(
                module="crewai.task",
                name="Task.execute_sync",
                wrapper=_TaskExecuteSyncWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Task.execute_sync: {e}")

        # Wrap ToolUsage._use (TOOL span)
        try:
            wrap_function_wrapper(
                module="crewai.tools.tool_usage",
                name="ToolUsage._use",
                wrapper=_ToolUseWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.debug(f"Could not wrap ToolUsage._use: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Uninstrument CrewAI framework."""
        if not _CREWAI_LOADED:
            logger.debug(
                "CrewAI modules were not available for uninstrumentation."
            )
            return
        try:
            unwrap(crewai.crew.Crew, "kickoff")
            unwrap(crewai.flow.flow.Flow, "kickoff_async")
            unwrap(crewai.agent.Agent, "execute_task")
            unwrap(crewai.task.Task, "execute_sync")
            unwrap(crewai.tools.tool_usage.ToolUsage, "_use")

        except Exception as e:
            logger.debug(f"Error during uninstrumenting: {e}")


class _CrewKickoffWrapper:
    """
    Wrapper for Crew.kickoff method to create CHAIN span.
    """

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Crew.kickoff to create CHAIN span."""
        inputs = kwargs.get("inputs", {})
        crew_name = getattr(instance, "name", None) or "crew.kickoff"

        genai_inputs = to_input_message("user", inputs)

        with self._tracer.start_as_current_span(
            name=crew_name,
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: OP_NAME_CREW,
                gen_ai_attributes.GEN_AI_SYSTEM: "crewai",
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.AGENT.value,
            },
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                output_val = (
                    result.raw if hasattr(result, "raw") else str(result)
                )
                genai_outputs = to_output_message("assistant", output_val)

                self._helper.on_completion(span, genai_inputs, genai_outputs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                self._helper.on_completion(span, genai_inputs, [])
                raise


class _FlowKickoffAsyncWrapper:
    """Wrapper for Flow.kickoff_async method to create CHAIN span."""

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    async def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Flow.kickoff_async to create CHAIN span."""
        inputs = kwargs.get("inputs", {})
        flow_name = getattr(instance, "name", None) or "flow.kickoff"

        genai_inputs = to_input_message("user", inputs)

        with self._tracer.start_as_current_span(
            name=flow_name,
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: OP_NAME_CREW,
                gen_ai_attributes.GEN_AI_SYSTEM: "crewai",
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.AGENT.value,
            },
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)
                genai_outputs = to_output_message("assistant", result)
                self._helper.on_completion(span, genai_inputs, genai_outputs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                self._helper.on_completion(span, genai_inputs, [])
                raise


class _AgentExecuteTaskWrapper:
    """Wrapper for Agent.execute_task method to create AGENT span."""

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Agent.execute_task to create AGENT span."""
        task = args[0] if args else kwargs.get("task")
        context = kwargs.get("context", "")
        tools = kwargs.get("tools", [])
        agent_role = getattr(instance, "role", "agent")

        span_name = f"Agent.{agent_role}"

        genai_inputs = extract_agent_inputs(task, context, tools)

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: OP_NAME_AGENT,
                gen_ai_attributes.GEN_AI_SYSTEM: "crewai",
                "gen_ai.agent.name": agent_role,
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.AGENT.value,
            },
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                genai_outputs = to_output_message("assistant", result)

                self._helper.on_completion(span, genai_inputs, genai_outputs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                self._helper.on_completion(span, genai_inputs, [])
                raise


class _TaskExecuteSyncWrapper:
    """Wrapper for Task.execute_sync method to create TASK span."""

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Task.execute_sync to create TASK span."""
        task_desc = getattr(instance, "description", "task")
        span_name = f"Task.{task_desc[:50]}"

        genai_inputs = to_input_message("user", task_desc)

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: OP_NAME_TASK,
                gen_ai_attributes.GEN_AI_SYSTEM: "crewai",
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.AGENT.value,
            },
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                genai_outputs = to_output_message("assistant", result)
                self._helper.on_completion(span, genai_inputs, genai_outputs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                self._helper.on_completion(span, genai_inputs, [])
                raise


class _ToolUseWrapper:
    """Wrapper for ToolUsage._use method to create TOOL span."""

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap ToolUsage._use to create TOOL span."""
        tool = args[0] if args else kwargs.get("tool")
        tool_name = (
            getattr(tool, "name", "unknown_tool") if tool else "unknown_tool"
        )

        tool_calling = args[1] if len(args) > 1 else kwargs.get("tool_calling")
        arguments = (
            getattr(tool_calling, "arguments", {}) if tool_calling else {}
        )
        genai_inputs = extract_tool_inputs(tool_name, arguments)

        with self._tracer.start_as_current_span(
            name=f"Tool.{tool_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: OP_NAME_TOOL,
                gen_ai_attributes.GEN_AI_SYSTEM: "crewai",
                gen_ai_attributes.GEN_AI_TOOL_NAME: tool_name,
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.TOOL.value,
            },
        ) as span:
            # Set tool description
            if tool and hasattr(tool, "description"):
                span.set_attribute("gen_ai.tool.description", tool.description)

            try:
                result = wrapped(*args, **kwargs)

                genai_outputs = to_output_message("tool", result)

                self._helper.on_completion(span, genai_inputs, genai_outputs)

                is_error = False
                if instance:
                    _run_attempts = getattr(instance, "_run_attempts", None)
                    _max_parsing_attempts = getattr(
                        instance, "_max_parsing_attempts", None
                    )
                    if (
                        _max_parsing_attempts
                        and _run_attempts
                        and _run_attempts > _max_parsing_attempts
                    ):
                        span.set_status(Status(StatusCode.ERROR))
                        is_error = True

                if not is_error:
                    span.set_status(Status(StatusCode.OK))

                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                self._helper.on_completion(span, genai_inputs, [])
                raise
