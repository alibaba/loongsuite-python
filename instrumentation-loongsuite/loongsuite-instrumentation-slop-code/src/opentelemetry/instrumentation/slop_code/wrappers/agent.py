# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""AGENT span wrapper for Agent.run_checkpoint."""

import logging

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.slop_code.utils import (
    SYSTEM_NAME,
    genai_messages,
    safe_get,
    set_optional_attr,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)

logger = logging.getLogger(__name__)


def _assistant_messages(instance):
    messages = []
    for step in safe_get(instance, "_steps", []) or []:
        role = safe_get(step, "role")
        role_value = safe_get(role, "value", role)
        if str(role_value).lower().endswith("assistant"):
            content = safe_get(step, "content")
            if content:
                messages.append({"role": "assistant", "content": content})
    if not messages:
        for msg in safe_get(instance, "_messages", []) or []:
            role = safe_get(msg, "role") or (
                msg.get("role") if isinstance(msg, dict) else None
            )
            if role == "assistant":
                content = safe_get(msg, "content") or (
                    msg.get("content") if isinstance(msg, dict) else None
                )
                if content:
                    messages.append({"role": "assistant", "content": content})
    return messages[-3:]


def _extract_system_prompt(instance):
    """Extract the system prompt from the agent instance."""
    system_prompt = safe_get(instance, "system_template")
    if system_prompt:
        return str(system_prompt)

    system_prompt = safe_get(instance, "system_prompt")
    if system_prompt:
        return str(system_prompt)

    for msg in safe_get(instance, "_messages", []) or []:
        role = safe_get(msg, "role") or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        if role == "system":
            content = safe_get(msg, "content") or (
                msg.get("content") if isinstance(msg, dict) else None
            )
            if content:
                return str(content)

    for step in safe_get(instance, "_steps", []) or []:
        role = safe_get(step, "role")
        role_value = safe_get(role, "value", role)
        if str(role_value).lower().endswith("system"):
            content = safe_get(step, "content")
            if content:
                return str(content)

    return None


class _AgentRunCheckpointWrapper:
    """Wrapper for Agent.run_checkpoint to create AGENT span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        task_input = args[0] if args else kwargs.get("task")
        agent_name = type(instance).__name__
        problem_name = safe_get(instance, "problem_name", "unknown")
        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "invoke_agent",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.AGENT.value,
            "gen_ai.framework": SYSTEM_NAME,
            "gen_ai.agent.name": agent_name,
            "gen_ai.agent.id": agent_name,
            "gen_ai.agent.description": "slop-code benchmark agent",
            "slop_code.problem.name": str(problem_name),
        }
        if task_input is not None:
            attrs["gen_ai.input.messages"] = genai_messages(
                [{"role": "user", "content": str(task_input)}]
            )

        system_prompt = _extract_system_prompt(instance)
        if system_prompt is not None:
            attrs["gen_ai.system.instructions"] = genai_messages(
                [{"role": "system", "content": system_prompt}]
            )

        with self._tracer.start_as_current_span(
            name=f"invoke_agent {agent_name}",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                agg = (
                    getattr(instance, "_otel_slop_aggregate_tokens", {}) or {}
                )
                input_tokens = int(agg.get("input", 0) or 0)
                output_tokens = int(agg.get("output", 0) or 0)

                usage = (
                    safe_get(result, "usage") if result is not None else None
                )
                net_tokens = (
                    safe_get(usage, "net_tokens")
                    if usage is not None
                    else None
                )
                if not input_tokens and net_tokens is not None:
                    input_tokens = int(safe_get(net_tokens, "input", 0) or 0)
                if not output_tokens and net_tokens is not None:
                    output_tokens = int(safe_get(net_tokens, "output", 0) or 0)

                if input_tokens:
                    set_optional_attr(
                        span,
                        gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
                        input_tokens,
                    )
                if output_tokens:
                    set_optional_attr(
                        span,
                        gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                        output_tokens,
                    )
                if input_tokens or output_tokens:
                    set_optional_attr(
                        span,
                        "gen_ai.usage.total_tokens",
                        input_tokens + output_tokens,
                    )

                messages = _assistant_messages(instance)
                if messages:
                    set_optional_attr(
                        span,
                        "gen_ai.output.messages",
                        genai_messages(messages),
                    )

                if usage is not None:
                    set_optional_attr(
                        span, "slop_code.usage.cost", safe_get(usage, "cost")
                    )
                    set_optional_attr(
                        span, "slop_code.usage.steps", safe_get(usage, "steps")
                    )
                set_optional_attr(
                    span,
                    "slop_code.elapsed_seconds",
                    safe_get(result, "elapsed")
                    if result is not None
                    else None,
                )
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute("error.type", type(exc).__name__)
                raise
