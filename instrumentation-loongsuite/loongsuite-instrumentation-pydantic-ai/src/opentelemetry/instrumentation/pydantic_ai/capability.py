# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
    GenAiSpanKindValues,
)
from pydantic_ai.capabilities.abstract import AbstractCapability

from opentelemetry.instrumentation.pydantic_ai.span_processor import (
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_OPERATION_NAME,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    normalize_genai_attributes,
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import (
        AgentNode,
        NodeResult,
        ValidatedToolArgs,
        WrapModelRequestHandler,
        WrapNodeRunHandler,
        WrapRunHandler,
        WrapToolExecuteHandler,
    )
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


class LoongSuiteInstrumentationCapability(AbstractCapability[Any]):
    """Pydantic AI Capability that adds LoongSuite GenAI span semantics."""

    def get_ordering(self):
        from pydantic_ai.capabilities.abstract import CapabilityOrdering
        from pydantic_ai.capabilities.instrumentation import Instrumentation

        return CapabilityOrdering(wrapped_by=(Instrumentation,))

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        span = trace.get_current_span()
        agent_name = _agent_name(ctx)
        run_id = getattr(ctx, "run_id", None)
        _set_attributes(
            span,
            {
                GEN_AI_FRAMEWORK: FRAMEWORK_NAME,
                GEN_AI_SPAN_KIND: GenAiSpanKindValues.AGENT.value,
                "gen_ai.agent.id": _agent_id(agent_name, run_id),
            },
        )
        result = await handler()
        _set_total_tokens_from_usage(span, getattr(ctx, "usage", None))
        return result

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: Any,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        span = trace.get_current_span()
        _set_attributes(
            span,
            {
                GEN_AI_FRAMEWORK: FRAMEWORK_NAME,
                GEN_AI_SPAN_KIND: GenAiSpanKindValues.LLM.value,
            },
        )
        response = await handler(request_context)
        _set_total_tokens_from_usage(span, getattr(response, "usage", None))
        return response

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        span = trace.get_current_span()
        _set_attributes(
            span,
            {
                GEN_AI_FRAMEWORK: FRAMEWORK_NAME,
                GEN_AI_SPAN_KIND: GenAiSpanKindValues.TOOL.value,
                "gen_ai.tool.description": getattr(
                    tool_def,
                    "description",
                    None,
                ),
                "gen_ai.tool.type": "function",
            },
        )
        return await handler(args)

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        handler: WrapNodeRunHandler[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        if not _is_model_request_node(node):
            return await handler(node)

        tracer = trace.get_tracer(__name__)
        attributes = {
            GEN_AI_FRAMEWORK: FRAMEWORK_NAME,
            GEN_AI_OPERATION_NAME: "react",
            GEN_AI_SPAN_KIND: GenAiSpanKindValues.STEP.value,
            GEN_AI_REACT_ROUND: _react_round(ctx),
        }
        with tracer.start_as_current_span(
            "react step",
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await handler(node)
            except Exception as exc:
                span.record_exception(exc, escaped=True)
                span.set_status(StatusCode.ERROR)
                span.set_attribute(GEN_AI_REACT_FINISH_REASON, "error")
                raise
            span.set_attribute(
                GEN_AI_REACT_FINISH_REASON,
                _finish_reason(result),
            )
            return result


def _set_total_tokens_from_usage(span: trace.Span, usage: Any) -> None:
    if usage is None or not span.is_recording():
        return
    attrs = {}
    if hasattr(usage, "opentelemetry_attributes"):
        attrs = usage.opentelemetry_attributes()
    input_tokens = _to_int(attrs.get(GEN_AI_USAGE_INPUT_TOKENS))
    output_tokens = _to_int(attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS))
    if input_tokens is not None and output_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_TOTAL_TOKENS, input_tokens + output_tokens)


def _set_attributes(span: trace.Span, attributes: dict[str, Any]) -> None:
    if not span.is_recording():
        return
    normalized = normalize_genai_attributes(
        {key: value for key, value in attributes.items() if value is not None}
    )
    for key, value in normalized.items():
        span.set_attribute(key, value)


def _agent_name(ctx: Any) -> str:
    agent = getattr(ctx, "agent", None)
    return (getattr(agent, "name", None) if agent is not None else None) or "agent"


def _agent_id(agent_name: str, run_id: str | None) -> str:
    raw = f"{agent_name}:{run_id or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _react_round(ctx: Any) -> int:
    value = _to_int(getattr(ctx, "run_step", None))
    if value is None:
        return 1
    return max(1, value)


def _is_model_request_node(node: Any) -> bool:
    return type(node).__name__ == "ModelRequestNode"


def _finish_reason(result: Any) -> str:
    name = type(result).__name__
    if name == "End":
        return "stop"
    return name or "unknown"


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
