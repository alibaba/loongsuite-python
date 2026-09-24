# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""STEP span wrappers for MiniSWEAgent ReAct iterations."""

import logging

from opentelemetry import context as context_api
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

_STEP_SPAN_ATTR = "_otel_slop_step_span"
_STEP_TOKEN_ATTR = "_otel_slop_step_token"
_AGG_TOKENS_ATTR = "_otel_slop_aggregate_tokens"


def _estimate_tokens(text) -> int:
    if text is None:
        return 0
    text = str(text)
    return max(1, (len(text) + 3) // 4) if text else 0


def _add_agent_tokens(instance, input_tokens: int, output_tokens: int) -> None:
    current = getattr(instance, _AGG_TOKENS_ATTR, {"input": 0, "output": 0})
    current["input"] = int(current.get("input", 0)) + int(input_tokens or 0)
    current["output"] = int(current.get("output", 0)) + int(output_tokens or 0)
    setattr(instance, _AGG_TOKENS_ATTR, current)


class _MiniSWEStepWrapper:
    """Start a STEP span before the model call and keep it open for tool execution."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        usage = safe_get(instance, "usage")
        current_steps = safe_get(usage, "steps", 0) if usage else 0
        step_num = current_steps + 1

        messages = safe_get(instance, "_messages", [])
        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "react",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.STEP.value,
            gen_ai_extended_attributes.GEN_AI_REACT_ROUND: step_num,
            "gen_ai.framework": SYSTEM_NAME,
        }
        if messages:
            attrs["gen_ai.input.messages"] = genai_messages(messages)

        span = self._tracer.start_span(
            "react step", kind=SpanKind.INTERNAL, attributes=attrs
        )
        token = context_api.attach(trace_api.set_span_in_context(span))
        setattr(instance, _STEP_SPAN_ATTR, span)
        setattr(instance, _STEP_TOKEN_ATTR, token)

        try:
            result = wrapped(*args, **kwargs)
            _record_step_result(instance, span, result, messages)
            if result is None:
                _finish_step(instance, Status(StatusCode.OK), "stop")
            return result
        except Exception as exc:
            span.record_exception(exc)
            _finish_step(instance, Status(StatusCode.ERROR, str(exc)), "error")
            raise


class _MiniSWEObservationWrapper:
    """Finish the current STEP span after the environment/tool observation."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        try:
            return wrapped(*args, **kwargs)
        except Exception as exc:
            span = getattr(instance, _STEP_SPAN_ATTR, None)
            if span is not None:
                span.record_exception(exc)
            _finish_step(instance, Status(StatusCode.ERROR, str(exc)), "error")
            raise
        finally:
            if getattr(instance, _STEP_SPAN_ATTR, None) is not None:
                _finish_step(instance, Status(StatusCode.OK), "stop")


def _record_step_result(instance, span, result, messages) -> None:
    if not isinstance(result, dict):
        return
    token_usage = result.get("token_usage")
    input_tokens = (
        safe_get(token_usage, "input") if token_usage is not None else None
    )
    output_tokens = (
        safe_get(token_usage, "output") if token_usage is not None else None
    )
    content = result.get("content")
    if not input_tokens:
        input_tokens = _estimate_tokens(genai_messages(messages))
    if not output_tokens:
        output_tokens = _estimate_tokens(content)
    set_optional_attr(
        span, gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens
    )
    set_optional_attr(
        span, gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens
    )
    if input_tokens is not None and output_tokens is not None:
        set_optional_attr(
            span, "gen_ai.usage.total_tokens", input_tokens + output_tokens
        )
        _add_agent_tokens(instance, input_tokens, output_tokens)
    if token_usage is not None:
        set_optional_attr(
            span,
            gen_ai_extended_attributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
            safe_get(token_usage, "cache_read"),
        )
        set_optional_attr(
            span,
            gen_ai_extended_attributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
            safe_get(token_usage, "cache_write"),
        )
    set_optional_attr(span, "slop_code.step.cost", result.get("step_cost"))
    if content is not None:
        set_optional_attr(
            span,
            "gen_ai.output.messages",
            genai_messages([{"role": "assistant", "content": content}]),
        )


def _finish_step(instance, status: Status, finish_reason: str) -> None:
    span = getattr(instance, _STEP_SPAN_ATTR, None)
    token = getattr(instance, _STEP_TOKEN_ATTR, None)
    if span is None:
        return
    try:
        span.set_attribute(
            gen_ai_extended_attributes.GEN_AI_REACT_FINISH_REASON,
            finish_reason,
        )
        span.set_status(status)
        span.end()
    finally:
        if token is not None:
            context_api.detach(token)
        for attr in (_STEP_SPAN_ATTR, _STEP_TOKEN_ATTR):
            try:
                delattr(instance, attr)
            except AttributeError:
                pass
