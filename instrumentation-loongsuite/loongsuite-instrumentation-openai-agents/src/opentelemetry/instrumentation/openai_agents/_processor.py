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

from __future__ import annotations

import json
import logging
from typing import Any

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

from opentelemetry import context as otel_context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import (
    Span as OTelSpan,
)
from opentelemetry.trace import (
    SpanKind,
    StatusCode,
    set_span_in_context,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
)

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "openai_agents"
_ATTR_HANDOFF_FROM = "gen_ai.openai.agents.handoff.from_agent"
_ATTR_HANDOFF_TO = "gen_ai.openai.agents.handoff.to_agent"
_ATTR_GUARDRAIL_NAME = "gen_ai.openai.agents.guardrail.name"
_ATTR_GUARDRAIL_TRIGGERED = "gen_ai.openai.agents.guardrail.triggered"


def _dont_throw(func):
    """Decorator that catches and logs exceptions to avoid
    crashing the user application."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.debug("Error in %s", func.__name__, exc_info=True)

    return wrapper


def _safe_json(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


class OTelTracingProcessor(TracingProcessor):
    """Bridges openai-agents SDK tracing to OpenTelemetry spans.

    Implements the SDK's TracingProcessor interface and translates
    each SDK span into an OTel span following the GenAI semantic
    conventions.
    """

    def __init__(
        self,
        handler: ExtendedTelemetryHandler,
        capture_content: bool = True,
    ):
        self._handler = handler
        self._capture_content = capture_content
        # SDK span_id -> (OTel span, context token)
        self._span_map: dict[str, tuple[OTelSpan, object | None]] = {}
        # SDK trace_id -> (OTel span, context token)
        self._trace_map: dict[str, tuple[OTelSpan, object | None]] = {}

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------

    @_dont_throw
    def on_trace_start(self, trace: Trace) -> None:
        span_name = f"invoke_workflow {trace.name}"
        otel_span = self._handler._tracer.start_span(
            name=span_name, kind=SpanKind.INTERNAL
        )
        otel_span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "invoke_workflow")
        otel_span.set_attribute(GenAI.GEN_AI_SYSTEM, _PROVIDER_NAME)
        otel_span.set_attribute(GenAI.GEN_AI_PROVIDER_NAME, _PROVIDER_NAME)
        ctx_token = otel_context.attach(set_span_in_context(otel_span))
        self._trace_map[trace.trace_id] = (otel_span, ctx_token)

    @_dont_throw
    def on_trace_end(self, trace: Trace) -> None:
        entry = self._trace_map.pop(trace.trace_id, None)
        if entry is None:
            return
        otel_span, ctx_token = entry
        otel_span.end()
        if ctx_token is not None:
            otel_context.detach(ctx_token)

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    @_dont_throw
    def on_span_start(self, span: Span[Any]) -> None:
        span_data = span.span_data
        parent_ctx = None

        # Resolve parent: SDK parent_id -> OTel parent span context
        if span.parent_id and span.parent_id in self._span_map:
            parent_otel_span, _ = self._span_map[span.parent_id]
            parent_ctx = set_span_in_context(parent_otel_span)
        elif span.trace_id in self._trace_map:
            parent_otel_span, _ = self._trace_map[span.trace_id]
            parent_ctx = set_span_in_context(parent_otel_span)

        otel_span = self._create_span_for(span_data, parent_ctx)
        if otel_span is None:
            return

        ctx_token = otel_context.attach(set_span_in_context(otel_span))
        self._span_map[span.span_id] = (otel_span, ctx_token)

    @_dont_throw
    def on_span_end(self, span: Span[Any]) -> None:
        entry = self._span_map.pop(span.span_id, None)
        if entry is None:
            return
        otel_span, ctx_token = entry

        self._apply_end_attributes(otel_span, span)

        if span.error:
            otel_span.set_status(StatusCode.ERROR, span.error["message"])
            otel_span.set_attribute("error.type", span.error["message"])

        otel_span.end()
        if ctx_token is not None:
            otel_context.detach(ctx_token)

    # ------------------------------------------------------------------
    # Span creation per type
    # ------------------------------------------------------------------

    def _create_span_for(
        self,
        span_data: Any,
        parent_ctx: Any | None,
    ) -> OTelSpan | None:
        if isinstance(span_data, AgentSpanData):
            return self._create_agent_span(span_data, parent_ctx)
        if isinstance(span_data, GenerationSpanData):
            return self._create_generation_span(span_data, parent_ctx)
        if isinstance(span_data, ResponseSpanData):
            return self._create_response_span(span_data, parent_ctx)
        if isinstance(span_data, FunctionSpanData):
            return self._create_function_span(span_data, parent_ctx)
        if isinstance(span_data, HandoffSpanData):
            return self._create_handoff_span(span_data, parent_ctx)
        if isinstance(span_data, GuardrailSpanData):
            return self._create_guardrail_span(span_data, parent_ctx)
        # Fallback for custom/unknown span types
        return self._create_generic_span(span_data, parent_ctx)

    def _create_agent_span(
        self, data: AgentSpanData, parent_ctx: Any | None
    ) -> OTelSpan:
        span_name = (
            f"{GenAI.GenAiOperationNameValues.INVOKE_AGENT.value} {data.name}"
        )
        span = self._handler._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )
        span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME,
            GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
        )
        span.set_attribute(GenAI.GEN_AI_SYSTEM, _PROVIDER_NAME)
        span.set_attribute(GenAI.GEN_AI_AGENT_NAME, data.name)
        return span

    def _create_generation_span(
        self,
        data: GenerationSpanData,
        parent_ctx: Any | None,
    ) -> OTelSpan:
        model_name = data.model or "unknown"
        span_name = f"chat {model_name}"
        span = self._handler._tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            context=parent_ctx,
        )
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "chat")
        span.set_attribute(GenAI.GEN_AI_SYSTEM, "openai")
        if data.model:
            span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, data.model)
        if data.model_config:
            self._set_model_config(span, data.model_config)
        return span

    def _create_response_span(
        self,
        data: ResponseSpanData,
        parent_ctx: Any | None,
    ) -> OTelSpan:
        model_name = "unknown"
        if data.response and hasattr(data.response, "model"):
            model_name = data.response.model or "unknown"
        span_name = f"chat {model_name}"
        span = self._handler._tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            context=parent_ctx,
        )
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "chat")
        span.set_attribute(GenAI.GEN_AI_SYSTEM, "openai")
        if model_name != "unknown":
            span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, model_name)
        return span

    def _create_function_span(
        self,
        data: FunctionSpanData,
        parent_ctx: Any | None,
    ) -> OTelSpan:
        span_name = (
            f"{GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value} {data.name}"
        )
        span = self._handler._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )
        span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME,
            GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
        )
        span.set_attribute(GenAI.GEN_AI_SYSTEM, _PROVIDER_NAME)
        span.set_attribute(GenAI.GEN_AI_TOOL_NAME, data.name)
        span.set_attribute(GenAI.GEN_AI_TOOL_TYPE, "function")
        return span

    def _create_handoff_span(
        self,
        data: HandoffSpanData,
        parent_ctx: Any | None,
    ) -> OTelSpan:
        from_name = data.from_agent or "unknown"
        to_name = data.to_agent or "unknown"
        span_name = f"{from_name} -> {to_name}"
        span = self._handler._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )
        span.set_attribute(GenAI.GEN_AI_SYSTEM, _PROVIDER_NAME)
        span.set_attribute(_ATTR_HANDOFF_FROM, from_name)
        span.set_attribute(_ATTR_HANDOFF_TO, to_name)
        if data.from_agent:
            span.set_attribute(GenAI.GEN_AI_AGENT_NAME, data.from_agent)
        return span

    def _create_guardrail_span(
        self,
        data: GuardrailSpanData,
        parent_ctx: Any | None,
    ) -> OTelSpan:
        span_name = f"guardrail {data.name}"
        span = self._handler._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )
        span.set_attribute(GenAI.GEN_AI_SYSTEM, _PROVIDER_NAME)
        span.set_attribute(_ATTR_GUARDRAIL_NAME, data.name)
        return span

    def _create_generic_span(
        self,
        span_data: Any,
        parent_ctx: Any | None,
    ) -> OTelSpan:
        span_type = getattr(span_data, "type", "unknown")
        span_name = getattr(span_data, "name", span_type)
        span = self._handler._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )
        span.set_attribute(GenAI.GEN_AI_SYSTEM, _PROVIDER_NAME)
        return span

    # ------------------------------------------------------------------
    # End-of-span attribute population
    # ------------------------------------------------------------------

    def _apply_end_attributes(
        self, otel_span: OTelSpan, sdk_span: Span[Any]
    ) -> None:
        span_data = sdk_span.span_data
        if isinstance(span_data, AgentSpanData):
            self._apply_agent_end(otel_span, span_data)
        elif isinstance(span_data, GenerationSpanData):
            self._apply_generation_end(otel_span, span_data)
        elif isinstance(span_data, ResponseSpanData):
            self._apply_response_end(otel_span, span_data)
        elif isinstance(span_data, FunctionSpanData):
            self._apply_function_end(otel_span, span_data)
        elif isinstance(span_data, HandoffSpanData):
            self._apply_handoff_end(otel_span, span_data)
        elif isinstance(span_data, GuardrailSpanData):
            self._apply_guardrail_end(otel_span, span_data)

    def _apply_agent_end(self, span: OTelSpan, data: AgentSpanData) -> None:
        if data.tools:
            span.set_attribute(
                "gen_ai.openai.agents.agent.tools",
                data.tools,
            )
        if data.handoffs:
            span.set_attribute(
                "gen_ai.openai.agents.agent.handoffs",
                data.handoffs,
            )
        if data.output_type:
            span.set_attribute(GenAI.GEN_AI_OUTPUT_TYPE, data.output_type)

    def _apply_generation_end(
        self, span: OTelSpan, data: GenerationSpanData
    ) -> None:
        if data.model:
            span.set_attribute(GenAI.GEN_AI_RESPONSE_MODEL, data.model)
        if data.usage:
            input_tokens = data.usage.get("input_tokens") or data.usage.get(
                "prompt_tokens"
            )
            output_tokens = data.usage.get("output_tokens") or data.usage.get(
                "completion_tokens"
            )
            if input_tokens is not None:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_INPUT_TOKENS,
                    input_tokens,
                )
            if output_tokens is not None:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_OUTPUT_TOKENS,
                    output_tokens,
                )
        if data.model_config:
            self._set_model_config(span, data.model_config)
        if self._capture_content:
            if data.input:
                span.set_attribute(
                    "gen_ai.input.messages",
                    _safe_json(list(data.input)),
                )
            if data.output:
                span.set_attribute(
                    "gen_ai.output.messages",
                    _safe_json(list(data.output)),
                )

    def _apply_response_end(
        self, span: OTelSpan, data: ResponseSpanData
    ) -> None:
        resp = data.response
        if resp is None:
            return
        if hasattr(resp, "model") and resp.model:
            span.set_attribute(GenAI.GEN_AI_RESPONSE_MODEL, resp.model)
        if hasattr(resp, "id") and resp.id:
            span.set_attribute(GenAI.GEN_AI_RESPONSE_ID, resp.id)
        if hasattr(resp, "usage") and resp.usage:
            usage = resp.usage
            if hasattr(usage, "input_tokens"):
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_INPUT_TOKENS,
                    usage.input_tokens,
                )
            if hasattr(usage, "output_tokens"):
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_OUTPUT_TOKENS,
                    usage.output_tokens,
                )
        if self._capture_content and data.input:
            span.set_attribute(
                "gen_ai.input.messages",
                _safe_json(data.input),
            )

    def _apply_function_end(
        self, span: OTelSpan, data: FunctionSpanData
    ) -> None:
        if self._capture_content:
            if data.input is not None:
                span.set_attribute(
                    GenAI.GEN_AI_TOOL_CALL_ARGUMENTS,
                    str(data.input),
                )
            if data.output is not None:
                span.set_attribute(
                    GenAI.GEN_AI_TOOL_CALL_RESULT,
                    str(data.output),
                )
        if data.mcp_data:
            span.set_attribute(
                "gen_ai.openai.agents.mcp.server",
                _safe_json(data.mcp_data),
            )

    def _apply_handoff_end(
        self, span: OTelSpan, data: HandoffSpanData
    ) -> None:
        if data.to_agent:
            span.set_attribute(_ATTR_HANDOFF_TO, data.to_agent)

    def _apply_guardrail_end(
        self, span: OTelSpan, data: GuardrailSpanData
    ) -> None:
        span.set_attribute(_ATTR_GUARDRAIL_TRIGGERED, data.triggered)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_model_config(span: OTelSpan, config: Any) -> None:
        if not isinstance(config, dict):
            return
        _CONFIG_ATTRS = {
            "temperature": GenAI.GEN_AI_REQUEST_TEMPERATURE,
            "max_tokens": GenAI.GEN_AI_REQUEST_MAX_TOKENS,
            "top_p": GenAI.GEN_AI_REQUEST_TOP_P,
            "top_k": GenAI.GEN_AI_REQUEST_TOP_K,
        }
        for key, attr in _CONFIG_ATTRS.items():
            val = config.get(key)
            if val is not None:
                span.set_attribute(attr, val)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @_dont_throw
    def shutdown(self) -> None:
        self._span_map.clear()
        self._trace_map.clear()

    @_dont_throw
    def force_flush(self) -> None:
        pass
