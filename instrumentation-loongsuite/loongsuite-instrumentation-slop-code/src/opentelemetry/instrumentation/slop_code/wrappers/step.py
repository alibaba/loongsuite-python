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

"""STEP span wrapper for MiniSWEAgent.agent_step."""

import logging

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.slop_code.utils import (
    SYSTEM_NAME,
    safe_get,
    safe_get_nested,
    set_optional_attr,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import gen_ai_extended_attributes

logger = logging.getLogger(__name__)


class _MiniSWEStepWrapper:
    """Wrapper for MiniSWEAgent.agent_step to create STEP span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        # Determine current step number (1-based)
        usage = safe_get(instance, "usage")
        current_steps = safe_get(usage, "steps", 0) if usage else 0
        step_num = current_steps + 1

        span_name = f"react.step.{step_num}"

        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "react",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.STEP.value,
            gen_ai_extended_attributes.GEN_AI_REACT_ROUND: step_num,
        }

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                # Extract token usage from result if available
                if isinstance(result, dict):
                    token_usage = result.get("token_usage")
                    if token_usage is not None:
                        set_optional_attr(
                            span,
                            gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
                            safe_get(token_usage, "input"),
                        )
                        set_optional_attr(
                            span,
                            gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                            safe_get(token_usage, "output"),
                        )
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
                    step_cost = result.get("step_cost")
                    set_optional_attr(span, "slop_code.step.cost", step_cost)
                elif result is not None:
                    # Result might be a tuple or object; try attribute access
                    token_usage = safe_get(result, "token_usage")
                    if token_usage is not None:
                        set_optional_attr(
                            span,
                            gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
                            safe_get(token_usage, "input"),
                        )
                        set_optional_attr(
                            span,
                            gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                            safe_get(token_usage, "output"),
                        )

                span.set_status(Status(StatusCode.OK))
                span.set_attribute(gen_ai_extended_attributes.GEN_AI_REACT_FINISH_REASON, "stop")
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute(gen_ai_extended_attributes.GEN_AI_REACT_FINISH_REASON, "error")
                raise
