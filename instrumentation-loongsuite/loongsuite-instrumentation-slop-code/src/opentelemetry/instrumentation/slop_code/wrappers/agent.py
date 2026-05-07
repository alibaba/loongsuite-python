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

"""AGENT span wrapper for Agent.run_checkpoint."""

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


class _AgentRunCheckpointWrapper:
    """Wrapper for Agent.run_checkpoint to create AGENT span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        agent_name = type(instance).__name__
        problem_name = safe_get(instance, "problem_name", "unknown")

        span_name = f"agent.{agent_name}"

        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "invoke_agent",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.AGENT.value,
            "gen_ai.agent.name": agent_name,
            "slop_code.problem.name": str(problem_name),
        }

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                # Extract after-call attributes from result
                if result is not None:
                    usage = safe_get(result, "usage")
                    if usage is not None:
                        net_tokens = safe_get(usage, "net_tokens")
                        if net_tokens is not None:
                            set_optional_attr(
                                span,
                                gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
                                safe_get(net_tokens, "input"),
                            )
                            set_optional_attr(
                                span,
                                gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                                safe_get(net_tokens, "output"),
                            )
                        cost = safe_get(usage, "cost")
                        set_optional_attr(span, "slop_code.usage.cost", cost)
                        steps = safe_get(usage, "steps")
                        set_optional_attr(span, "slop_code.usage.steps", steps)

                    elapsed = safe_get(result, "elapsed")
                    set_optional_attr(span, "slop_code.elapsed_seconds", elapsed)

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__name__)
                raise
