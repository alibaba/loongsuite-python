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

"""TASK span wrapper for AgentRunner._run_checkpoint."""

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


class _TaskRunCheckpointWrapper:
    """Wrapper for AgentRunner._run_checkpoint to create TASK span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        # _run_checkpoint(self, checkpoint, checkpoint_save_dir, is_first_checkpoint)
        checkpoint = args[0] if args else kwargs.get("checkpoint")
        is_first_checkpoint = args[2] if len(args) > 2 else kwargs.get("is_first_checkpoint", False)

        checkpoint_name = safe_get(checkpoint, "name", "unknown")
        checkpoint_order = safe_get(checkpoint, "order")

        span_name = f"task.{checkpoint_name}"

        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "run_task",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: "TASK",
            "slop_code.checkpoint.name": str(checkpoint_name),
        }

        if checkpoint_order is not None:
            attrs["slop_code.checkpoint.order"] = checkpoint_order
        attrs["slop_code.is_first_checkpoint"] = bool(is_first_checkpoint)

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                # Extract after-call attributes from summary
                if result is not None:
                    had_error = safe_get(result, "had_error")
                    set_optional_attr(span, "slop_code.had_error", had_error)

                    passed_policy = safe_get(result, "passed_policy")
                    set_optional_attr(span, "slop_code.passed_policy", passed_policy)

                # Token usage from agent
                agent = safe_get(instance, "agent")
                if agent is not None:
                    net_tokens = safe_get_nested(agent, "usage", "net_tokens")
                    if net_tokens is not None:
                        input_tokens = safe_get(net_tokens, "input")
                        output_tokens = safe_get(net_tokens, "output")
                        set_optional_attr(span, gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                        set_optional_attr(span, gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
