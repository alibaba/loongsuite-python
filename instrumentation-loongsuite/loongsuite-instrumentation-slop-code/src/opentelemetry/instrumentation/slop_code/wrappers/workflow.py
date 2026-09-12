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

"""CHAIN/workflow span wrapper for run_agent_on_problem."""

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
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)

logger = logging.getLogger(__name__)


class _WorkflowWrapper:
    """Wrapper for run_agent_on_problem to create workflow (CHAIN) span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        # run_agent_on_problem(problem_config, problem_name, config, progress_queue, output_path)
        problem_name = (
            args[1] if len(args) > 1 else kwargs.get("problem_name", "unknown")
        )
        config = args[2] if len(args) > 2 else kwargs.get("config")

        span_name = f"chain {problem_name}"

        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "workflow",
            gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: "CHAIN",
            "gen_ai.framework": SYSTEM_NAME,
            "input.value": str(problem_name),
            "slop_code.problem.name": str(problem_name),
        }

        # Extract optional attributes from config
        if config is not None:
            model_name = safe_get_nested(config, "model_def", "name")
            set_optional_attr_dict(
                attrs, gen_ai_attributes.GEN_AI_REQUEST_MODEL, model_name
            )

            agent_type = safe_get_nested(config, "agent_config", "type")
            set_optional_attr_dict(attrs, "slop_code.agent.type", agent_type)

            pass_policy = safe_get_nested(config, "pass_policy", "value")
            if pass_policy is None:
                pass_policy_obj = safe_get(config, "pass_policy")
                if pass_policy_obj is not None and hasattr(
                    pass_policy_obj, "value"
                ):
                    pass_policy = pass_policy_obj.value
            set_optional_attr_dict(attrs, "slop_code.pass_policy", pass_policy)

        try:
            with self._tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
                attributes={k: v for k, v in attrs.items() if v is not None},
            ) as span:
                try:
                    result = wrapped(*args, **kwargs)

                    if isinstance(result, dict):
                        summary = result.get("summary")
                        if isinstance(summary, dict):
                            set_optional_attr(
                                span, "slop_code.state", summary.get("state")
                            )
                            set_optional_attr(
                                span,
                                "slop_code.passed_policy",
                                summary.get("passed_policy"),
                            )
                            set_optional_attr(
                                span, "output.value", str(summary)
                            )

                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        finally:
            # Flush AFTER the `with` block so the workflow span itself
            # is `on_end`-delivered to the SpanProcessor before we ask it
            # to drain. run_agent_on_problem is the last meaningful work
            # item inside the per-problem worker subprocess; once it
            # returns, the process is reaped by ProcessPoolExecutor's
            # shutdown which can short-circuit BatchSpanProcessor's
            # atexit handler. Without this explicit flush the CHAIN span
            # (and the tail batch of TASK/AGENT/STEP spans) gets dropped.
            try:
                provider = trace_api.get_tracer_provider()
                flush = getattr(provider, "force_flush", None)
                if callable(flush):
                    flush(timeout_millis=5000)
            except Exception as flush_err:  # noqa: BLE001
                logger.debug(
                    "force_flush after workflow span failed: %s", flush_err
                )


def set_optional_attr_dict(attrs: dict, key: str, value) -> None:
    """Add to attrs dict only if value is not None."""
    if value is not None:
        if isinstance(value, str) and len(value) > 1024:
            value = value[:1024]
        attrs[key] = value
