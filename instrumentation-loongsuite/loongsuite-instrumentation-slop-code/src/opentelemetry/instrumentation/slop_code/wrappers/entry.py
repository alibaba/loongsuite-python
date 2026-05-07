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

"""ENTRY span wrapper for slop_code.entrypoints.commands.run_agent.run_agent."""

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


class _EntryWrapper:
    """Wrapper for run_agent to create ENTRY span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        span_name = "slop-code.enter"

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: "enter",
                gen_ai_attributes.GEN_AI_SYSTEM: SYSTEM_NAME,
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.ENTRY.value,
            },
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
