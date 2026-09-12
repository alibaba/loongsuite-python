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

"""LLM span wrapper for grade_file_async (Rubric Judge)."""

import logging

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.slop_code.utils import (
    SYSTEM_NAME,
    genai_messages,
    json_dumps_attr,
    set_optional_attr,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)

logger = logging.getLogger(__name__)


class _RubricGradeWrapper:
    """Wrapper for grade_file_async to create LLM span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        # grade_file_async(prompt_prefix, criteria_text, file_name, model, provider, temperature, ...)
        model = kwargs.get("model") or (
            args[3] if len(args) > 3 else "unknown"
        )
        provider = kwargs.get("provider") or (
            args[4] if len(args) > 4 else None
        )
        temperature = kwargs.get("temperature") or (
            args[5] if len(args) > 5 else None
        )

        # Determine system name from provider
        system_name = SYSTEM_NAME
        if provider is not None:
            provider_val = (
                provider.value if hasattr(provider, "value") else str(provider)
            )
            system_name = provider_val.lower()

        span_name = f"chat {model}"

        attrs = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "chat",
            gen_ai_attributes.GEN_AI_SYSTEM: system_name,
            gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.LLM.value,
            gen_ai_attributes.GEN_AI_REQUEST_MODEL: str(model),
            "gen_ai.provider.name": system_name,
            "gen_ai.framework": SYSTEM_NAME,
        }

        prompt_prefix = (
            args[0] if len(args) > 0 else kwargs.get("prompt_prefix")
        )
        criteria_text = (
            args[1] if len(args) > 1 else kwargs.get("criteria_text")
        )
        if prompt_prefix is not None or criteria_text is not None:
            attrs["gen_ai.input.messages"] = genai_messages(
                [
                    {
                        "role": "user",
                        "content": str(prompt_prefix or "")
                        + "\n\n"
                        + str(criteria_text or ""),
                    }
                ]
            )

        if temperature is not None:
            attrs[gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE] = float(
                temperature
            )

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=attrs,
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)

                # result is tuple[list[dict], dict[str, Any]]
                if isinstance(result, tuple) and len(result) >= 2:
                    response_data = result[1]
                    if isinstance(response_data, dict):
                        _set_usage_from_response(span, response_data)
                        response_id = response_data.get("id")
                        set_optional_attr(
                            span, "gen_ai.response.id", response_id
                        )
                        if response_data.get("choices") is not None:
                            span.set_attribute(
                                "gen_ai.output.messages",
                                json_dumps_attr(response_data.get("choices")),
                            )

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


def _set_usage_from_response(span, response_data: dict) -> None:
    """Extract and set token usage attributes from response_data."""
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return

    # OpenRouter format: prompt_tokens / completion_tokens
    # Bedrock format (normalized): input_tokens / output_tokens
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens") or usage.get(
        "output_tokens"
    )

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

    # Cache tokens (OpenRouter specific)
    cache_read = usage.get("cache_read_input_tokens")
    set_optional_attr(
        span,
        gen_ai_extended_attributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        cache_read,
    )

    cache_creation = usage.get("cache_creation_input_tokens")
    set_optional_attr(
        span,
        gen_ai_extended_attributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        cache_creation,
    )
