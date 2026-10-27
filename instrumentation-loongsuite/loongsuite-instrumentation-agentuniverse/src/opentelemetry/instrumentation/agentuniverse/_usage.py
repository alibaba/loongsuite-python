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

"""Read provider totals without changing the framework's business objects."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from agentuniverse.llm.llm_output import TokenUsage

_current_call: ContextVar[Any] = ContextVar(
    "agentuniverse_llm_call", default=None
)


@contextmanager
def usage_call_context(call):
    """Only held while calling the framework or advancing its iterator."""
    token = _current_call.set(call)
    try:
        yield
    finally:
        _current_call.reset(token)


def _count(value: Any) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )


def provider_usage(
    payload: Any, provider: str = "openai"
) -> Optional[TokenUsage]:
    if not isinstance(payload, dict):
        return None
    if provider == "anthropic":
        # Anthropic input_tokens excludes cache reads and writes.
        values = [
            payload.get(key, 0)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        ]
        if not all(_count(value) for value in values):
            return None
        if "input_tokens" not in payload or "output_tokens" not in payload:
            return None
        inputs, outputs, reads, writes = values
        return TokenUsage(
            text_in=inputs + writes, cached_in=reads, text_out=outputs
        )
    inputs = payload.get("prompt_tokens", payload.get("input_tokens"))
    outputs = payload.get("completion_tokens", payload.get("output_tokens"))
    if not _count(inputs) or not _count(outputs):
        return None
    details_in = (
        payload.get("prompt_tokens_details")
        or payload.get("input_tokens_details")
        or {}
    )
    details_out = (
        payload.get("completion_tokens_details")
        or payload.get("output_tokens_details")
        or {}
    )
    if not isinstance(details_in, dict) or not isinstance(details_out, dict):
        return None
    fields = {
        "image_in": details_in.get("image_tokens", 0),
        "audio_in": details_in.get("audio_tokens", 0),
        "cached_in": details_in.get("cached_tokens", 0),
        "image_out": details_out.get("image_tokens", 0),
        "audio_out": details_out.get("audio_tokens", 0),
        "cached_out": details_out.get("cached_tokens", 0),
        "reasoning_out": details_out.get("reasoning_tokens", 0),
    }
    fields = {
        key: 0 if value is None else value for key, value in fields.items()
    }
    if not all(_count(value) for value in fields.values()):
        return None
    # These details are subsets of provider totals. TokenUsage's properties
    # sum disjoint buckets, so only the remainder belongs in its text buckets.
    fields["text_in"] = inputs - sum(
        fields[key] for key in ("image_in", "audio_in", "cached_in")
    )
    fields["text_out"] = outputs - sum(
        fields[key]
        for key in ("image_out", "audio_out", "cached_out", "reasoning_out")
    )
    if fields["text_in"] < 0 or fields["text_out"] < 0:
        return None
    return TokenUsage(**fields)


def capture_provider_usage(payload: Any, provider: str) -> None:
    call = _current_call.get()
    if call is not None:
        usage = provider_usage(payload, provider)
        if usage is not None:
            # Usage events are cumulative snapshots, including usage-only tails.
            call.provider_usage = usage
