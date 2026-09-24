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

"""Cache usage extraction shared by AgentScope v1 and v2 adapters."""

from typing import Any

from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
from opentelemetry.util.genai.types import LLMInvocation


def _safe_get(obj: Any, key: str) -> Any:
    """Support ChatUsage dicts whose missing attributes raise KeyError."""
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return getattr(obj, key, None)
    except (KeyError, AttributeError):
        return None


@hook_advice("agentscope", "extract_cache_tokens")
def _extract_cache_tokens(
    usage: Any, invocation: LLMInvocation | InvokeAgentInvocation
) -> None:
    """Read AgentScope, Anthropic and OpenAI/DashScope cache usage fields.

    Explicit top-level values, including zero, take precedence. Fall back
    independently for each counter; absent usage must not imply a cache miss.
    """
    cache_creation = _safe_get(usage, "cache_creation_input_tokens")
    cache_read = _safe_get(usage, "cache_read_input_tokens")
    if cache_read is None:
        cache_read = _safe_get(usage, "cache_input_tokens")

    if cache_creation is None or cache_read is None:
        details = _safe_get(usage, "prompt_tokens_details")
        if cache_creation is None:
            cache_creation = _safe_get(details, "cache_creation_input_tokens")
        if cache_read is None:
            cache_read = _safe_get(details, "cached_tokens")

    if cache_creation is not None:
        invocation.usage_cache_creation_input_tokens = cache_creation
    if cache_read is not None:
        invocation.usage_cache_read_input_tokens = cache_read
