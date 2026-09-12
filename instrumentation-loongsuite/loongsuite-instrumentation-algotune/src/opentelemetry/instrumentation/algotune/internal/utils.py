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

"""Shared helpers for AlgoTune wrappers."""

from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.algotune.config import (
    ALGOTUNE_OTEL_MAX_CONTENT_LENGTH,
)

# Aliyun ARMS GenAI conventions (mirrors the values used by the other Robin
# instrumentations such as minisweagent / pinchbench).
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

ALGOTUNE_FRAMEWORK_VALUE = "AlgoTune"

# Instance attribute names used by wrappers to share state across hooks
# without polluting AlgoTune's public API.
INST_STEP_SPAN_ATTR = "_otel_algo_step_span"
INST_STEP_TOKEN_ATTR = "_otel_algo_step_token"
INST_ROUND_ATTR = "_otel_algo_round"
INST_LITELLM_ATTEMPTS_ATTR = "_otel_algo_litellm_attempts"


def truncate(
    text: Any, max_len: int = ALGOTUNE_OTEL_MAX_CONTENT_LENGTH
) -> str:
    """Coerce ``text`` to ``str`` and truncate it to ``max_len`` characters."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:  # noqa: BLE001
            return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def provider_from_model(model_name: str) -> str:
    """Best-effort provider inference from a LiteLLM-style model name.

    AlgoTune uses LiteLLM-style model identifiers (e.g.
    ``openai/gpt-4o``, ``anthropic/claude-3-5-sonnet``). When no
    explicit prefix is present we fall back to substring heuristics.
    """
    if not model_name:
        return "unknown"
    name = model_name.lower()
    if "/" in name:
        prefix = name.split("/", 1)[0]
        # LiteLLM accepts a handful of provider prefixes; map common ones.
        if prefix in {
            "openai",
            "anthropic",
            "vertex_ai",
            "gemini",
            "google",
            "mistral",
            "azure",
            "azure_ai",
            "bedrock",
            "groq",
            "deepseek",
            "openrouter",
            "together_ai",
        }:
            if prefix == "vertex_ai" or prefix == "gemini":
                return "google"
            if prefix == "azure_ai":
                return "azure"
            return prefix
    if "claude" in name or "anthropic" in name:
        return "anthropic"
    if "gemini" in name or "vertex" in name or "google" in name:
        return "google"
    if "mistral" in name:
        return "mistral"
    if "deepseek" in name:
        return "deepseek"
    if "qwen" in name or "dashscope" in name:
        return "dashscope"
    if "gpt" in name or "openai" in name or "o1" in name or "o3" in name:
        return "openai"
    return "unknown"


def safe_close_step(instance: Any) -> None:
    """End any STEP span dangling on ``instance`` and detach its context.

    Used as a safety net in ``run_task``'s ``finally`` block so that a STEP
    span never outlives the AGENT span (e.g. when ``get_response`` returns
    None and the loop ``break``s before ``handle_function_call`` runs, or
    when an exception propagates past STEP cleanup).
    """
    from opentelemetry import context as otel_context  # local import

    span = getattr(instance, INST_STEP_SPAN_ATTR, None)
    token = getattr(instance, INST_STEP_TOKEN_ATTR, None)
    try:
        if span is not None and span.is_recording():
            span.end()
    except Exception:  # noqa: BLE001
        pass
    try:
        if token is not None:
            otel_context.detach(token)
    except Exception:  # noqa: BLE001
        pass
    try:
        setattr(instance, INST_STEP_SPAN_ATTR, None)
        setattr(instance, INST_STEP_TOKEN_ATTR, None)
    except Exception:  # noqa: BLE001
        pass
