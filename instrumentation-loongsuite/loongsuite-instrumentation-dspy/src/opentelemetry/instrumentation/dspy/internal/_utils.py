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

"""Data extraction helpers for DSPy instrumentation."""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry.instrumentation.dspy.internal.config import (
    model_name_enabled,
)
from opentelemetry.util.genai.extended_types import RetrievalDocument
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
    is_experimental_mode,
)

logger = logging.getLogger(__name__)

_MAX_VALUE_LENGTH = 4096


def should_capture_content() -> bool:
    """Whether input/output payloads may be written to span attributes."""
    try:
        if not is_experimental_mode():
            return False
        return get_content_capturing_mode() in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    except ValueError:
        logger.debug("Failed to resolve content capturing mode", exc_info=True)
        return False


def to_plain(obj: Any) -> Any:
    """Convert DSPy containers into JSON-friendly plain Python values."""
    store = getattr(obj, "_store", None)
    if isinstance(store, dict):
        return {
            key: to_plain(value)
            for key, value in store.items()
            if not key.startswith("dspy_")
        }
    if isinstance(obj, dict):
        return {key: to_plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(item) for item in obj]
    return obj


def safe_json(obj: Any, max_len: int = _MAX_VALUE_LENGTH) -> str:
    """Serialize *obj* to a truncated JSON string, never raising."""
    try:
        text = json.dumps(to_plain(obj), ensure_ascii=False, default=str)
    except Exception:
        logger.debug(
            "Failed to serialize DSPy value, falling back to str()",
            exc_info=True,
        )
        text = str(obj)
    if len(text) > max_len:
        return text[:max_len] + "...[truncated]"
    return text


def normalize_callback_inputs(inputs: Any) -> Any:
    """Flatten the ``*args``/``**kwargs`` shape produced by ``with_callbacks``.

    ``inspect.getcallargs`` reports ``Module.__call__(self, *args, **kwargs)``
    as ``{"args": (...), "kwargs": {...}}``, which hides the real signature
    fields from every downstream extractor.
    """
    if not isinstance(inputs, dict):
        return inputs
    if not inputs or not set(inputs).issubset({"args", "kwargs"}):
        return inputs

    args = inputs.get("args") or ()
    kwargs = inputs.get("kwargs") or {}
    if args and kwargs:
        return {"args": list(args), **kwargs}
    if args:
        return list(args)
    return dict(kwargs)


def build_input_messages(inputs: Any) -> list[InputMessage]:
    """Represent module inputs as a single user message."""
    if inputs is None:
        return []
    return [InputMessage(role="user", parts=[Text(content=safe_json(inputs))])]


def build_output_messages(outputs: Any) -> list[OutputMessage]:
    """Represent module outputs as a single assistant message."""
    if outputs is None:
        return []
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=safe_json(outputs))],
            finish_reason="stop",
        )
    ]


def resolve_request_model() -> str | None:
    """Best-effort model name of the LM configured for the current context.

    The value is normalized the same way ``loongsuite-instrumentation-litellm``
    normalizes ``gen_ai.request.model`` (provider prefix stripped), so the
    framework spans and the LLM spans below them share one ``modelName``
    dimension instead of splitting it.
    """
    if not model_name_enabled():
        return None
    try:
        import dspy  # noqa: PLC0415

        model = getattr(dspy.settings.get("lm"), "model", None)
    except Exception:
        logger.debug("Failed to resolve DSPy LM model name", exc_info=True)
        return None

    if not isinstance(model, str) or not model:
        return None
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def extract_tool_definitions(instance: Any) -> list[ToolDefinition]:
    """Convert ``ReAct.tools`` into GenAI tool definitions."""
    tools = getattr(instance, "tools", None)
    if isinstance(tools, dict):
        candidates = list(tools.values())
    elif isinstance(tools, (list, tuple)):
        candidates = list(tools)
    else:
        return []

    definitions: list[ToolDefinition] = []
    for tool in candidates:
        name = getattr(tool, "name", None)
        if not name:
            continue
        definitions.append(
            FunctionToolDefinition(
                name=str(name),
                description=getattr(tool, "desc", None),
                parameters=getattr(tool, "args", None),
            )
        )
    return definitions


def extract_query(inputs: Any) -> str | None:
    """Pull the retrieval query out of normalized ``Retrieve`` inputs."""
    if isinstance(inputs, dict):
        query = inputs.get("query")
    elif isinstance(inputs, (list, tuple)) and inputs:
        query = inputs[0]
    else:
        query = None
    return query if isinstance(query, str) else None


def extract_top_k(inputs: Any, instance: Any) -> float | None:
    """Resolve the effective ``k`` of a ``Retrieve`` call."""
    top_k = inputs.get("k") if isinstance(inputs, dict) else None
    if top_k is None:
        top_k = getattr(instance, "k", None)
    if isinstance(top_k, bool) or not isinstance(top_k, (int, float)):
        return None
    return float(top_k)


def build_retrieval_documents(outputs: Any) -> list[RetrievalDocument]:
    """Convert retriever outputs into retrieval documents.

    DSPy retrievers return passages without ids or scores, so ordinal ids are
    synthesized and ``score`` stays unset rather than being faked.
    """
    passages = getattr(outputs, "passages", None)
    if passages is None and isinstance(outputs, dict):
        passages = outputs.get("passages")
    if passages is None and isinstance(outputs, (list, tuple)):
        passages = outputs
    if not isinstance(passages, (list, tuple)):
        return []

    capture_content = should_capture_content()
    documents: list[RetrievalDocument] = []
    for index, passage in enumerate(passages):
        content = None
        score = None
        if isinstance(passage, dict):
            content = passage.get("long_text") or passage.get("text")
            raw_score = passage.get("score")
            if isinstance(raw_score, (int, float)) and not isinstance(
                raw_score, bool
            ):
                score = float(raw_score)
        elif isinstance(passage, str):
            content = passage
        else:
            content = str(passage)

        documents.append(
            RetrievalDocument(
                id=str(index),
                score=score,
                content=content if capture_content else None,
            )
        )
    return documents


def aggregate_lm_usage(outputs: Any) -> dict[str, int]:
    """Sum ``Prediction.get_lm_usage()`` across every LM that was called.

    Returns an empty dict unless ``dspy.settings.track_usage`` was enabled.
    """
    getter = getattr(outputs, "get_lm_usage", None)
    if not callable(getter):
        return {}
    try:
        usage_by_lm = getter()
    except Exception:
        logger.debug("Failed to read DSPy LM usage", exc_info=True)
        return {}
    if not isinstance(usage_by_lm, dict):
        return {}

    totals: dict[str, int] = {}
    for usage in usage_by_lm.values():
        if not isinstance(usage, dict):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            totals[key] = totals.get(key, 0) + value
    return totals
