"""SpanProcessor that normalizes Cognee-native spans to gen-ai semantic conventions.

Cognee's own ``new_span`` calls already create spans with names like
``cognee.api.cognify``, ``cognee.pipeline.task.extract_graph``,
``cognee.retrieval.vector_search`` and ``cognee.llm.completion``. We rewrite
their name and inject ``gen_ai.span.kind`` / ``gen_ai.operation.name`` in
``on_start`` (the only point at which the SDK Span is still mutable).

Cognee-specific attributes (``cognee.search.query``, ``cognee.pipeline.task_name`` …)
are set by Cognee's own code during the span body, after ``on_start`` runs.
Because the SDK Span becomes read-only once ``end()`` is called, we cannot
migrate them in ``on_end``. Instead, ``on_start`` wraps the span's
``set_attribute`` method so that when Cognee calls
``span.set_attribute("cognee.search.query", value)`` we ALSO set the matching
``gen_ai.*`` key on the same span. The wrapping is idempotent.

This replaces the previous ``install_attribute_migration_patch`` which patched
``cognee.modules.observability.new_span`` — that patch broke Cognee's
exception path (``CollectionNotFoundError`` fallback was swallowed, triggering
``RuntimeError: generator didn't stop after throw()``) and never actually ran
because cognee re-imports ``new_span`` lazily in some code paths.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from opentelemetry.context import Context
from opentelemetry.instrumentation.cognee.config import MAX_PAYLOAD_BYTES
from opentelemetry.instrumentation.cognee.semconv import (
    COGNEE_PIPELINE_TASK_NAME,
    COGNEE_RESULT_COUNT,
    COGNEE_RESULT_SUMMARY,
    COGNEE_RETRIEVAL_TOP_K,
    COGNEE_SEARCH_QUERY,
    COGNEE_SEARCH_TOP_K,
    COGNEE_VECTOR_COLLECTION,
)
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

logger = logging.getLogger(__name__)


# prefix -> (gen_ai.span.kind, gen_ai.operation.name, new span name or None)
# When new name is None, the processor derives the name from the original span.
_PREFIX_RULES: tuple[tuple[str, str, str, Optional[str]], ...] = (
    ("cognee.api.cognify", "CHAIN", "workflow", "workflow cognify"),
    ("cognee.api.search", "CHAIN", "workflow", "workflow search"),
    ("cognee.api.recall", "CHAIN", "workflow", "workflow recall"),
    ("cognee.api.remember.import", "CHAIN", "workflow", "workflow remember_import"),
    ("cognee.api.remember", "CHAIN", "workflow", "workflow remember"),
    ("cognee.llm.completion", "CHAIN", "task", "task llm_completion"),
    ("cognee.llm.summarize", "CHAIN", "task", "task llm_summarize"),
    ("cognee.pipeline.task.", "TASK", "run_task", None),
    ("cognee.retrieval.", "RETRIEVER", "retrieval", None),
)

# Cognee attribute -> gen-ai attribute migration table.
# Applied by the ``set_attribute`` interceptor installed on each Cognee span
# in ``CogneeAttributeSpanProcessor.on_start``.
_MIGRATION_MAP: dict[str, str] = {
    COGNEE_SEARCH_QUERY: "gen_ai.retrieval.query.text",
    COGNEE_PIPELINE_TASK_NAME: "gen_ai.task.name",
    COGNEE_RESULT_SUMMARY: "gen_ai.output.value",
    COGNEE_RESULT_COUNT: "gen_ai.output.items_count",
    COGNEE_VECTOR_COLLECTION: "gen_ai.data_source.id",
    COGNEE_SEARCH_TOP_K: "gen_ai.request.top_k",
    COGNEE_RETRIEVAL_TOP_K: "gen_ai.request.top_k",
}


def _truncate(value, max_len: int = MAX_PAYLOAD_BYTES):
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "..."
    return value


class CogneeAttributeSpanProcessor(SpanProcessor):
    """Set gen-ai span.kind / operation.name / name on Cognee-native spans.

    Uses ``on_start`` because the SDK Span becomes immutable after ``end()``.
    The name is known at start time (passed to ``start_span``), so prefix-based
    rewriting works.

    In ``on_start`` we also wrap the span's ``set_attribute`` method so that
    when Cognee code later calls ``span.set_attribute("cognee.search.query", v)``
    we mirror the write to ``gen_ai.retrieval.query.text`` on the same span.
    This is the only reliable hook point because the SDK Span is still mutable
    at ``on_start`` and the wrap is visible to all subsequent ``set_attribute``
    calls in the span body.
    """

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        try:
            name = span.name or ""
            matched = False
            for prefix, kind, op, new_name in _PREFIX_RULES:
                if not name.startswith(prefix):
                    continue
                matched = True
                try:
                    span.set_attribute("gen_ai.span.kind", kind)
                    span.set_attribute("gen_ai.operation.name", op)
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("on_start set_attribute failed: %s", e)
                if new_name:
                    try:
                        span.update_name(new_name)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("update_name failed: %s", e)
                elif prefix == "cognee.pipeline.task.":
                    task_name = name[len(prefix) :]
                    try:
                        span.update_name(f"task {task_name}")
                        # Pre-inject gen_ai.task.name from span name so the
                        # attribute is present even if Cognee never sets
                        # cognee.pipeline.task_name (e.g. for early-exit paths).
                        span.set_attribute("gen_ai.task.name", task_name)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("update_name(task) failed: %s", e)
                elif prefix == "cognee.retrieval.":
                    try:
                        span.update_name(
                            f"retrieval {name[len('cognee.retrieval.') :]}"
                        )
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("update_name(retrieval) failed: %s", e)
                break
            # Wrap set_attribute for any Cognee-native span so cognee.* attrs
            # set during the span body are mirrored to gen_ai.* on the same
            # span. We do this regardless of whether the prefix matched,
            # because Cognee may create spans whose names we don't recognize
            # yet still set cognee.* attributes worth migrating.
            if matched or name.startswith("cognee."):
                _wrap_span_set_attribute(span)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("CogneeAttributeSpanProcessor.on_start failed: %s", e)

    def on_end(self, span: ReadableSpan) -> None:
        # No-op — span is already read-only.
        return

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _wrap_span_set_attribute(span: Any) -> None:
    """Wrap a single span's ``set_attribute`` to mirror cognee.* → gen_ai.*.

    Idempotent — if already wrapped, do nothing.
    """
    if getattr(span, "_cognee_genai_set_attr_wrapped", False):
        return
    original_set_attribute = getattr(span, "set_attribute", None)
    if original_set_attribute is None or not callable(original_set_attribute):
        return

    def _set_attribute(key, value):
        try:
            gen_ai_key = _MIGRATION_MAP.get(key)
            if gen_ai_key is not None:
                # Don't clobber a gen_ai attr set explicitly by the user.
                existing = getattr(span, "attributes", None) or {}
                if gen_ai_key not in existing:
                    original_set_attribute(gen_ai_key, _truncate(value))
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("attribute migration failed for %s: %s", key, e)
        return original_set_attribute(key, value)

    try:
        # Span is a regular Python object — bind the wrapper. We avoid using
        # ``types.MethodType`` because some Span implementations are C-extension.
        span.set_attribute = _set_attribute  # type: ignore[method-assign]
        span._cognee_genai_set_attr_wrapped = True  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as e:
        logger.debug("could not wrap span.set_attribute: %s", e)
