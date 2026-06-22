"""SpanProcessor that normalizes Cognee-native spans to gen-ai semantic conventions.

Cognee's own ``new_span`` calls already create spans with names like
``cognee.api.cognify``, ``cognee.pipeline.task.extract_graph``,
``cognee.retrieval.vector_search`` and ``cognee.llm.completion``. We rewrite
their name and inject ``gen_ai.span.kind`` / ``gen_ai.operation.name`` in
``on_start`` (the only point at which the SDK Span is still mutable).

Cognee-specific attributes (``cognee.search.query``, ``cognee.pipeline.task_name`` …)
are set by Cognee's own code during the span body, after ``on_start`` runs.
Because the SDK Span becomes read-only once ``end()`` is called, we cannot
migrate them in ``on_end``. Instead, ``install_attribute_migration_patch``
wraps ``cognee.modules.observability.new_span`` so the span returned to
Cognee's caller has its ``set_attribute`` method intercepted — when Cognee
sets a ``cognee.*`` key, we ALSO set the corresponding ``gen_ai.*`` key on
the same span.
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
# Applied by the set_attribute interceptor installed via
# ``install_attribute_migration_patch``.
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
    """

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        try:
            name = span.name or ""
            for prefix, kind, op, new_name in _PREFIX_RULES:
                if not name.startswith(prefix):
                    continue
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
                    try:
                        span.update_name(f"task {name[len(prefix):]}")
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("update_name(task) failed: %s", e)
                elif prefix == "cognee.retrieval.":
                    try:
                        span.update_name(f"retrieval {name[len('cognee.retrieval.') :]}")
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("update_name(retrieval) failed: %s", e)
                return
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("CogneeAttributeSpanProcessor.on_start failed: %s", e)

    def on_end(self, span: ReadableSpan) -> None:
        # No-op — span is already read-only.
        return

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def install_attribute_migration_patch() -> None:
    """Patch ``cognee.modules.observability.new_span`` to migrate cognee.* attrs.

    When Cognee calls ``span.set_attribute("cognee.search.query", value)``, we
    also call ``span.set_attribute("gen_ai.retrieval.query.text", value)`` on
    the same span. This runs before ``end()`` so the SDK Span is still mutable.
    """
    try:
        import cognee.modules.observability as cognee_obs  # type: ignore
    except ImportError:
        logger.debug(
            "cognee.modules.observability not importable — attribute migration patch skipped"
        )
        return

    original = getattr(cognee_obs, "new_span", None)
    if original is None or getattr(original, "_cognee_genai_patched", False):
        return

    try:
        import contextlib

        @contextlib.contextmanager
        def _patched_new_span(name: str):
            ctx = original(name) if original else _noop_ctx()
            try:
                with ctx as span:
                    if span is not None and not isinstance(span, _NullSpanType):
                        _wrap_span_set_attribute(span)
                    yield span
            except Exception:
                # Fall back to yielding a null span if the original blows up.
                yield cognee_obs._NullSpan()

        def _noop_ctx():
            import contextlib

            @contextlib.contextmanager
            def _cm():
                yield None

            return _cm()

        _patched_new_span._cognee_genai_patched = True  # type: ignore[attr-defined]
        cognee_obs.new_span = _patched_new_span
    except Exception as e:
        logger.debug("install_attribute_migration_patch failed: %s", e)


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


# Late import to avoid hard dependency on cognee at module load.
try:
    from cognee.modules.observability import (
        _NullSpan as _NullSpanType,  # type: ignore
    )
except ImportError:  # pragma: no cover - cognee not installed

    class _NullSpanType:  # type: ignore[no-redef]
        pass
