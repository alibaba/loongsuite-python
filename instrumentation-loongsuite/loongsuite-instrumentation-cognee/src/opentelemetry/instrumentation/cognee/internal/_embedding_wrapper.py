"""EMBEDDING wrapper for non-LiteLLM Cognee embedding adapters.

LiteLLMEmbeddingEngine is intentionally NOT wrapped here — its embeddings go
through ``litellm.aembedding``, which is already instrumented by
``loongsuite-instrumentation-litellm``. Wrapping it would duplicate the
EMBEDDING span.

Three adapters are wrapped:
- OllamaEmbeddingEngine        (provider="ollama")
- FastembedEmbeddingEngine     (provider="fastembed")
- OpenAICompatibleEmbeddingEngine (provider="openai")
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.cognee.config import (
    is_internal_phases_enabled,
)
from opentelemetry.instrumentation.cognee.internal._util import (
    maybe_capture,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import EmbeddingInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


# (module, class_name, provider_name)
_EMBEDDING_TARGETS: tuple[tuple[str, str, str], ...] = (
    (
        "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine",
        "OllamaEmbeddingEngine",
        "ollama",
    ),
    (
        "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine",
        "FastembedEmbeddingEngine",
        "fastembed",
    ),
    (
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine",
        "OpenAICompatibleEmbeddingEngine",
        "openai",
    ),
)


def _resolve_request_model(instance: Any) -> str:
    model = getattr(instance, "model", None)
    if model is None:
        return ""
    return str(model)


def _resolve_dimension_count(instance: Any) -> int | None:
    dims = getattr(instance, "dimensions", None)
    if dims is None:
        getter = getattr(instance, "get_vector_size", None)
        if callable(getter):
            try:
                dims = getter()
            except Exception:
                dims = None
    try:
        return int(dims) if dims is not None else None
    except (TypeError, ValueError):
        return None


def _make_embedding_wrapper(
    handler: ExtendedTelemetryHandler,
    provider: str,
) -> Callable:
    async def _embedding_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        # Internal-phases gate: disabled by default. LiteLLM path covers the
        # common case; the user opts into these adapter spans explicitly.
        if not is_internal_phases_enabled():
            return await wrapped(*args, **kwargs)

        request_model = _resolve_request_model(instance)
        dimension_count = _resolve_dimension_count(instance)

        invocation = EmbeddingInvocation(
            request_model=request_model,
            provider=provider,
            dimension_count=dimension_count,
            encoding_formats=["float"],
        )
        handler.start_embedding(invocation)
        try:
            result = await wrapped(*args, **kwargs)
            captured = maybe_capture(result)
            if captured is not None:
                invocation.attributes["gen_ai.embedding.response"] = captured
            handler.stop_embedding(invocation)
            return result
        except Exception as e:
            handler.fail_embedding(invocation, Error(message=str(e), type=type(e)))
            raise

    return _embedding_wrapper


def install_embedding_wrappers(handler: ExtendedTelemetryHandler) -> None:
    if not is_internal_phases_enabled():
        logger.debug(
            "Cognee internal phases disabled — skipping non-LiteLLM embedding wrappers"
        )
        return
    for module, class_name, provider in _EMBEDDING_TARGETS:
        try:
            wrap_function_wrapper(
                module,
                f"{class_name}.embed_text",
                _make_embedding_wrapper(handler, provider),
            )
        except Exception as e:
            logger.debug(
                "Failed to wrap %s.%s.embed_text: %s", module, class_name, e
            )


def uninstall_embedding_wrappers() -> None:
    from opentelemetry.instrumentation.utils import unwrap

    for module, class_name, _ in _EMBEDDING_TARGETS:
        try:
            unwrap(module, f"{class_name}.embed_text")
        except Exception as e:
            logger.debug(
                "Failed to unwrap %s.%s.embed_text: %s", module, class_name, e
            )
