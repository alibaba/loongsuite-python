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

"""
Embedding instrumentation patch for LangChain Embeddings.

Because ``embed_documents`` and ``embed_query`` are abstract, every
subclass overrides them.  We use the same strategy as ``patch_rerank.py``:

1. Retroactively wrap methods on **every existing subclass** at
   instrumentation time.
2. Install a ``__init_subclass__`` hook on ``Embeddings`` so that
   any subclass defined **after** instrumentation is also wrapped
   automatically.
"""

from __future__ import annotations

import contextvars
import logging
from typing import TYPE_CHECKING, Any

import wrapt

from opentelemetry.util.genai.extended_types import EmbeddingInvocation
from opentelemetry.util.genai.types import Error

if TYPE_CHECKING:
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
    )

logger = logging.getLogger(__name__)

# Depth counter to avoid duplicate spans when a proxy embeddings
# delegates to an inner embeddings (both subclasses are patched),
# or when default aembed_* calls patched embed_* via run_in_executor.
_EMBEDDING_CALL_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "opentelemetry_langchain_embedding_call_depth",
    default=0,
)

# Module-level state for uninstrumentation.
_original_init_subclass: Any = None
_patched_classes: set[type] = set()
_already_instrumented: bool = False

_WRAPPER_TAG = "_loongsuite_embedding_wrapped"

_SYNC_METHODS = ("embed_documents", "embed_query")
_ASYNC_METHODS = ("aembed_documents", "aembed_query")


# ---------------------------------------------------------------------------
# Helpers — metadata extraction
# ---------------------------------------------------------------------------


def _extract_embedding_provider(instance: Any) -> str:
    """Infer a provider name from an Embeddings instance."""
    cls_name = type(instance).__name__
    module = type(instance).__module__ or ""

    _HINTS = [
        ("openai", "openai"),
        ("azure", "azure"),
        ("cohere", "cohere"),
        ("huggingface", "huggingface"),
        ("sentence_transformers", "sentence_transformers"),
        ("google", "google"),
        ("bedrock", "aws_bedrock"),
        ("ollama", "ollama"),
        ("jina", "jina"),
        ("voyage", "voyageai"),
        ("mistral", "mistral"),
        ("dashscope", "dashscope"),
        ("together", "together"),
        ("fireworks", "fireworks"),
    ]
    lower = (module + "." + cls_name).lower()
    for hint, provider in _HINTS:
        if hint in lower:
            return provider

    return "langchain"


def _extract_embedding_model(instance: Any) -> str:
    """Extract a model name from an Embeddings instance.

    Falls back to the class name when no explicit model attribute is
    present, so spans never carry an empty model identifier.
    """
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        val = getattr(instance, attr, None)
        if val and isinstance(val, str):
            return val

    cls = type(instance)
    return cls.__name__ or repr(cls)


def _extract_server_address_port(
    instance: Any,
) -> tuple[str | None, int | None]:
    """Extract server address and port from an Embeddings instance."""
    from urllib.parse import urlparse  # noqa: PLC0415

    for attr in (
        "openai_api_base",
        "base_url",
        "api_base",
        "endpoint",
        "endpoint_url",
    ):
        val = getattr(instance, attr, None)
        if val and isinstance(val, str):
            try:
                parsed = urlparse(val)
                host = parsed.hostname
                port = parsed.port
                return host, port
            except Exception:
                continue

    # Some providers store a client object with base_url
    client = getattr(instance, "client", None)
    if client is not None:
        client_base = getattr(client, "base_url", None)
        if client_base:
            url_str = str(client_base)
            try:
                parsed = urlparse(url_str)
                return parsed.hostname, parsed.port
            except Exception:
                pass

    return None, None


def _extract_dimension_count(instance: Any) -> int | None:
    """Extract embedding dimension count from an Embeddings instance."""
    for attr in ("dimensions", "dimension", "embedding_dim"):
        val = getattr(instance, attr, None)
        if val is not None and isinstance(val, int) and val > 0:
            return val
    return None


def _extract_encoding_formats(instance: Any) -> list[str] | None:
    """Extract encoding formats from an Embeddings instance."""
    for attr in ("encoding_format", "embedding_format"):
        val = getattr(instance, attr, None)
        if val and isinstance(val, str):
            return [val]
        if val and isinstance(val, list):
            return val
    return None


# ---------------------------------------------------------------------------
# Wrapper factories
# ---------------------------------------------------------------------------


def _build_invocation(instance: Any) -> EmbeddingInvocation:
    """Build an ``EmbeddingInvocation`` with all extractable attributes."""
    server_address, server_port = _extract_server_address_port(instance)
    return EmbeddingInvocation(
        request_model=_extract_embedding_model(instance),
        provider=_extract_embedding_provider(instance),
        server_address=server_address,
        server_port=server_port,
        dimension_count=_extract_dimension_count(instance),
        encoding_formats=_extract_encoding_formats(instance),
    )


def _make_sync_wrapper(handler: "ExtendedTelemetryHandler") -> Any:
    """Return a ``wrapt``-style wrapper for sync embedding methods.

    Used for both ``embed_documents`` and ``embed_query`` since their
    instrumentation logic is identical.
    """

    def wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        parent_depth = _EMBEDDING_CALL_DEPTH.get()
        depth_token = _EMBEDDING_CALL_DEPTH.set(parent_depth + 1)
        try:
            if parent_depth > 0:
                return wrapped(*args, **kwargs)

            invocation = _build_invocation(instance)

            try:
                handler.start_embedding(invocation)
            except Exception:
                logger.debug("Failed to start embedding span", exc_info=True)
                return wrapped(*args, **kwargs)

            try:
                result = wrapped(*args, **kwargs)
                handler.stop_embedding(invocation)
                return result
            except Exception as exc:
                handler.fail_embedding(
                    invocation, Error(message=str(exc), type=type(exc))
                )
                raise
        finally:
            _EMBEDDING_CALL_DEPTH.reset(depth_token)

    return wrapper


def _make_async_wrapper(handler: "ExtendedTelemetryHandler") -> Any:
    """Return a ``wrapt``-style wrapper for async embedding methods.

    Used for both ``aembed_documents`` and ``aembed_query`` since their
    instrumentation logic is identical.
    """

    def wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        async def _instrumented() -> Any:
            parent_depth = _EMBEDDING_CALL_DEPTH.get()
            depth_token = _EMBEDDING_CALL_DEPTH.set(parent_depth + 1)
            try:
                if parent_depth > 0:
                    return await wrapped(*args, **kwargs)

                invocation = _build_invocation(instance)

                try:
                    handler.start_embedding(invocation)
                except Exception:
                    logger.debug(
                        "Failed to start embedding span", exc_info=True
                    )
                    return await wrapped(*args, **kwargs)

                try:
                    result = await wrapped(*args, **kwargs)
                    handler.stop_embedding(invocation)
                    return result
                except Exception as exc:
                    handler.fail_embedding(
                        invocation, Error(message=str(exc), type=type(exc))
                    )
                    raise
            finally:
                _EMBEDDING_CALL_DEPTH.reset(depth_token)

        return _instrumented()

    return wrapper


# ---------------------------------------------------------------------------
# Subclass discovery
# ---------------------------------------------------------------------------


def _all_subclasses(cls: type) -> set[type]:
    """Recursively collect every subclass of *cls*."""
    result: set[type] = set()
    queue = list(cls.__subclasses__())
    while queue:
        sub = queue.pop()
        if sub not in result:
            result.add(sub)
            queue.extend(sub.__subclasses__())
    return result


# ---------------------------------------------------------------------------
# Per-class patching / unpatching
# ---------------------------------------------------------------------------


def _patch_class(
    cls: type,
    sync_wrapper: Any,
    async_wrapper: Any,
) -> None:
    """Wrap embedding methods on *cls*.

    Only wraps methods that are defined directly in *cls* (i.e. present
    in ``cls.__dict__``).  Skips classes that are already wrapped.
    """
    if getattr(cls, _WRAPPER_TAG, False):
        return

    _method_wrappers = {
        "embed_documents": sync_wrapper,
        "embed_query": sync_wrapper,
        "aembed_documents": async_wrapper,
        "aembed_query": async_wrapper,
    }

    for method_name, wrapper_fn in _method_wrappers.items():
        if method_name in cls.__dict__:
            original = cls.__dict__[method_name]
            if not isinstance(original, wrapt.FunctionWrapper):
                setattr(
                    cls,
                    method_name,
                    wrapt.FunctionWrapper(original, wrapper_fn),
                )

    setattr(cls, _WRAPPER_TAG, True)
    _patched_classes.add(cls)


def _unpatch_class(cls: type) -> None:
    """Restore original methods on *cls*."""
    for method_name in (*_SYNC_METHODS, *_ASYNC_METHODS):
        method = cls.__dict__.get(method_name)
        if isinstance(method, wrapt.FunctionWrapper):
            setattr(cls, method_name, method.__wrapped__)

    try:
        delattr(cls, _WRAPPER_TAG)
    except AttributeError:
        pass


def instrument_embeddings(
    handler: "ExtendedTelemetryHandler",
) -> None:
    """Wrap all current and future ``Embeddings`` subclasses.

    This function is idempotent: calling it more than once without an
    intervening :func:`uninstrument_embeddings` is a no-op.  This guards
    against losing the true ``__init_subclass__`` original to the
    already-patched hook on a re-instrument.
    """
    global _original_init_subclass, _already_instrumented  # noqa: PLW0603

    if _already_instrumented:
        return

    try:
        from langchain_core.embeddings import Embeddings  # noqa: PLC0415
    except ImportError as exc:
        logger.debug(
            "Embeddings not available, skipping embedding instrumentation: %s",
            exc,
        )
        return

    sync_wrapper = _make_sync_wrapper(handler)
    async_wrapper = _make_async_wrapper(handler)

    # 1. Retroactively patch every existing subclass.
    for cls in _all_subclasses(Embeddings):
        _patch_class(cls, sync_wrapper, async_wrapper)

    # 2. Install an __init_subclass__ hook so future subclasses are
    #    patched automatically.
    _original_init_subclass = Embeddings.__dict__.get("__init_subclass__")

    @classmethod  # type: ignore[misc]
    def _patched_init_subclass(cls: type, **kwargs: Any) -> None:
        if _original_init_subclass is not None:
            if isinstance(_original_init_subclass, classmethod):
                _original_init_subclass.__func__(cls, **kwargs)
            else:
                _original_init_subclass(**kwargs)
        else:
            super(Embeddings, cls).__init_subclass__(**kwargs)
        _patch_class(cls, sync_wrapper, async_wrapper)

    Embeddings.__init_subclass__ = _patched_init_subclass  # type: ignore[assignment]
    _already_instrumented = True

    logger.debug(
        "Patched Embeddings (%d existing subclass(es))",
        len(_patched_classes),
    )


def uninstrument_embeddings() -> None:
    """Restore original methods on all patched embeddings classes."""
    global _original_init_subclass, _already_instrumented  # noqa: PLW0603

    if not _already_instrumented:
        return

    try:
        from langchain_core.embeddings import Embeddings  # noqa: PLC0415

        if _original_init_subclass is not None:
            Embeddings.__init_subclass__ = _original_init_subclass  # type: ignore[assignment]
        else:
            if "__init_subclass__" in Embeddings.__dict__:
                delattr(Embeddings, "__init_subclass__")
    except Exception:
        logger.debug(
            "Failed to restore Embeddings.__init_subclass__",
            exc_info=True,
        )

    for cls in list(_patched_classes):
        try:
            _unpatch_class(cls)
        except Exception:
            logger.debug("Failed to unpatch %s", cls, exc_info=True)
    _patched_classes.clear()
    _original_init_subclass = None
    _already_instrumented = False

    logger.debug("Restored Embeddings subclasses")
