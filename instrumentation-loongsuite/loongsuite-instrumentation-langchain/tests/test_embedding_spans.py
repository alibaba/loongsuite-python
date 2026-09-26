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

"""Tests for embedding span creation and attributes."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.embeddings import Embeddings

from opentelemetry.instrumentation.langchain.internal import patch_embedding
from opentelemetry.trace import StatusCode

# ---------------------------------------------------------------------------
# Fake embeddings for testing
# ---------------------------------------------------------------------------


class FakeEmbeddings(Embeddings):
    """Basic fake embeddings with model_name."""

    model_name: str = "fake-embed-model"

    def __init__(self, model_name: str = "fake-embed-model"):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeOpenAIEmbeddings(Embeddings):
    """Fake embeddings with OpenAI-style attributes for server/dimension tests."""

    model_name: str = "text-embedding-3-small"
    openai_api_base: str = "https://api.openai.com:443/v1"
    dimensions: int = 1536

    def __init__(self):
        self.model_name = "text-embedding-3-small"
        self.openai_api_base = "https://api.openai.com:443/v1"
        self.dimensions = 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeErrorEmbeddings(Embeddings):
    """Embeddings that always fail."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise ValueError("embedding failure")

    def embed_query(self, text: str) -> list[float]:
        raise ValueError("embedding failure")


class FakeAsyncEmbeddings(Embeddings):
    """Embeddings with native async implementations."""

    model_name: str = "fake-async-embed-model"

    def __init__(self, model_name: str = "fake-async-embed-model"):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.4, 0.5] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.4, 0.5]


class FakeAsyncErrorEmbeddings(Embeddings):
    """Async embeddings that always fail."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise ValueError("sync embedding failure")

    def embed_query(self, text: str) -> list[float]:
        raise ValueError("sync embedding failure")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raise ValueError("async embedding failure")

    async def aembed_query(self, text: str) -> list[float]:
        raise ValueError("async embedding failure")


class FakeProxyEmbeddings(Embeddings):
    """A proxy that delegates to an inner Embeddings instance."""

    def __init__(self):
        self.inner = FakeEmbeddings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBEDDING_SPAN_NAME = "embeddings"


def _find_embedding_spans(span_exporter):
    spans = span_exporter.get_finished_spans()
    return [s for s in spans if s.name.startswith(_EMBEDDING_SPAN_NAME)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingSpanCreation:
    def test_embed_documents_creates_span(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        result = emb.embed_documents(["hello", "world"])
        assert len(result) == 2

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1

    def test_embed_query_creates_span(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        result = emb.embed_query("hello")
        assert isinstance(result, list)

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1

    def test_embed_documents_error_span(self, instrument, span_exporter):
        emb = FakeErrorEmbeddings()
        with pytest.raises(ValueError, match="embedding failure"):
            emb.embed_documents(["fail"])

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1

    def test_embed_query_error_span(self, instrument, span_exporter):
        emb = FakeErrorEmbeddings()
        with pytest.raises(ValueError, match="embedding failure"):
            emb.embed_query("fail")

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1


class TestEmbeddingSpanAttributes:
    def test_operation_name(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.operation.name") == "embeddings"

    def test_span_kind_attribute(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.span.kind") == "EMBEDDING"

    def test_provider_attribute(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.provider.name") == "langchain"

    def test_model_attribute(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.request.model") == "fake-embed-model"

    def test_span_name_includes_model(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        assert "fake-embed-model" in spans[0].name

    def test_server_address_and_port(self, instrument, span_exporter):
        emb = FakeOpenAIEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("server.address") == "api.openai.com"
        assert attrs.get("server.port") == 443

    def test_dimension_count(self, instrument, span_exporter):
        emb = FakeOpenAIEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.embeddings.dimension.count") == 1536

    def test_no_server_attrs_when_absent(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert "server.address" not in attrs
        assert "server.port" not in attrs
        assert "gen_ai.embeddings.dimension.count" not in attrs


class TestAsyncEmbeddingSpans:
    def test_async_embed_documents_creates_span(
        self, instrument, span_exporter
    ):
        emb = FakeAsyncEmbeddings()
        result = asyncio.run(emb.aembed_documents(["hello", "world"]))
        assert len(result) == 2

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1

    def test_async_embed_query_creates_span(self, instrument, span_exporter):
        emb = FakeAsyncEmbeddings()
        result = asyncio.run(emb.aembed_query("hello"))
        assert isinstance(result, list)

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1

    def test_async_embed_documents_attributes(self, instrument, span_exporter):
        emb = FakeAsyncEmbeddings()
        asyncio.run(emb.aembed_documents(["test"]))

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.operation.name") == "embeddings"
        assert attrs.get("gen_ai.span.kind") == "EMBEDDING"
        assert attrs.get("gen_ai.request.model") == "fake-async-embed-model"

    def test_async_embed_documents_error_span(self, instrument, span_exporter):
        emb = FakeAsyncErrorEmbeddings()
        with pytest.raises(ValueError, match="async embedding failure"):
            asyncio.run(emb.aembed_documents(["fail"]))

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1

    def test_async_embed_query_error_span(self, instrument, span_exporter):
        emb = FakeAsyncErrorEmbeddings()
        with pytest.raises(ValueError, match="async embedding failure"):
            asyncio.run(emb.aembed_query("fail"))

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1


class TestEmbeddingDeduplication:
    def test_proxy_embed_documents_single_span(
        self, instrument, span_exporter
    ):
        """A proxy that delegates to an inner embeddings should produce
        exactly one embedding span, not two."""
        proxy = FakeProxyEmbeddings()
        result = proxy.embed_documents(["test"])
        assert len(result) == 1

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) == 1, (
            f"Expected exactly 1 embedding span, got {len(spans)}"
        )

    def test_proxy_embed_query_single_span(self, instrument, span_exporter):
        proxy = FakeProxyEmbeddings()
        result = proxy.embed_query("test")
        assert isinstance(result, list)

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) == 1, (
            f"Expected exactly 1 embedding span, got {len(spans)}"
        )

    def test_direct_embeddings_still_creates_span(
        self, instrument, span_exporter
    ):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) == 1


class TestEmbeddingInitSubclassHook:
    def test_post_instrumentation_subclass_creates_span(
        self, instrument, span_exporter
    ):
        class LateDefinedEmbeddings(Embeddings):
            model_name: str = "late-embed-model"

            def __init__(self):
                self.model_name = "late-embed-model"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [1.0]

        emb = LateDefinedEmbeddings()
        emb.embed_documents(["hello"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.request.model") == "late-embed-model"

    def test_post_instrumentation_async_subclass_creates_span(
        self, instrument, span_exporter
    ):
        class LateAsyncEmbeddings(Embeddings):
            model_name: str = "late-async-embed-model"

            def __init__(self):
                self.model_name = "late-async-embed-model"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [1.0]

            async def aembed_documents(
                self, texts: list[str]
            ) -> list[list[float]]:
                return [[2.0] for _ in texts]

        emb = LateAsyncEmbeddings()
        asyncio.run(emb.aembed_documents(["hello"]))

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.request.model") == "late-async-embed-model"


class TestEmbeddingUninstrumentation:
    def test_no_spans_after_uninstrument(self, instrument, span_exporter):
        emb = FakeEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1

        instrument.uninstrument()
        span_exporter.clear()

        emb.embed_documents(["test"])
        spans = _find_embedding_spans(span_exporter)
        assert len(spans) == 0

    def test_double_instrument_then_uninstrument_restores_hook(
        self, instrument, span_exporter
    ):
        """Re-instrumenting before uninstrument must not corrupt the
        captured ``__init_subclass__`` original.  Otherwise uninstrument
        would restore the *patched* hook instead of the true original,
        and subclasses defined post-uninstrument would still be auto-patched.

        Calls ``instrument_embeddings`` directly to bypass the
        instrumentor's framework-level double-instrument guard, so this
        regression test actually exercises the module-level idempotency
        flag added in :mod:`patch_embedding`.
        """
        original_hook = patch_embedding._original_init_subclass

        # Second call would have clobbered ``_original_init_subclass``
        # with the already-patched hook prior to the idempotency fix.
        # ``handler`` argument is unused when the guard short-circuits.
        patch_embedding.instrument_embeddings(handler=None)

        assert patch_embedding._original_init_subclass is original_hook, (
            "Re-instrument must not overwrite the captured original hook"
        )

        instrument.uninstrument()
        span_exporter.clear()

        class PostUninstrumentEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[9.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [9.0]

        emb = PostUninstrumentEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) == 0, (
            "Subclasses defined after uninstrument must not be auto-patched"
        )


class TestEmbeddingModelFallback:
    def test_missing_model_attribute_falls_back_to_class_name(
        self, instrument, span_exporter
    ):
        """When no model attribute is exposed, the span must still carry a
        non-empty ``gen_ai.request.model`` (using the class name) so that
        the GenAI semantic-convention required attribute is never dropped.
        """

        class NoModelAttrEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [0.0]

        emb = NoModelAttrEmbeddings()
        emb.embed_documents(["test"])

        spans = _find_embedding_spans(span_exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert attrs.get("gen_ai.request.model") == "NoModelAttrEmbeddings"
        # Span name must not end with a trailing space from an empty model.
        assert spans[0].name.strip() == spans[0].name
        assert "NoModelAttrEmbeddings" in spans[0].name
