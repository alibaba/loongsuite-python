# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

from google.genai.models import AsyncModels, Models
from google.genai.types import (
    ContentEmbedding,
    ContentEmbeddingStatistics,
    EmbedContentResponse,
)

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import StatusCode

from ..common.base import TestCase

_EMBEDDING_DIMENSION_COUNT = getattr(
    GenAIAttributes,
    "GEN_AI_EMBEDDINGS_DIMENSION_COUNT",
    "gen_ai.embeddings.dimension.count",
)


class TestEmbeddings(TestCase):
    def setUp(self):
        super().setUp()
        self._original_embed_content = Models.embed_content
        self._original_async_embed_content = AsyncModels.embed_content

        self.mock_response = EmbedContentResponse(
            embeddings=[
                ContentEmbedding(
                    values=[0.1, 0.2, 0.3],
                    statistics=ContentEmbeddingStatistics(
                        token_count=5,
                        truncated=False,
                    ),
                )
            ]
        )

        self.embed_content_mock = MagicMock(return_value=self.mock_response)
        self.async_embed_content_mock = AsyncMock(
            return_value=self.mock_response
        )

        def _sync_embed_wrapped(self_obj, *args, **kwargs):
            return self.embed_content_mock(*args, **kwargs)

        async def _async_embed_wrapped(self_obj, *args, **kwargs):
            return await self.async_embed_content_mock(*args, **kwargs)

        Models.embed_content = _sync_embed_wrapped
        AsyncModels.embed_content = _async_embed_wrapped

    def tearDown(self):
        super().tearDown()
        Models.embed_content = self._original_embed_content
        AsyncModels.embed_content = self._original_async_embed_content

    def test_sync_embed_content(self):
        response = self.client.models.embed_content(
            model="text-embedding-004",
            contents="hello world",
        )

        self.assertEqual(response, self.mock_response)
        self.embed_content_mock.assert_called_once_with(
            model="text-embedding-004",
            contents="hello world",
        )

        spans = self.otel.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.name, "embeddings text-embedding-004")
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        attrs = span.attributes
        self.assertEqual(
            attrs[GenAIAttributes.GEN_AI_OPERATION_NAME], "embeddings"
        )
        self.assertEqual(attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME], "gemini")
        self.assertEqual(
            attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], "text-embedding-004"
        )
        self.assertEqual(attrs[_EMBEDDING_DIMENSION_COUNT], 3)
        self.assertEqual(
            attrs["server.address"],
            "generativelanguage.googleapis.com",
        )

    def test_async_embed_content(self):
        async def run_test():
            response = await self.client.aio.models.embed_content(
                model="text-embedding-004",
                contents="hello world",
            )
            self.assertEqual(response, self.mock_response)
            self.async_embed_content_mock.assert_called_once_with(
                model="text-embedding-004",
                contents="hello world",
            )

        asyncio.run(run_test())

        spans = self.otel.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.name, "embeddings text-embedding-004")
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        attrs = span.attributes
        self.assertEqual(
            attrs[GenAIAttributes.GEN_AI_OPERATION_NAME], "embeddings"
        )
        self.assertEqual(attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME], "gemini")
        self.assertEqual(
            attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], "text-embedding-004"
        )
        self.assertEqual(attrs[_EMBEDDING_DIMENSION_COUNT], 3)
        self.assertEqual(
            attrs["server.address"],
            "generativelanguage.googleapis.com",
        )

    def test_embed_content_multiple_inputs(self):
        _ = self.client.models.embed_content(
            model="text-embedding-004",
            contents=["hello", "world"],
        )

        spans = self.otel.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        attrs = span.attributes
        self.assertEqual(
            attrs[GenAIAttributes.GEN_AI_OPERATION_NAME], "embeddings"
        )
        self.assertEqual(attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME], "gemini")
        self.assertEqual(
            attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], "text-embedding-004"
        )
        self.assertEqual(attrs[_EMBEDDING_DIMENSION_COUNT], 3)

    def test_embed_content_error(self):
        self.embed_content_mock.side_effect = ValueError("invalid model")

        with self.assertRaises(ValueError):
            self.client.models.embed_content(
                model="bad-model",
                contents="test",
            )

        spans = self.otel.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.status.description, "invalid model")
