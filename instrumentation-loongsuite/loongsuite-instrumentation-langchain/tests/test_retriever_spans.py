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

"""Tests for Retriever span creation and attributes."""

from typing import List

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from opentelemetry.instrumentation.langchain.internal.semconv import (
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY_TEXT,
)
from opentelemetry.trace import StatusCode


class FakeRetriever(BaseRetriever):
    """A fake retriever for testing."""

    docs: List[Document] = []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self.docs or [
            Document(
                page_content=f"Result for: {query}",
                metadata={"source": "test"},
            )
        ]


class FakeErrorRetriever(BaseRetriever):
    """A fake retriever that always fails."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise ValueError("retriever failure")


def _find_retriever_spans(span_exporter):
    spans = span_exporter.get_finished_spans()
    return [s for s in spans if "retrieval" in s.name.lower()]


class TestRetrieverSpanCreation:
    def test_retriever_creates_span(self, instrument, span_exporter):
        retriever = FakeRetriever()
        docs = retriever.invoke("test query")
        assert len(docs) >= 1

        retriever_spans = _find_retriever_spans(span_exporter)
        assert len(retriever_spans) >= 1

    def test_retriever_error_span(self, instrument, span_exporter):
        retriever = FakeErrorRetriever()
        with pytest.raises(ValueError, match="retriever failure"):
            retriever.invoke("fail query")

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1


class TestRetrieverInputOutputContent:
    """Verify retriever query and documents in span attributes."""

    def test_retrieval_query_captured(self, instrument, span_exporter):
        retriever = FakeRetriever()
        retriever.invoke("machine learning basics")

        retriever_spans = _find_retriever_spans(span_exporter)
        assert len(retriever_spans) >= 1
        attrs = dict(retriever_spans[0].attributes)

        query_val = attrs.get(GEN_AI_RETRIEVAL_QUERY_TEXT, "")
        assert "machine learning basics" in query_val, (
            f"Expected 'machine learning basics' in retrieval.query.text, got: {query_val}"
        )

    def test_retrieval_documents_captured(self, instrument, span_exporter):
        retriever = FakeRetriever()
        retriever.invoke("test docs query")

        retriever_spans = _find_retriever_spans(span_exporter)
        assert len(retriever_spans) >= 1
        attrs = dict(retriever_spans[0].attributes)

        docs_val = attrs.get(GEN_AI_RETRIEVAL_DOCUMENTS, "")
        assert "Result for: test docs query" in docs_val, (
            f"Expected document content in retrieval.documents, got: {docs_val}"
        )

    def test_no_content_when_disabled(
        self, instrument_no_content, span_exporter
    ):
        """NO_CONTENT omits the opt-in query and document attributes."""
        retriever = FakeRetriever()
        retriever.invoke("secret query")

        retriever_spans = _find_retriever_spans(span_exporter)
        assert len(retriever_spans) >= 1
        attrs = dict(retriever_spans[0].attributes)

        assert GEN_AI_RETRIEVAL_QUERY_TEXT not in attrs, (
            "Query should NOT be captured when content capture is disabled"
        )
        assert GEN_AI_RETRIEVAL_DOCUMENTS not in attrs, (
            "Documents are opt-in and should not be captured when disabled"
        )
