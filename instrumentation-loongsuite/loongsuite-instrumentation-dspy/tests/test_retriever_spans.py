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

"""RETRIEVER span coverage for ``dspy.Retrieve``."""

import json

import dspy
import pytest

from opentelemetry.instrumentation.dspy.internal.semconv import (
    GEN_AI_OPERATION_NAME,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY_TEXT,
)

from ._helpers import single


class _Passage:
    def __init__(self, long_text):
        self.long_text = long_text


def _dummy_rm(query, k=3, **kwargs):
    return [_Passage(f"passage {index} about {query}") for index in range(k)]


@pytest.mark.usefixtures("instrument")
def test_retrieve_emits_retriever_span(span_exporter):
    dspy.settings.configure(rm=_dummy_rm)

    result = dspy.Retrieve(k=2)(query="OpenTelemetry")

    assert len(result.passages) == 2
    spans = span_exporter.get_finished_spans()
    retriever = single(spans, "RETRIEVER")

    assert retriever.name == "retrieval Retrieve"
    assert retriever.attributes[GEN_AI_OPERATION_NAME] == "retrieval"
    assert retriever.attributes[GenAI.GEN_AI_DATA_SOURCE_ID] == "Retrieve"
    assert retriever.attributes[GenAI.GEN_AI_REQUEST_TOP_K] == 2
    assert retriever.attributes[GEN_AI_RETRIEVAL_QUERY_TEXT] == "OpenTelemetry"

    documents = json.loads(retriever.attributes[GEN_AI_RETRIEVAL_DOCUMENTS])
    assert [doc["id"] for doc in documents] == ["0", "1"]
    # DSPy retrievers do not return relevance scores; none is invented.
    assert all(doc.get("score") is None for doc in documents)


@pytest.mark.usefixtures("instrument")
def test_retrieve_top_k_falls_back_to_instance(span_exporter):
    dspy.settings.configure(rm=_dummy_rm)

    dspy.Retrieve(k=5)(query="OpenTelemetry")

    retriever = single(span_exporter.get_finished_spans(), "RETRIEVER")
    assert retriever.attributes[GenAI.GEN_AI_REQUEST_TOP_K] == 5


@pytest.mark.usefixtures("instrument_no_content")
def test_retrieve_omits_query_and_documents_without_content_capture(
    span_exporter,
):
    dspy.settings.configure(rm=_dummy_rm)

    dspy.Retrieve(k=2)(query="OpenTelemetry")

    retriever = single(span_exporter.get_finished_spans(), "RETRIEVER")
    assert GEN_AI_RETRIEVAL_QUERY_TEXT not in retriever.attributes


@pytest.mark.usefixtures("instrument")
def test_retrieve_failure_marks_span_as_error(span_exporter):
    def _failing_rm(query, k=3, **kwargs):
        raise RuntimeError("index unavailable")

    dspy.settings.configure(rm=_failing_rm)

    with pytest.raises(RuntimeError):
        dspy.Retrieve(k=2)(query="OpenTelemetry")

    retriever = single(span_exporter.get_finished_spans(), "RETRIEVER")
    assert retriever.status.status_code.name == "ERROR"
