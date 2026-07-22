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

"""Tests for DeepAgents instrumentor lifecycle."""

from __future__ import annotations

from opentelemetry.instrumentation.deepagents import DeepAgentsInstrumentor
from opentelemetry.instrumentation.deepagents.internal.patch import (
    DEEPAGENTS_METADATA_KEY,
    GRAPH_METHODS_WRAPPED_ATTR,
    GRAPH_ORIGINAL_METHODS_ATTR,
    REACT_AGENT_METADATA_KEY,
    _mark_graph,
    _wrap_graph_methods,
    uninstrument_create_deep_agent,
)
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.langgraph import LangGraphInstrumentor


class _DummyGraph:
    def __init__(self):
        self.seen_stream_config = None
        self.bound_config = None

    def _effective_config(self, config):
        if self.bound_config is None:
            return config
        if config is None:
            return self.bound_config

        merged = {**self.bound_config, **config}
        metadata = dict(self.bound_config.get("metadata") or {})
        metadata.update(config.get("metadata") or {})
        merged["metadata"] = metadata
        return merged

    def invoke(self, _input, config=None):
        return self._effective_config(config)

    def stream(self, _input, config=None):
        effective_config = self._effective_config(config)
        self.seen_stream_config = effective_config
        return iter([effective_config])

    def with_config(self, config=None, **_kwargs):
        graph = _DummyGraph()
        graph.bound_config = config
        return graph


def _cleanup_instrumentors() -> None:
    for instrumentor in (
        DeepAgentsInstrumentor(),
        LangGraphInstrumentor(),
        LangChainInstrumentor(),
    ):
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_uninstrument_cleans_dependencies_it_instrumented():
    _cleanup_instrumentors()
    instrumentor = DeepAgentsInstrumentor()

    try:
        instrumentor.instrument()

        assert LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert LangGraphInstrumentor().is_instrumented_by_opentelemetry

        instrumentor.uninstrument()

        assert not instrumentor.is_instrumented_by_opentelemetry
        assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert not LangGraphInstrumentor().is_instrumented_by_opentelemetry
    finally:
        _cleanup_instrumentors()


def test_uninstrument_restores_wrapped_graph_methods():
    _cleanup_instrumentors()
    graph = _DummyGraph()

    try:
        _mark_graph(graph)
        _wrap_graph_methods(graph)

        assert getattr(graph, GRAPH_METHODS_WRAPPED_ATTR) is True
        assert getattr(graph, REACT_AGENT_METADATA_KEY) is True
        assert getattr(graph, DEEPAGENTS_METADATA_KEY) is True

        invoke_config = graph.invoke("hello")
        assert invoke_config["metadata"][DEEPAGENTS_METADATA_KEY] is True

        stream_iter = graph.stream("hello")
        assert (
            graph.seen_stream_config["metadata"][DEEPAGENTS_METADATA_KEY]
            is True
        )
        assert next(stream_iter)["metadata"][DEEPAGENTS_METADATA_KEY] is True

        child_graph = graph.with_config({"metadata": {"customer": "kept"}})
        assert getattr(child_graph, GRAPH_METHODS_WRAPPED_ATTR) is True
        assert getattr(child_graph, DEEPAGENTS_METADATA_KEY) is True
        child_config = child_graph.invoke("hello")
        assert child_config["metadata"]["customer"] == "kept"
        assert child_config["metadata"][DEEPAGENTS_METADATA_KEY] is True

        uninstrument_create_deep_agent()

        assert not hasattr(graph, GRAPH_METHODS_WRAPPED_ATTR)
        assert not hasattr(graph, GRAPH_ORIGINAL_METHODS_ATTR)
        assert not hasattr(graph, REACT_AGENT_METADATA_KEY)
        assert not hasattr(graph, DEEPAGENTS_METADATA_KEY)
        assert not hasattr(child_graph, GRAPH_METHODS_WRAPPED_ATTR)
        assert not hasattr(child_graph, GRAPH_ORIGINAL_METHODS_ATTR)
        assert not hasattr(child_graph, REACT_AGENT_METADATA_KEY)
        assert not hasattr(child_graph, DEEPAGENTS_METADATA_KEY)
        assert graph.invoke("hello") is None
        assert child_graph.invoke("hello") == {
            "metadata": {"customer": "kept"}
        }

        graph.seen_stream_config = "not-called"
        stream_iter = graph.stream("hello")
        assert graph.seen_stream_config is None
        assert next(stream_iter) is None
    finally:
        uninstrument_create_deep_agent()
        _cleanup_instrumentors()


def test_uninstrument_preserves_preinstrumented_dependencies():
    _cleanup_instrumentors()
    langchain_instrumentor = LangChainInstrumentor()
    langgraph_instrumentor = LangGraphInstrumentor()
    instrumentor = DeepAgentsInstrumentor()

    try:
        langchain_instrumentor.instrument()
        langgraph_instrumentor.instrument()

        instrumentor.instrument()
        instrumentor.uninstrument()

        assert langchain_instrumentor.is_instrumented_by_opentelemetry
        assert langgraph_instrumentor.is_instrumented_by_opentelemetry
    finally:
        _cleanup_instrumentors()
