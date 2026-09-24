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

# -*- coding: utf-8 -*-
"""
pytest + VCR Mem0 integration test (using factory monkeypatch to cover all stages).
In environment without external dependencies, via monkeypatch factory products, avoid real network/storage dependencies,
while allowing Mem0 instrumentation to take effect on internal stages (vector/reranker/graph etc).
"""

from typing import Any, Dict, List, cast

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

_mem0 = pytest.importorskip("mem0")
_factory = pytest.importorskip("mem0.utils.factory")


class _FakeVectorStore:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def search(self, **kwargs: Dict[str, Any]) -> List[Any]:
        # return empty/or fixed matches
        class _Hit:
            def __init__(self, _id: str, _data: str) -> None:
                self.id = _id
                self.score = 0.5
                self.payload = {
                    "data": _data,
                    "hash": None,
                    "created_at": None,
                    "updated_at": None,
                }

        results: List[Any] = []
        for k, v in self._items.items():
            results.append(_Hit(k, v.get("data", "")))
        limit_value: Any = kwargs.get("limit", 5)
        try:
            limit = int(limit_value) if limit_value is not None else 5
        except Exception:
            limit = 5
        return results[:limit]

    def insert(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payloads_value: Any = kwargs.get("payloads")
        payloads: List[Dict[str, Any]] = []
        if isinstance(payloads_value, list):
            for item in cast(List[Any], payloads_value):
                if isinstance(item, dict):
                    typed_item: Dict[str, Any] = cast(Dict[str, Any], item)
                    payloads.append(typed_item)
        for idx, p in enumerate(payloads):
            self._items[str(len(self._items) + idx)] = {"data": p.get("data")}
        return {"ok": True}

    # Reserve other interfaces to avoid AttributeError
    def update(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def delete(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def list(self, **kwargs: Dict[str, Any]) -> List[List[Any]]:
        """return double layer list structure to match certain vector storage implementations"""

        class _Hit:
            def __init__(self, _id: str, _data: str) -> None:
                self.id = _id
                self.score = 0.5
                self.payload = {
                    "data": _data,
                    "hash": None,
                    "created_at": None,
                    "updated_at": None,
                }

        hits: List[Any] = []
        for k, v in self._items.items():
            hits.append(_Hit(k, v.get("data", "")))
        # return double layer list [[...]] to match mem0 expectation
        return [hits]

    def get(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"item": None}

    def reset(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._items.clear()
        return {"ok": True}


class _FakeReranker:
    def rerank(
        self, **kwargs: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        docs_value: Any = kwargs.get("docs")
        docs: List[Any] = []
        if isinstance(docs_value, list):
            docs = cast(List[Any], docs_value)
        out: List[Dict[str, Any]] = []
        for i, d in enumerate(docs):
            out.append(
                {
                    "id": getattr(d, "id", str(i)),
                    "rerank_score": 0.2,
                    "text": getattr(d, "text", str(d)),
                }
            )
        top_k_value: Any = kwargs.get("top_k", len(out))
        try:
            top_k = int(top_k_value) if top_k_value is not None else len(out)
        except Exception:
            top_k = len(out)
        return {"results": out[:top_k]}


class _FakeGraphStore:
    def add(self, **kwargs: Dict[str, Any]) -> Dict[str, List[int]]:
        return {"nodes": [1]}

    def get_all(self, **kwargs: Dict[str, Any]) -> Dict[str, List[int]]:
        return {"nodes": []}

    def search(self, **kwargs: Dict[str, Any]) -> Dict[str, List[int]]:
        return {"nodes": []}

    def delete_all(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def reset(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}


class _FakeEmbedder:
    def embed(self, *args: Any, **kwargs: Dict[str, Any]) -> List[float]:
        # return fixed length vector (consistent with downstream expectation)
        return [0.0] * 128

    class config:  # provide telemetry read configuration
        embedding_dims = 128


class _FakeLLM:
    def generate_response(self, **kwargs: Dict[str, Any]) -> str:
        # return simple string, avoid external logic assuming dict.strip causing errors
        return "ok"


def _patch_factories(monkeypatch: Any) -> None:
    # General patch attempt: try to override create by module path list
    def try_patch(module_path: str, factory_name: str, create_fn: Any) -> None:
        try:
            mod = __import__(module_path, fromlist=[factory_name])
            fac = getattr(mod, factory_name, None)
            if fac and hasattr(fac, "create"):
                monkeypatch.setattr(fac, "create", create_fn)
        except Exception:
            # ignore exception
            pass

    # VectorStore
    def vs_create(*a: Any, **k: Any) -> _FakeVectorStore:
        return _FakeVectorStore()

    try_patch("mem0.utils.factory", "VectorStoreFactory", vs_create)
    try_patch("mem0.vector_stores.factory", "VectorStoreFactory", vs_create)

    # Reranker
    def rr_create(*a: Any, **k: Any) -> _FakeReranker:
        return _FakeReranker()

    try_patch("mem0.utils.factory", "RerankerFactory", rr_create)
    try_patch("mem0.rerank.factory", "RerankerFactory", rr_create)

    # GraphStore
    def gs_create(*a: Any, **k: Any) -> _FakeGraphStore:
        return _FakeGraphStore()

    try_patch("mem0.utils.factory", "GraphStoreFactory", gs_create)
    try_patch("mem0.graph.factory", "GraphStoreFactory", gs_create)

    # Embedder
    def emb_create(*a: Any, **k: Any) -> _FakeEmbedder:
        return _FakeEmbedder()

    try_patch("mem0.utils.factory", "EmbedderFactory", emb_create)
    try_patch("mem0.embeddings.factory", "EmbedderFactory", emb_create)

    # LLM
    def llm_create(*a: Any, **k: Any) -> _FakeLLM:
        return _FakeLLM()

    try_patch("mem0.utils.factory", "LLMFactory", llm_create)
    try_patch("mem0.llms.factory", "LLMFactory", llm_create)


@pytest.mark.vcr()
def test_memory_add_full_flow(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    _patch_factories(monkeypatch)

    from mem0.memory.main import Memory  # noqa: PLC0415

    m = cast(Any, Memory())
    # Override instance internal dependencies to avoid real external calls
    m.llm = _FakeLLM()
    m.embedding_model = _FakeEmbedder()
    m.vector_store = _FakeVectorStore()
    m.graph = _FakeGraphStore()
    test_message = "我叫小王，我不喜欢川菜，常住上海。"
    assert m.add(test_message, user_id="u_123") is not None

    spans = span_exporter.get_finished_spans()
    # verify top-level operation type via attributes, not relying on span.name
    add_span = next(
        s
        for s in spans
        if s.attributes.get("gen_ai.memory.operation") == "add"
    )
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in add_span.attributes

    # Verify input messages collection (messages passed as positional parameter)
    assert "gen_ai.memory.input.messages" in add_span.attributes
    assert test_message in add_span.attributes["gen_ai.memory.input.messages"]

    # Verify output messages collection (if has results)
    if (
        "gen_ai.memory.result.count" in add_span.attributes
        and add_span.attributes["gen_ai.memory.result.count"] > 0
    ):
        # If has results, should have output messages
        assert "gen_ai.memory.output.messages" in add_span.attributes


@pytest.mark.vcr()
def test_memory_search_full_flow(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    _patch_factories(monkeypatch)

    from mem0.memory.main import Memory  # noqa: PLC0415

    m = cast(Any, Memory())
    # Override instance internal dependencies
    m.llm = _FakeLLM()
    m.embedding_model = _FakeEmbedder()
    m.vector_store = _FakeVectorStore()
    m.graph = _FakeGraphStore()
    try:
        setattr(m, "reranker", _FakeReranker())
    except Exception:
        # ignore exception
        pass
    test_query = "川菜推荐"
    assert m.search(test_query, user_id="u_123", limit=2) is not None

    spans = span_exporter.get_finished_spans()
    # verify top-level operation type via attributes
    search_span = next(
        s
        for s in spans
        if s.attributes.get("gen_ai.memory.operation") == "search"
    )

    # Verify input messages collection (query passed as positional parameter)
    assert "gen_ai.memory.input.messages" in search_span.attributes
    assert test_query in search_span.attributes["gen_ai.memory.input.messages"]

    # Verify output messages collection (if has results)
    # FakeVectorStore search will return results, so should have output messages
    if "gen_ai.memory.result.count" in search_span.attributes:
        # As long as executed, regardless of whether results are empty, may have output messages
        # here not mandatory, because results may be empty
        pass


@pytest.mark.vcr()
def test_internal_subphases_attributes(
    span_exporter: Any,
    instrument_with_factories_patched: Any,
    monkeypatch: Any,
) -> None:
    """
    Cover internal subphases: whether vector, graph, reranking key attributes are captured.
    Use fake implementations to avoid real dependencies; only verify attribute existence and basic value ranges.

    Note: LLM and Embedding do not need to generate spans (handled by corresponding LLM plugins)
    """
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from mem0.configs.base import MemoryConfig, RerankerConfig  # noqa: PLC0415
    from mem0.memory.main import Memory  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import FakeEmbedder, FakeLLM  # noqa: PLC0415

    # Configure reranker so Memory creates via factory (thus will be instrumented)
    config = MemoryConfig()
    # Set reranker config, Memory will create via RerankerFactory.create
    # Using valid provider (cohere), but will be intercepted by patch_factories to return FakeReranker
    config.reranker = RerankerConfig(provider="cohere", config={})
    m = cast(Any, Memory(config=config))

    # Only manually replace LLM and Embedder (these do not need to generate mem0 subphase spans)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    m.vector_store.reset()
    m.vector_store.insert(
        payloads=[
            {"data": "我喜欢川菜和火锅。"},
            {"data": "我喜欢清淡饮食。"},
            {"data": "我常住上海。"},
        ]
    )

    # Search across multiple seeded candidates so reranking deterministically runs.
    assert (
        m.search("川菜推荐", user_id="u_abc", limit=2, rerank=True) is not None
    )

    spans = span_exporter.get_finished_spans()

    # 1. Vector stage: must generate span
    vec_spans = [
        s
        for s in spans
        if s.attributes.get("gen_ai.memory.vector.method") is not None
    ]
    assert vec_spans, "should collect vector stage spans"
    valid_vec_methods = {
        "insert",
        "search",
        "update",
        "get",
        "list",
        "delete",
        "reset",
        "list_cols",
        "create_col",
    }
    assert any(
        s.attributes.get("gen_ai.memory.vector.method") in valid_vec_methods
        for s in vec_spans
    ), (
        f"Vector method should be one of valid values, actual: {[s.attributes.get('gen_ai.memory.vector.method') for s in vec_spans]}"
    )
    # Verify provider (now uses data_source.type)
    assert any(
        s.attributes.get("gen_ai.memory.data_source.type") is not None
        for s in vec_spans
    ), "Vector span should contain data_source.type"

    # 2. Reranker stage: if reranker created via factory, should generate span
    reranker_spans = [
        s
        for s in spans
        if s.attributes.get("gen_ai.provider.name") is not None
    ]
    if m.reranker is not None:
        assert reranker_spans, (
            "When reranker is enabled, should collect reranker stage spans"
        )
        # Verify attributes
        assert any(
            s.attributes.get("gen_ai.provider.name") is not None
            for s in reranker_spans
        ), "Reranker span should contain provider.name"

    # 3. Graph stage: if graph is enabled, should generate span
    graph_spans = [
        s
        for s in spans
        if s.attributes.get("gen_ai.memory.graph.method") is not None
    ]
    if m.enable_graph:
        assert graph_spans, (
            "When graph is enabled, should collect graph stage spans"
        )

    # 4. LLM and Embedding: do not require generating mem0 subphase spans
    # These should be handled by corresponding LLM/Embedding plugins


@pytest.mark.vcr()
def test_vector_operations_detailed_attributes(
    span_exporter: Any, instrument_with_factories_patched: Any
) -> None:
    """
    verify whether Vector operation detailed attributes are correctly collected.
    Including: provider, method, limit, filters.keys, result_count, etc.
    """
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from mem0.memory.main import Memory  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import FakeEmbedder, FakeLLM  # noqa: PLC0415

    m = cast(Any, Memory())
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    # Trigger search operation with filters
    m.add("测试data", user_id="u_vec_test")
    m.search("测试", user_id="u_vec_test", agent_id="a_vec_test", limit=5)

    spans = span_exporter.get_finished_spans()
    vec_spans = [
        s for s in spans if "gen_ai.memory.vector.method" in s.attributes
    ]

    assert vec_spans, "should contain to vector span"

    # Verify required attributes (now uses data_source.type)
    assert any(
        s.attributes.get("gen_ai.memory.data_source.type") == "fake"
        for s in vec_spans
    ), "should contain data_source.type"
    assert any(
        s.attributes.get("gen_ai.memory.vector.method")
        in {"insert", "search", "get", "list_cols", "create_col"}
        for s in vec_spans
    ), "should contain method"

    # Verify recommended attributes
    search_spans = [
        s
        for s in vec_spans
        if s.attributes.get("gen_ai.memory.vector.method") == "search"
    ]
    if search_spans:
        search_span = search_spans[0]
        # Verify limit
        assert (
            "gen_ai.memory.vector.limit" in search_span.attributes
            or "gen_ai.memory.vector.k" in search_span.attributes
        ), "search span should contain limit or k"
        # Verify filter_keys (note: actual attribute name is filter_keys, not filters.keys)
        assert "gen_ai.memory.vector.filter_keys" in search_span.attributes, (
            "search span should contain filter_keys"
        )
        filter_keys = search_span.attributes.get(
            "gen_ai.memory.vector.filter_keys"
        )
        assert "user_id" in filter_keys, "filter_keys should contain user_id"
        # Verify result_count
        assert "gen_ai.memory.vector.result_count" in search_span.attributes, (
            "search span should contain result_count"
        )


@pytest.mark.vcr()
def test_graph_operations_detailed_attributes(
    span_exporter: Any, instrument_with_factories_patched: Any
) -> None:
    """
    verify whether Graph operation detailed attributes are correctly collected.
    Including: provider, method, result_count, etc.
    """
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from mem0.configs.base import (  # noqa: PLC0415
        GraphStoreConfig,
        MemoryConfig,
    )
    from mem0.memory.main import Memory  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import FakeEmbedder, FakeLLM  # noqa: PLC0415

    # Enable graph (using kuzu as provider, but will be intercepted by patch_factories to return FakeGraphStore)
    config = MemoryConfig()
    # Kuzu supports in-memory database
    config.graph_store = GraphStoreConfig(
        provider="kuzu", config={"path": ":memory:"}
    )
    m = cast(Any, Memory(config=config))
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    # Trigger graph operations
    m.add("小明喜欢Python编程", user_id="u_graph_test")

    spans = span_exporter.get_finished_spans()
    graph_spans = [
        s for s in spans if "gen_ai.memory.graph.method" in s.attributes
    ]

    assert graph_spans, "should containto graph span"

    # Verify required attributes (now uses data_source.type)
    assert any(
        s.attributes.get("gen_ai.memory.data_source.type") == "fake"
        for s in graph_spans
    ), "should contain data_source.type"
    assert any(
        s.attributes.get("gen_ai.memory.graph.method")
        in {"add", "search", "get_all", "delete_all"}
        for s in graph_spans
    ), "should contain method"

    # Verify recommended attributes
    add_spans = [
        s
        for s in graph_spans
        if s.attributes.get("gen_ai.memory.graph.method") == "add"
    ]
    if add_spans:
        add_span = add_spans[0]
        # Verify result_count
        result_count = add_span.attributes.get(
            "gen_ai.memory.graph.result_count"
        )
        assert result_count is not None and result_count >= 0, (
            f"add span should contain result_count, actual value: {result_count}"
        )


@pytest.mark.vcr()
def test_reranker_operations_detailed_attributes(
    span_exporter: Any, instrument_with_factories_patched: Any
) -> None:
    """
    verify whether Reranker operation detailed attributes are correctly collected.
    Including: provider, top_k, input_count, etc.
    """
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from mem0.configs.base import MemoryConfig, RerankerConfig  # noqa: PLC0415
    from mem0.memory.main import Memory  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import FakeEmbedder, FakeLLM  # noqa: PLC0415

    config = MemoryConfig()
    config.reranker = RerankerConfig(provider="cohere", config={})
    m = cast(Any, Memory(config=config))
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    m.vector_store.reset()
    m.vector_store.insert(
        payloads=[
            {"data": "测试data1"},
            {"data": "测试data2"},
            {"data": "测试data3"},
        ]
    )

    m.search("测试", user_id="u_rerank_test", limit=2, rerank=True)

    spans = span_exporter.get_finished_spans()
    # Now uses gen_ai.provider.name instead of gen_ai.memory.reranker.provider
    reranker_spans = [
        s for s in spans if "gen_ai.provider.name" in s.attributes
    ]

    assert reranker_spans, (
        "expected mem0 reranker instrumentation to emit at least one reranker "
        "span, but none were found"
    )

    reranker_span = reranker_spans[0]
    assert reranker_span.attributes.get("gen_ai.provider.name") == "fake", (
        "should contain correct provider.name"
    )
    documents_count = reranker_span.attributes.get(
        "gen_ai.rerank.documents_count"
    )
    assert documents_count == 2, (
        f"documents_count should match seeded candidate count, got {documents_count}"
    )
    top_k = reranker_span.attributes.get("gen_ai.request.top_k")
    assert top_k == 2, f"top_k should reflect the search limit, got {top_k}"
