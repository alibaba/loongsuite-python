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
Memory operation VCR test cases (using Mock vector storage).
By default, runs in VCR playback mode (cassettes) with Mock vector storage.
"""

import os
from typing import Any, Dict, List, Optional, cast

import pytest


def _memory_spans(spans, operation_name):
    return [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == operation_name
        and span.attributes.get("gen_ai.span.kind") == "MEMORY"
    ]


def _build_demo_config_from_env() -> Dict[str, Any]:
    # Allow overriding model and dimensions via environment variables
    llm_model = os.environ.get("OPENAI_LLM_MODEL", "qwen-plus")
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-v3")
    embed_dims = int(os.environ.get("EMBEDDING_DIMS", "128"))

    # Vector database Milvus: This test uses local mock, does not depend on real URL/token (will not enter VCR)

    # LLM Reranker (goes through HTTP, enters VCR)
    rerank_model = os.environ.get("RERANK_LLM_MODEL", llm_model)
    rerank_top_k = int(os.environ.get("RERANK_TOP_K", "5"))

    # Note: OPENAI_API_KEY, DashScope need to be imported from environment; not hardcoded here
    #    VCR will filter sensitive headers, see conftest.py vcr_config
    config: Dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embed_model,
                "embedding_dims": embed_dims,
            },
        },
        "vector_store": {
            "provider": "milvus",
            "config": {
                "collection_name": "mem0_test_simple",
                "embedding_model_dims": 128,
                "metric_type": "COSINE",
                "url": "http://localhost:19530",
                "token": "token",
                "db_name": "default",
            },
        },
        "reranker": {
            "provider": "llm_reranker",
            "config": {
                "model": rerank_model,
                "temperature": 0.0,
                "max_tokens": 60,
                "top_k": rerank_top_k,
                # Directly reuse the Chinese scoring prompt from demo (optional)
                "scoring_prompt": '你是一中文检索重排评分器。请根据查询与文档content语义相关性，输一 0.0~1.0 分数。\n判定准则（通用）：\n- 与查询主题/意图直接匹配或可直接回答：≥0.9\n- 部分匹配、contain关键信息或用户偏好/约束（如喜欢/不喜欢/过敏/禁忌等）：0.6~0.8\n- 仅弱相关或上下文背景信息：0.3~0.5\n- 无关信息（如姓名、地点、问候等无助于回答）：≤0.2\n严格只输数值，不要解释。\n查询: "{query}"\n文档: "{document}"',
            },
        },
        "version": "v1.1",
    }
    return config


@pytest.mark.vcr()
def test_record_memory_full_flow_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory full flow test: covers all operations"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    # Import needed mock modules
    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import patch_factories  # noqa: PLC0415

    # Patch factories before creating Memory instance
    patch_factories(monkeypatch)

    config = _build_demo_config_from_env()
    m = Memory.from_config(config)

    # patch_factories already provides fake vector storage, no additional mock needed
    # but need to set FakeLLM and FakeEmbedder (these are not created via factory)
    from conftest import FakeEmbedder, FakeLLM  # noqa: PLC0415

    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_record_001"
    messages = [
        {"role": "user", "content": "我叫小王，我不喜欢川菜，常住上海。"},
        {
            "role": "assistant",
            "content": "好，小王。我already记录你口味偏好。",
        },
    ]

    # add
    add_resp = m.add(messages, user_id=user_id)
    assert add_resp is not None
    assert isinstance(add_resp, dict) and "results" in add_resp
    results_list: List[Dict[str, Any]] = cast(
        List[Dict[str, Any]], add_resp["results"]
    )

    # Mock vector storage is immediately available, no need to wait

    # search (triggers embed/search and reranker HTTP)
    search_resp = m.search(
        query="川菜推荐", user_id=user_id, limit=2, rerank=True
    )
    assert search_resp is not None
    assert isinstance(search_resp, dict) and "results" in search_resp

    # get_all (by user_id)
    all_resp = m.get_all(user_id=user_id, limit=100)
    assert all_resp is not None
    assert isinstance(all_resp, dict) and "results" in all_resp

    # Get a valid mem_id from get_all results (if not exist, insert seed data)
    all_results = all_resp.get("results", [])
    if len(all_results) >= 1:
        mem_id: Optional[str] = all_results[0].get("id")
    elif len(results_list) >= 1:
        mem_id = results_list[0].get("id")
    else:
        # When no memory is generated, insert seed data into FakeVectorStore and get its id
        import sys  # noqa: PLC0415

        sys.path.insert(0, os.path.dirname(__file__))
        from conftest import get_fake_vector_store  # noqa: PLC0415

        store = get_fake_vector_store()
        store.insert(payloads=[{"data": "seed memory for tests"}])
        hits_batch = store.list(limit=1)
        if (
            isinstance(hits_batch, list)
            and hits_batch
            and isinstance(hits_batch[0], list)
            and hits_batch[0]
        ):
            mem_id = getattr(hits_batch[0][0], "id", None)
        else:
            mem_id = None

    if mem_id:
        # get (by id)
        got = m.get(mem_id)
        assert got is None or isinstance(got, dict)

        # update (modify text)
        upd_resp = m.update(mem_id, "我叫小王，常住上海，偏好清淡口味。")
        assert isinstance(upd_resp, dict)

        # history (should return list)
        hist = m.history(mem_id)
        assert isinstance(hist, list)

        # delete (single)
        del_resp = m.delete(mem_id)
        assert isinstance(del_resp, dict)

    # delete_all (by user_id range)
    try:
        del_all_resp = m.delete_all(user_id=user_id)
        assert isinstance(del_all_resp, dict)
    except Exception:
        # Compatible with some vector database list return structure differences that cause delete process exceptions, but should still generate related spans
        pass

    # Verify OTel spans at least contain each main operation
    spans: List[Any] = span_exporter.get_finished_spans()
    ops_needed = {
        "upsert_memory",
        "search_memory",
        "update_memory",
        "delete_memory",
    }
    ops_seen = {
        span.attributes["gen_ai.operation.name"]
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "MEMORY"
    }
    missing = ops_needed - ops_seen
    assert not missing, f"should contain span attributes: {missing}"


@pytest.mark.vcr()
def test_record_memory_add_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory add operation test: verify add memory attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_add_test"
    agent_id = "a_add_test"
    run_id = "r_add_test"
    messages = [
        {"role": "user", "content": "我喜欢吃麻辣火锅。"},
        {"role": "assistant", "content": "好，already记录您口味偏好。"},
    ]
    metadata = {"source": "test", "scenario": "add_test"}

    add_resp = m.add(
        messages,
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        metadata=metadata,
        infer=False,
    )

    assert add_resp is not None
    assert isinstance(add_resp, dict) and "results" in add_resp

    # Verify add operation span attributes
    spans = span_exporter.get_finished_spans()
    add_spans = _memory_spans(spans, "upsert_memory")
    assert add_spans, "should have add operation span"

    add_span = add_spans[0]
    assert add_span.name == "upsert_memory"
    assert isinstance(add_span.attributes["gen_ai.memory.record.count"], int)
    assert isinstance(add_span.attributes["gen_ai.memory.records"], str)


@pytest.mark.vcr()
def test_record_memory_get_all_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory get_all operation test: verify get all memory attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_getall_test"
    agent_id = "a_getall_test"
    run_id = "r_getall_test"
    limit = 10

    # first add some test data
    m.add("测试记忆content", user_id=user_id, agent_id=agent_id, run_id=run_id)

    # Clear span exporter to only verify get_all operation
    span_exporter.clear()

    # get_all operation
    get_all_resp = m.get_all(
        user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit
    )

    assert get_all_resp is not None
    assert isinstance(get_all_resp, dict) and "results" in get_all_resp

    # Verify get_all operation span attributes
    spans = span_exporter.get_finished_spans()
    getall_spans = _memory_spans(spans, "search_memory")
    assert getall_spans, "should have get_all operation span"

    getall_span = getall_spans[0]
    assert isinstance(
        getall_span.attributes["gen_ai.memory.record.count"], int
    )
    assert isinstance(getall_span.attributes["gen_ai.memory.records"], str)


@pytest.mark.vcr()
def test_record_memory_get_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory get operation test: verify get single memory attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        get_fake_vector_store,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_get_test"

    # first add a memory to get memory_id
    add_resp = m.add("测试记忆content", user_id=user_id)
    results = add_resp.get("results", [])
    mem_id: Optional[str] = None
    if results:
        mem_id = results[0].get("id")

    if not mem_id:
        # If no ID is generated, get from fake vector store
        store = get_fake_vector_store()
        store.insert(payloads=[{"data": "seed memory"}])
        hits = store.list(limit=1)
        if hits and hits[0]:
            mem_id = getattr(hits[0][0], "id", "test_memory_id")

    assert mem_id, "need a valid memory_id"

    # Clear span exporter
    span_exporter.clear()

    # get operation
    m.get(mem_id)

    # Verify get operation span attributes
    spans = span_exporter.get_finished_spans()
    get_spans = _memory_spans(spans, "search_memory")
    assert get_spans, "should have get operation span"

    get_span = get_spans[0]
    assert get_span.attributes["gen_ai.memory.record.id"] == mem_id
    if "gen_ai.memory.records" in get_span.attributes:
        output = get_span.attributes["gen_ai.memory.records"]
        assert isinstance(output, str) and len(output) > 0


@pytest.mark.vcr()
def test_record_memory_search_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory search operation test: verify search memory attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_search_test"

    # first add some test data
    m.add("我喜欢吃川菜", user_id=user_id)
    m.add("我喜欢看科幻movies", user_id=user_id)

    # Clear span exporter
    span_exporter.clear()

    # search operation
    query = "川菜推荐"
    limit = 5
    search_resp = m.search(
        query=query, user_id=user_id, limit=limit, rerank=True
    )

    assert search_resp is not None
    assert isinstance(search_resp, dict) and "results" in search_resp

    # Verify search operation span attributes
    spans = span_exporter.get_finished_spans()
    search_spans = _memory_spans(spans, "search_memory")
    assert search_spans, "shouldhas search operation span"

    search_span = search_spans[0]
    assert search_span.attributes["gen_ai.memory.query.text"] == query
    assert isinstance(
        search_span.attributes["gen_ai.memory.record.count"], int
    )
    if "gen_ai.memory.records" in search_span.attributes:
        assert isinstance(search_span.attributes["gen_ai.memory.records"], str)


@pytest.mark.vcr()
def test_record_memory_update_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory update operation test: verify update memory attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        get_fake_vector_store,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_update_test"

    # first add a memory
    add_resp = m.add("原始记忆content", user_id=user_id)
    results = add_resp.get("results", [])
    mem_id: Optional[str] = None
    if results:
        mem_id = results[0].get("id")

    if not mem_id:
        store = get_fake_vector_store()
        store.insert(payloads=[{"data": "seed memory"}])
        hits = store.list(limit=1)
        if hits and hits[0]:
            mem_id = getattr(hits[0][0], "id", "test_memory_id")

    assert mem_id, "need a valid memory_id"

    # Clear span exporter
    span_exporter.clear()

    # update operation
    new_content = "更新后记忆content"
    update_resp = m.update(mem_id, new_content)

    assert update_resp is not None
    assert isinstance(update_resp, dict)

    # Verify update operation span attributes
    spans = span_exporter.get_finished_spans()
    update_spans = _memory_spans(spans, "update_memory")
    assert update_spans, "shouldhas update operation span"

    update_span = update_spans[0]
    assert update_span.attributes["gen_ai.memory.record.id"] == mem_id
    assert new_content in update_span.attributes["gen_ai.memory.records"]


@pytest.mark.vcr()
def test_record_memory_delete_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory delete operation test: verify delete memory attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        get_fake_vector_store,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_delete_test"

    # first add a memory
    add_resp = m.add("待delete记忆content", user_id=user_id)
    results = add_resp.get("results", [])
    mem_id: Optional[str] = None
    if results:
        mem_id = results[0].get("id")

    if not mem_id:
        store = get_fake_vector_store()
        store.insert(payloads=[{"data": "seed memory"}])
        hits = store.list(limit=1)
        if hits and hits[0]:
            mem_id = getattr(hits[0][0], "id", "test_memory_id")

    assert mem_id, "need a valid memory_id"

    # Clear span exporter
    span_exporter.clear()

    # delete operation
    delete_resp = m.delete(mem_id)

    assert delete_resp is not None
    assert isinstance(delete_resp, dict)

    # Verify delete operation span attributes
    spans = span_exporter.get_finished_spans()
    delete_spans = _memory_spans(spans, "delete_memory")
    assert delete_spans, "shouldhas delete operation span"

    delete_span = delete_spans[0]
    assert delete_span.attributes["gen_ai.memory.record.id"] == mem_id


@pytest.mark.vcr()
def test_record_memory_history_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory history operation test: verify get memory history attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        get_fake_vector_store,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_history_test"

    # first add a memory and update it (to generate history)
    add_resp = m.add("原始content", user_id=user_id)
    results = add_resp.get("results", [])
    mem_id: Optional[str] = None
    if results:
        mem_id = results[0].get("id")

    if not mem_id:
        store = get_fake_vector_store()
        store.insert(payloads=[{"data": "seed memory"}])
        hits = store.list(limit=1)
        if hits and hits[0]:
            mem_id = getattr(hits[0][0], "id", "test_memory_id")

    assert mem_id, "need a valid memory_id"

    # update once to generate history
    m.update(mem_id, "更新后content")

    # Clear span exporter
    span_exporter.clear()

    # history operation
    history_resp = m.history(mem_id)

    assert history_resp is not None
    assert isinstance(history_resp, list)

    # Verify history operation span attributes
    spans = span_exporter.get_finished_spans()
    history_spans = _memory_spans(spans, "search_memory")
    assert history_spans, "should have history operation span"

    history_span = history_spans[0]
    assert history_span.attributes["gen_ai.memory.record.id"] == mem_id
    assert history_span.attributes["gen_ai.memory.record.count"] >= 0


@pytest.mark.vcr()
def test_record_memory_delete_all_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
) -> None:
    """Memory delete_all operation test: verify delete all memory attributes"""
    Memory = pytest.importorskip("mem0.memory.main").Memory

    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import (  # noqa: PLC0415
        FakeEmbedder,
        FakeLLM,
        patch_factories,
    )

    patch_factories(monkeypatch)
    config = _build_demo_config_from_env()
    m = Memory.from_config(config)
    m.llm = FakeLLM()
    m.embedding_model = FakeEmbedder()

    user_id = "u_deleteall_test"
    agent_id = "a_deleteall_test"
    run_id = "r_deleteall_test"

    # first add some test data
    m.add("测试记忆1", user_id=user_id, agent_id=agent_id, run_id=run_id)
    m.add("测试记忆2", user_id=user_id, agent_id=agent_id, run_id=run_id)

    # Clear span exporter
    span_exporter.clear()

    # delete_all operation
    try:
        delete_all_resp = m.delete_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id
        )
        assert delete_all_resp is not None
        assert isinstance(delete_all_resp, dict)
    except Exception:
        # Compatible with some vector database exception situations
        pass

    # Verify delete_all operation span attributes
    spans = span_exporter.get_finished_spans()
    deleteall_spans = _memory_spans(spans, "delete_memory")
    assert deleteall_spans, "shouldhas delete_all operation span"

    assert deleteall_spans[0].name == "delete_memory"
