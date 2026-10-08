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
MemoryClient (managed API) VCR test cases.
Runs in VCR playback mode (cassettes) by default.

To record new cassettes:
- Set environment variable RECORD_MEM0_CLIENT=1
- Provide valid MEM0_API_KEY (optional MEM0_HOST)
- This test uses same user_id, memory_id parameters as test_memoryclient.py
"""

import os
import time

import pytest

# Fixed parameters consistent with test_memoryclient.py
ORG_ID = "org_TYLzC5IhE3P60cQCUl7CIo1dUUgoE8TEeiWvBlNO"
PROJECT_ID = "proj_DlCemE8JjiUbx7odgCJiW3Rx2BuaK8waIkswkmGz"
HARDCODED_MEMORY_ID_1 = "252712fd-ef31-4856-a7f1-1edb8679170c"
HARDCODED_MEMORY_ID_2 = "30bc8557-8609-4738-b77e-4f0b3d6c041c"
SEARCH_USER_ID = "test1"
DELETE_USER_ID = "test-delete1"
E2E_USER_ID = "test_user_client_full"
E2E_AGENT_ID = "test_agent_client_full"
E2E_RUN_ID = "test_run_client_full"


def _memory_spans(spans, operation_name):
    return [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == operation_name
        and span.attributes.get("gen_ai.span.kind") == "MEMORY"
    ]


def _new_client():
    pytest.importorskip("mem0")
    MemoryClient = pytest.importorskip("mem0.client.main").MemoryClient  # type: ignore
    host = os.environ.get("MEM0_HOST", "https://api.mem0.ai")
    # If recording mode, need to provide real API key
    # If playback mode, use fake value (cassettes already have recorded responses)
    is_recording = os.environ.get("RECORD_MEM0_CLIENT") == "1"
    if is_recording:
        api_key = os.environ.get("MEM0_API_KEY")
        if not api_key:
            pytest.skip("Recording mode requires MEM0_API_KEY")
    else:
        # Playback mode uses fake API key (VCR intercepts requests and returns recorded responses)
        api_key = "test_api_key"  # Consistent with cassette
    # Use org_id and project_id consistent with test_memoryclient.py
    return MemoryClient(
        api_key=api_key, host=host, org_id=ORG_ID, project_id=PROJECT_ID
    )


def _unique_user_id(suffix: str) -> str:
    base = os.environ.get("MEM0_TEST_USER_PREFIX", "u_client_rec_")
    return f"{base}{suffix}"


@pytest.mark.vcr(cassette_name="test_client_add_vcr")
def test_client_add_vcr(span_exporter, instrument_with_content):
    """MemoryClient add operation test: verify attributes for adding memory."""
    client = _new_client()

    messages = [
        {
            "role": "user",
            "content": "My name is John and I love reading science fiction books.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you John! I've noted your interest in science fiction.",
        },
    ]
    metadata = {
        "source": "test_memoryclient",
        "scenario": "client_full_1",
        "group": 1,
    }

    add_resp = client.add(
        messages,
        user_id=E2E_USER_ID,
        agent_id=E2E_AGENT_ID,
        run_id=f"{E2E_RUN_ID}_group_1",
        metadata=metadata,
        async_mode=False,
        infer=False,
    )

    assert isinstance(add_resp, dict)

    # Verify add operation span attributes
    spans = span_exporter.get_finished_spans()
    add_spans = _memory_spans(spans, "upsert_memory")
    assert add_spans, "Should have add operation span"

    add_span = add_spans[0]
    assert add_span.name == "upsert_memory"
    assert isinstance(add_span.attributes["gen_ai.memory.record.count"], int)

    # Server attributes
    assert "server.address" in add_span.attributes
    assert "server.port" in add_span.attributes

    if "gen_ai.memory.records" in add_span.attributes:
        assert isinstance(add_span.attributes["gen_ai.memory.records"], str)


@pytest.mark.vcr(cassette_name="test_client_get_all_vcr")
def test_client_get_all_vcr(span_exporter, instrument_with_content):
    """MemoryClient get_all operation test: verify attributes for getting all memories (using test1 user)"""
    client = _new_client()

    # Use test1 user (existing data)
    top_k = 20
    filters = {"AND": [{"user_id": SEARCH_USER_ID}]}
    all_resp = client.get_all(filters=filters, top_k=top_k)

    assert isinstance(all_resp, dict) and "results" in all_resp

    # Verify get_all operation span attributes
    spans = span_exporter.get_finished_spans()
    getall_spans = _memory_spans(spans, "search_memory")
    assert getall_spans, "Should have get_all operation span"

    getall_span = getall_spans[0]
    assert isinstance(
        getall_span.attributes["gen_ai.memory.record.count"], int
    )
    assert "server.address" in getall_span.attributes
    assert "server.port" in getall_span.attributes


@pytest.mark.vcr(cassette_name="test_client_get_vcr")
def test_client_get_vcr(span_exporter, instrument_with_content):
    """MemoryClient get operation test: verify attributes for getting single memory (using hardcoded memory_id)"""
    client = _new_client()

    # Use hardcoded memory_id (existing data)
    get_resp = client.get(HARDCODED_MEMORY_ID_1)

    assert isinstance(get_resp, dict)

    # Verify get operation span attributes
    spans = span_exporter.get_finished_spans()
    get_spans = _memory_spans(spans, "search_memory")
    assert get_spans, "Should have get operation span"

    get_span = get_spans[0]
    assert (
        get_span.attributes["gen_ai.memory.record.id"] == HARDCODED_MEMORY_ID_1
    )
    assert "server.address" in get_span.attributes


@pytest.mark.vcr(cassette_name="test_client_search_vcr")
def test_client_search_vcr(span_exporter, instrument_with_content):
    """MemoryClient search operation test: verify attributes for searching memories (using test1 user, simplified parameters)"""
    client = _new_client()

    # Simplified parameters: only keep query and user_id in filters
    query = "science fiction books"
    filters = {"user_id": SEARCH_USER_ID}
    search_resp = client.search(query=query, filters=filters)

    assert isinstance(search_resp, dict) and "results" in search_resp

    # Verify search operation span attributes
    spans = span_exporter.get_finished_spans()
    search_spans = _memory_spans(spans, "search_memory")
    assert search_spans, "Should have search operation span"

    search_span = search_spans[0]
    assert search_span.attributes["gen_ai.memory.query.text"] == query
    assert isinstance(
        search_span.attributes["gen_ai.memory.record.count"], int
    )

    assert "server.address" in search_span.attributes


@pytest.mark.vcr(cassette_name="test_client_update_vcr")
def test_client_update_vcr(span_exporter, instrument_with_content):
    """MemoryClient update operation test: verify attributes for updating memory (using hardcoded memory_id)"""
    client = _new_client()

    # Use hardcoded memory_id (existing data)
    new_text = "User loves reading science fiction books"
    update_resp = client.update(memory_id=HARDCODED_MEMORY_ID_1, text=new_text)

    assert isinstance(update_resp, dict)

    # Verify update operation span attributes
    spans = span_exporter.get_finished_spans()
    update_spans = _memory_spans(spans, "update_memory")
    assert update_spans, "Should have update operation span"

    update_span = update_spans[0]
    assert (
        update_span.attributes["gen_ai.memory.record.id"]
        == HARDCODED_MEMORY_ID_1
    )
    assert "server.address" in update_span.attributes

    if "gen_ai.memory.records" in update_span.attributes:
        assert new_text in update_span.attributes["gen_ai.memory.records"]


@pytest.mark.vcr(cassette_name="test_client_delete_vcr")
def test_client_delete_vcr(span_exporter, instrument_with_content):
    """MemoryClient delete operation test: verify attributes for deleting memory (using test-delete1 user)"""
    client = _new_client()

    # Use test-delete1 user, get memories for this user first
    filters_delete = {"AND": [{"user_id": DELETE_USER_ID}]}
    all_for_delete = client.get_all(filters=filters_delete, top_k=10)
    delete_memories = all_for_delete.get("results", [])

    if not delete_memories:
        pytest.skip(f"No memories found for user_id='{DELETE_USER_ID}'")

    mem_id = delete_memories[0]["id"]

    # Clear span exporter
    span_exporter.clear()

    # delete operation
    delete_resp = client.delete(mem_id)

    assert isinstance(delete_resp, dict)

    # Verify delete operation span attributes
    spans = span_exporter.get_finished_spans()
    delete_spans = _memory_spans(spans, "delete_memory")
    assert delete_spans, "Should have delete operation span"

    delete_span = delete_spans[0]
    assert delete_span.attributes["gen_ai.memory.record.id"] == mem_id
    assert "server.address" in delete_span.attributes


@pytest.mark.vcr(cassette_name="test_client_history_vcr")
def test_client_history_vcr(span_exporter, instrument_with_content):
    """MemoryClient history operation test: verify attributes for getting memory history (using hardcoded memory_id)"""
    client = _new_client()

    # Use hardcoded memory_id (existing data that has been updated, has history)
    history_resp = client.history(HARDCODED_MEMORY_ID_1)

    assert isinstance(history_resp, list)

    # Verify history operation span attributes
    spans = span_exporter.get_finished_spans()
    history_spans = _memory_spans(spans, "search_memory")
    assert history_spans, "Should have history operation span"

    history_span = history_spans[0]
    assert (
        history_span.attributes["gen_ai.memory.record.id"]
        == HARDCODED_MEMORY_ID_1
    )
    assert "server.address" in history_span.attributes


@pytest.mark.vcr(cassette_name="test_client_batch_update_vcr")
def test_client_batch_update_vcr(span_exporter, instrument_with_content):
    """MemoryClient batch_update operation test (using hardcoded memory_id)"""
    client = _new_client()

    # Use hardcoded memory_id for batch update
    batch_updates = [
        {
            "memory_id": HARDCODED_MEMORY_ID_1,
            "text": "User likes reading science fiction novels, mystery novels, and romance novels",
            "metadata": {"updated": True, "batch": True},
        },
        {
            "memory_id": HARDCODED_MEMORY_ID_2,
            "text": "User name is John",
            "metadata": {"updated": True, "batch": True},
        },
    ]

    upd_resp = client.batch_update(memories=batch_updates)
    assert isinstance(upd_resp, dict)


@pytest.mark.vcr(cassette_name="test_client_batch_delete_vcr")
def test_client_batch_delete_vcr(span_exporter, instrument_with_content):
    """MemoryClient batch_delete operation test (using test-delete1 user)"""
    client = _new_client()

    # Use test-delete1 user, get memories for batch deletion
    filters_delete = {"AND": [{"user_id": DELETE_USER_ID}]}
    all_for_delete = client.get_all(filters=filters_delete, top_k=10)
    delete_memories = all_for_delete.get("results", [])

    if len(delete_memories) < 2:
        pytest.skip(
            f"Not enough memories for user_id='{DELETE_USER_ID}' (need at least 2)"
        )

    # Take first two for batch deletion
    memory_id_2 = delete_memories[1]["id"]
    memory_id_3 = (
        delete_memories[2]["id"]
        if len(delete_memories) >= 3
        else delete_memories[1]["id"]
    )
    batch_deletes = [{"memory_id": memory_id_2}, {"memory_id": memory_id_3}]

    del_resp = client.batch_delete(batch_deletes)
    assert isinstance(del_resp, dict)


@pytest.mark.vcr(cassette_name="test_client_delete_all_vcr")
def test_client_delete_all_vcr(span_exporter, instrument_with_content):
    """MemoryClient delete_all operation test (using E2E user)"""
    client = _new_client()

    # First add some sample data to E2E user
    client.add(
        [
            {"role": "user", "content": "用户C 喜欢阅读"},
            {"role": "assistant", "content": "记录：喜欢阅读"},
        ],
        user_id=E2E_USER_ID,
        agent_id=E2E_AGENT_ID,
        async_mode=False,
        output_format="v1.1",
    )
    time.sleep(0.2)

    # Delete all data for E2E user
    del_all = client.delete_all(user_id=E2E_USER_ID, agent_id=E2E_AGENT_ID)
    assert isinstance(del_all, dict)
