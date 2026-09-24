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
# pyright: ignore
"""
Tests for Mem0 hooks (before/after) plumbing.

These tests validate that:
- Hooks are called for top-level and inner operations
- hook_context is shared between before/after for a single call
- Hook exceptions are swallowed and do not break the wrapped call
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from opentelemetry.instrumentation.mem0.internal import _wrapper as wrapper_mod
from opentelemetry.instrumentation.mem0.internal._wrapper import (
    GraphStoreWrapper,
    MemoryOperationWrapper,
    RerankerWrapper,
    VectorStoreWrapper,
)
from opentelemetry.instrumentation.mem0.types import set_memory_hooks
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.genai.extended_memory import MemoryInvocation


class _DummyTelemetryHandler:
    def start_memory(self, invocation: Any) -> None:
        return None

    def stop_memory(self, invocation: Any) -> None:
        return None

    def fail_memory(self, invocation: Any, error: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_memory_hooks_sync_async_and_exception_paths():
    calls: List[Tuple[str, Dict[str, Any]]] = []
    seen_exc: Dict[str, Any] = {"exc": None}

    def before(span, operation, instance, args, kwargs, hook_context):
        hook_context["start_time"] = 1
        calls.append(("before", hook_context))

    def after(
        span, operation, instance, args, kwargs, hook_context, result, exc
    ):
        # Validate context is shared and exception is surfaced
        assert hook_context.get("start_time") == 1
        if exc is not None:
            seen_exc["exc"] = exc
        calls.append(("after", hook_context))

    w = MemoryOperationWrapper(_DummyTelemetryHandler())
    set_memory_hooks(w, memory_before_hook=before, memory_after_hook=after)

    # Cover helper: _normalize_call_parameters with positional args mapping
    def _sig(self, memory_id, data, *, user_id=None):  # noqa: ARG001
        return None

    normalized = wrapper_mod._normalize_call_parameters(  # type: ignore[attr-defined]
        _sig, args=("mid", "payload"), kwargs={"user_id": "u1"}
    )
    assert normalized["memory_id"] == "mid"
    assert normalized["data"] == "payload"
    assert normalized["user_id"] == "u1"

    # Cover helper: _apply_custom_extractor_output_to_invocation and leftover attribute mapping
    inv = MemoryInvocation(operation="add")
    MemoryOperationWrapper._apply_custom_extractor_output_to_invocation(
        inv,
        {
            "user_id": "u1",
            "limit": "3",
            "threshold": "0.7",
            "rerank": False,
            "server_address": "example.com",
            "server_port": "443",
            "attributes": {"k1": "v1"},
            "custom_k": "custom_v",
        },
    )
    assert inv.user_id == "u1"
    assert inv.limit == 3
    assert inv.threshold == 0.7
    assert inv.rerank is False
    assert inv.server_address == "example.com"
    assert inv.server_port == 443
    assert inv.attributes.get("k1") == "v1"
    assert inv.attributes.get("custom_k") == "custom_v"

    # sync success
    def _fn_sync(*a, **k):
        return "ok"

    res_sync = w._execute_with_handler(
        _fn_sync,
        instance=object(),
        args=(),
        kwargs={"k": "v"},
        operation_name="add",
        extract_attributes_func=None,
        is_memory_client=False,
    )
    assert res_sync == "ok"

    # cover extract_server_info exception branch (host property raises)
    class _BadHost:
        @property
        def host(self):  # pragma: no cover
            raise RuntimeError("boom")

    _ = w._execute_with_handler(
        _fn_sync,
        instance=_BadHost(),
        args=(),
        kwargs={},
        operation_name="get_all",
        extract_attributes_func=None,
        is_memory_client=True,
    )

    # cover custom extractor function path (extract_attributes_func provided)
    inv2 = MemoryInvocation(operation="add")
    w._apply_extracted_attrs_to_invocation(
        inv2,
        instance=object(),
        normalized_kwargs={},
        operation_name="add",
        result={"results": []},
        extract_attributes_func=lambda kwargs, result: {
            "user_id": "u2",
            "attributes": {"x": 1},
        },
        is_memory_client=False,
    )
    assert inv2.user_id == "u2"
    assert inv2.attributes.get("x") == 1

    # async success
    async def _fn_async(*a, **k):
        return "ok2"

    res_async = await w._execute_with_handler_async(
        _fn_async,
        instance=object(),
        args=(),
        kwargs={"k": "v"},
        operation_name="search",
        extract_attributes_func=None,
        is_memory_client=False,
    )
    assert res_async == "ok2"

    # hook exceptions swallowed (both before/after)
    def before_boom(span, operation, instance, args, kwargs, hook_context):
        raise RuntimeError("boom")

    def after_boom(
        span, operation, instance, args, kwargs, hook_context, result, exc
    ):
        raise RuntimeError("boom2")

    set_memory_hooks(
        w, memory_before_hook=before_boom, memory_after_hook=after_boom
    )
    assert (
        w._execute_with_handler(
            _fn_sync,
            instance=object(),
            args=(),
            kwargs={},
            operation_name="get",
            extract_attributes_func=None,
            is_memory_client=False,
        )
        == "ok"
    )

    # exception path calls after_hook with exception
    set_memory_hooks(w, memory_before_hook=before, memory_after_hook=after)

    def _fn_raises(*a, **k):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        w._execute_with_handler(
            _fn_raises,
            instance=object(),
            args=(),
            kwargs={},
            operation_name="delete",
            extract_attributes_func=None,
            is_memory_client=False,
        )

    assert isinstance(seen_exc["exc"], ValueError)
    # For at least the first sync call, context is shared between before/after
    assert calls[0][1] is calls[1][1]


def test_inner_hooks_vector_graph_rerank_and_exception_paths(monkeypatch):
    # Cover both disabled and enabled internal phases paths
    monkeypatch.setattr(
        wrapper_mod, "is_internal_phases_enabled", lambda: False
    )
    tracer = TracerProvider().get_tracer("test")
    vw_disabled = VectorStoreWrapper(tracer)
    # when disabled, wrapper returns original result without span/hook
    assert vw_disabled.wrap_vector_operation("search")(
        lambda *a, **k: {"ok": True}, object(), (), {"limit": 1}
    ) == {"ok": True}

    # Enable internal phases for hook paths below
    monkeypatch.setattr(
        wrapper_mod, "is_internal_phases_enabled", lambda: True
    )

    calls: List[Tuple[str, str, str, Dict[str, Any]]] = []
    seen_exc: Dict[str, Any] = {"exc": None}

    def before(
        span, inner_name, operation, instance, args, kwargs, hook_context
    ):
        hook_context["start_time"] = 1
        calls.append(("before", inner_name, operation, hook_context))

    def after(
        span,
        inner_name,
        operation,
        instance,
        args,
        kwargs,
        hook_context,
        result,
        exc,
    ):
        assert hook_context.get("start_time") == 1
        if exc is not None:
            seen_exc["exc"] = exc
        calls.append(("after", inner_name, operation, hook_context))

    vw = VectorStoreWrapper(
        tracer, inner_before_hook=before, inner_after_hook=after
    )
    gw = GraphStoreWrapper(
        tracer, inner_before_hook=before, inner_after_hook=after
    )
    rw = RerankerWrapper(
        tracer, inner_before_hook=before, inner_after_hook=after
    )

    def wrapped(*a, **k):
        return {"results": [1]}

    # vector success
    res = vw.wrap_vector_operation("search")(
        wrapped, object(), (), {"limit": 1}
    )
    assert res == {"results": [1]}

    # Cover skip branch: mem0migrations collection should bypass span/hook
    class _Migrations:
        collection_name = "mem0migrations"

    calls_before = len(calls)
    assert vw.wrap_vector_operation("search")(
        wrapped, _Migrations(), (), {"limit": 1}
    ) == {"results": [1]}
    assert len(calls) == calls_before  # no hooks recorded

    def graph_wrapped(*a, **k):
        return {"nodes": [1]}

    def rerank_wrapped(*a, **k):
        return [{"id": 1}]

    # graph success
    assert gw.wrap_graph_operation("add")(graph_wrapped, object(), (), {}) == {
        "nodes": [1]
    }
    # rerank success
    assert rw.wrap_rerank()(
        rerank_wrapped, object(), (), {"query": "q", "documents": []}
    ) == [{"id": 1}]

    # after hook exception swallowed (vector)
    def after_boom(
        span,
        inner_name,
        operation,
        instance,
        args,
        kwargs,
        hook_context,
        result,
        exc,
    ):
        raise RuntimeError("boom")

    vw2 = VectorStoreWrapper(
        tracer, inner_before_hook=before, inner_after_hook=after_boom
    )
    assert vw2.wrap_vector_operation("search")(
        wrapped, object(), (), {"limit": 1}
    ) == {"results": [1]}

    # after hook receives exception (vector)
    def wrapped_raises(*a, **k):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        vw.wrap_vector_operation("search")(
            wrapped_raises, object(), (), {"limit": 1}
        )
    assert isinstance(seen_exc["exc"], ValueError)

    # graph exception path
    def graph_raises(*a, **k):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        gw.wrap_graph_operation("add")(graph_raises, object(), (), {})

    # rerank exception path
    def rerank_raises(*a, **k):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        rw.wrap_rerank()(
            rerank_raises, object(), (), {"query": "q", "documents": []}
        )

    # Sanity: at least one before/after pair shares context object (vector success)
    assert calls[0][0] == "before" and calls[1][0] == "after"
    assert calls[0][3] is calls[1][3]
