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

"""Tests for the LLM async-compat wrapper.

Validates that ``GenericAPIAdapter.aclient`` is rebuilt as an
``AsyncInstructor`` when ``litellm.acompletion`` is wrapped by a
callable class instance (the shape ``loongsuite-instrumentation-litellm``
puts in place) — the root-cause of the Cognee v1.2.1 + instructor 1.14.5
async bug reported by E2E.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import types

import pytest


class _SyncInstructor:
    """Shape of instructor.Instructor (sync path)."""

    def __init__(self, mode):
        self.mode = mode


class _AsyncInstructor:
    """Shape of instructor.AsyncInstructor."""

    def __init__(self, mode):
        self.mode = mode


class _LiteLLMAsyncCompletionWrapper:
    """Mimics ``loongsuite-instrumentation-litellm._wrapper.AsyncCompletionWrapper``.

    Crucially, ``inspect.iscoroutinefunction(instance)`` returns ``False``
    even though ``__call__`` is ``async def`` — this is the root cause of
    the instructor 1.14.5 sync-retry bug.
    """

    def __init__(self, original_acompletion):
        self.original_func = original_acompletion

    async def __call__(self, *args, **kwargs):
        return await self.original_acompletion(*args, **kwargs)


@pytest.fixture
def fake_instructor(monkeypatch):
    """Install a fake ``instructor`` module.

    ``from_litellm(completion, mode=...)`` returns ``_AsyncInstructor`` when
    ``inspect.iscoroutinefunction(completion)`` is True — matching real
    instructor behavior. This lets us verify the rebuild picks the async
    path when ``completion`` is a real ``async def``.
    """
    class _Mode:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f"Mode({self.value!r})"

    class _FakeInstructorModule:
        Mode = _Mode

        @staticmethod
        def from_litellm(completion, mode=_Mode("json_mode"), **kwargs):
            if inspect.iscoroutinefunction(completion):
                return _AsyncInstructor(mode)
            return _SyncInstructor(mode)

    fake = _FakeInstructorModule()
    monkeypatch.setitem(sys.modules, "instructor", fake)
    yield fake


@pytest.fixture
def fake_litellm(monkeypatch):
    """Install a fake ``litellm`` module with an async ``acompletion``."""

    async def _acompletion(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    class _FakeLitellmModule:
        acompletion = staticmethod(_acompletion)

    fake = _FakeLitellmModule()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    yield fake


def test_rebuild_aclient_as_async_when_sync(fake_instructor, fake_litellm):
    """When ``aclient`` is a sync ``Instructor``, rebuild as ``AsyncInstructor``."""
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    class FakeAdapter:
        def __init__(self):
            self.aclient = _SyncInstructor(mode=fake_instructor.Mode("json_mode"))

    instance = FakeAdapter()
    rebuilt = _llm_compat_wrapper._rebuild_aclient_as_async(instance)
    assert rebuilt is True
    assert type(instance.aclient).__name__ == "_AsyncInstructor"


def test_skip_rebuild_when_already_async(fake_instructor, fake_litellm):
    """When ``aclient`` is already ``AsyncInstructor``, skip rebuild."""
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    class FakeAdapter:
        def __init__(self):
            self.aclient = _AsyncInstructor(mode=fake_instructor.Mode("json_mode"))

    instance = FakeAdapter()
    rebuilt = _llm_compat_wrapper._rebuild_aclient_as_async(instance)
    assert rebuilt is False
    # Same instance — not rebuilt
    assert isinstance(instance.aclient, _AsyncInstructor)


def test_skip_rebuild_when_aclient_none(fake_instructor, fake_litellm):
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    class FakeAdapter:
        def __init__(self):
            self.aclient = None

    instance = FakeAdapter()
    rebuilt = _llm_compat_wrapper._rebuild_aclient_as_async(instance)
    assert rebuilt is False


def test_async_acompletion_wrapper_is_recognized_as_coroutine():
    """``_build_async_acompletion`` must return a real ``async def``."""
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    async def fake_acompletion(*args, **kwargs):
        return {"ok": True}

    wrapped = _llm_compat_wrapper._build_async_acompletion(fake_acompletion)
    assert inspect.iscoroutinefunction(wrapped) is True


def test_async_acompletion_wrapper_awaited_correctly(fake_litellm):
    """The wrapped acompletion resolves to ``litellm.acompletion`` at call time."""
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    async def fake_acompletion(*args, **kwargs):
        return {"result": "ok", "args": args, "kwargs": kwargs}

    wrapped = _llm_compat_wrapper._build_async_acompletion(fake_acompletion)
    result = asyncio.new_event_loop().run_until_complete(
        wrapped("a", kw="b")
    )
    # Underlying is the fake litellm.acompletion installed by fake_litellm,
    # since _async_acompletion resolves litellm.acompletion at call time.
    assert result == {"args": ("a",), "kwargs": {"kw": "b"}}


def test_init_wrapper_rebuilds_aclient(monkeypatch, fake_instructor, fake_litellm):
    """End-to-end: ``GenericAPIAdapter.__init__`` wrap rebuilds ``aclient``
    from a sync ``Instructor`` to an ``AsyncInstructor``."""
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    class GenericAPIAdapter:
        def __init__(self):
            # Simulate what Cognee does: instructor.from_litellm(litellm.acompletion, mode=...)
            # With the wrap installed, litellm.acompletion is the fake module's
            # async acompletion — but since the wrapper class shape is what
            # triggers the bug, we explicitly set a sync Instructor here.
            self.aclient = _SyncInstructor(mode=fake_instructor.Mode("json_mode"))

    mod_path = (
        "cognee.infrastructure.llm.structured_output_framework."
        "litellm_instructor.llm.generic_llm_api.adapter"
    )
    parts = mod_path.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            new_mod = types.ModuleType(sub)
            sys.modules[sub] = new_mod
            if i > 1:
                parent = sys.modules[".".join(parts[: i - 1])]
                setattr(parent, parts[i - 1], new_mod)
    sys.modules[mod_path].GenericAPIAdapter = GenericAPIAdapter

    _llm_compat_wrapper.install_llm_compat_wrapper()
    try:
        instance = GenericAPIAdapter()
        assert type(instance.aclient).__name__ == "_AsyncInstructor"
    finally:
        _llm_compat_wrapper.uninstall_llm_compat_wrapper()


def test_init_wrapper_handles_missing_aclient(monkeypatch, fake_instructor, fake_litellm):
    """If ``__init__`` doesn't set ``aclient``, the wrapper is a no-op."""
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    class GenericAPIAdapter:
        def __init__(self):
            pass  # no aclient

    mod_path = (
        "cognee.infrastructure.llm.structured_output_framework."
        "litellm_instructor.llm.generic_llm_api.adapter"
    )
    parts = mod_path.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            new_mod = types.ModuleType(sub)
            sys.modules[sub] = new_mod
            if i > 1:
                parent = sys.modules[".".join(parts[: i - 1])]
                setattr(parent, parts[i - 1], new_mod)
    sys.modules[mod_path].GenericAPIAdapter = GenericAPIAdapter

    _llm_compat_wrapper.install_llm_compat_wrapper()
    try:
        instance = GenericAPIAdapter()
        assert not hasattr(instance, "aclient") or instance.aclient is None
    finally:
        _llm_compat_wrapper.uninstall_llm_compat_wrapper()


def test_init_wrapper_idempotent_when_already_async(monkeypatch, fake_instructor, fake_litellm):
    """If ``__init__`` already produced an ``AsyncInstructor``, the wrap leaves it."""
    from opentelemetry.instrumentation.cognee.internal import (
        _llm_compat_wrapper,
    )

    class GenericAPIAdapter:
        def __init__(self):
            self.aclient = _AsyncInstructor(mode=fake_instructor.Mode("json_mode"))

    mod_path = (
        "cognee.infrastructure.llm.structured_output_framework."
        "litellm_instructor.llm.generic_llm_api.adapter"
    )
    parts = mod_path.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            new_mod = types.ModuleType(sub)
            sys.modules[sub] = new_mod
            if i > 1:
                parent = sys.modules[".".join(parts[: i - 1])]
                setattr(parent, parts[i - 1], new_mod)
    sys.modules[mod_path].GenericAPIAdapter = GenericAPIAdapter

    _llm_compat_wrapper.install_llm_compat_wrapper()
    try:
        instance = GenericAPIAdapter()
        # Should remain the same _AsyncInstructor — not rebuilt
        assert isinstance(instance.aclient, _AsyncInstructor)
    finally:
        _llm_compat_wrapper.uninstall_llm_compat_wrapper()
