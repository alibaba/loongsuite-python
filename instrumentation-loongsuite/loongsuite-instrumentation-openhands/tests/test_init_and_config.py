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

"""Tests for __init__.py (instrumentor plumbing) and config.py."""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(autouse=True)
def _reset_global_tracer():
    yield
    trace_api._TRACER_PROVIDER = None  # type: ignore[attr-defined]


@pytest.fixture
def tracer_provider():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider._exporter = exporter
    return provider


# ---------------------------------------------------------------------------
# _module_importable
# ---------------------------------------------------------------------------


def test_module_importable_existing():
    from opentelemetry.instrumentation.openhands import _module_importable

    assert _module_importable("os") is True


def test_module_importable_missing():
    from opentelemetry.instrumentation.openhands import _module_importable

    assert _module_importable("nonexistent_module_xyz_12345") is False


def test_module_importable_other_import_error():
    """Non-ModuleNotFoundError import errors should return True."""
    from opentelemetry.instrumentation.openhands import _module_importable

    mod_name = "_test_import_error_module"
    # Create a module that raises a generic exception on import
    types.ModuleType(mod_name)
    # We need to simulate an import error. Use importlib to make
    # importlib.import_module raise a non-ModuleNotFoundError.
    with mock.patch("importlib.import_module", side_effect=ValueError("bad")):
        result = _module_importable("anything")
    assert result is True


# ---------------------------------------------------------------------------
# _safe_wrap
# ---------------------------------------------------------------------------


def test_safe_wrap_module_not_importable():
    from opentelemetry.instrumentation.openhands import _safe_wrap

    result = _safe_wrap("nonexistent_module_abc", "func", lambda *a: None)
    assert result is False


def test_safe_wrap_attribute_error():
    from opentelemetry.instrumentation.openhands import _safe_wrap

    # Module exists but attribute doesn't
    result = _safe_wrap("os", "nonexistent_function_xyz", lambda *a: None)
    assert result is False


# ---------------------------------------------------------------------------
# _safe_unwrap
# ---------------------------------------------------------------------------


def test_safe_unwrap_missing_module():
    from opentelemetry.instrumentation.openhands import _safe_unwrap

    # Should not raise
    _safe_unwrap("nonexistent_module_xyz_12345", "func")


def test_safe_unwrap_missing_attr():
    from opentelemetry.instrumentation.openhands import _safe_unwrap

    _safe_unwrap("os", "nonexistent_attr_xyz")


def test_safe_unwrap_not_wrapped():
    from opentelemetry.instrumentation.openhands import _safe_unwrap

    _safe_unwrap("os", "path")  # os.path exists but isn't wrapped


def test_safe_unwrap_with_wrapped():
    from opentelemetry.instrumentation.openhands import _safe_unwrap

    mod_name = "_test_unwrap_mod"
    mod = types.ModuleType(mod_name)

    def original():
        return "original"

    def wrapper():
        return "wrapped"

    wrapper.__wrapped__ = original
    mod.func = wrapper
    sys.modules[mod_name] = mod
    try:
        _safe_unwrap(mod_name, "func")
        # After unwrap, func should be restored to original
        assert mod.func() == "original"
    finally:
        del sys.modules[mod_name]


def test_safe_unwrap_qualname():
    from opentelemetry.instrumentation.openhands import _safe_unwrap

    mod_name = "_test_unwrap_qual"
    mod = types.ModuleType(mod_name)

    def original():
        return "original"

    def wrapper():
        return "wrapped"

    wrapper.__wrapped__ = original

    class Inner:
        pass

    Inner.method = wrapper
    mod.MyClass = Inner
    sys.modules[mod_name] = mod
    try:
        _safe_unwrap(mod_name, "MyClass.method")
        assert Inner.method() == "original"
    finally:
        del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# OpenHandsInstrumentor
# ---------------------------------------------------------------------------


def test_instrumentation_dependencies():
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    inst = OpenHandsInstrumentor()
    deps = inst.instrumentation_dependencies()
    assert isinstance(deps, (list, tuple, set, frozenset))


def test_instrument_disabled(tracer_provider):
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    with mock.patch.dict(
        os.environ, {"OTEL_INSTRUMENTATION_OPENHANDS_ENABLED": "false"}
    ):
        # Need to reload config to pick up the env var
        import opentelemetry.instrumentation.openhands.config as cfg_mod

        original = cfg_mod.OTEL_INSTRUMENTATION_OPENHANDS_ENABLED
        cfg_mod.OTEL_INSTRUMENTATION_OPENHANDS_ENABLED = False
        try:
            inst = OpenHandsInstrumentor()
            inst.instrument(
                tracer_provider=tracer_provider, skip_dep_check=True
            )
            inst.uninstrument()
        finally:
            cfg_mod.OTEL_INSTRUMENTATION_OPENHANDS_ENABLED = original


def test_maybe_enable_litellm_not_available(tracer_provider):
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    inst = OpenHandsInstrumentor()
    # Should not raise when litellm is not available
    inst._maybe_enable_litellm(tracer_provider=tracer_provider)


def test_maybe_enable_litellm_available_but_fails(tracer_provider):
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    class FakeLiteLLMInstrumentor:
        _is_instrumented_by_opentelemetry = False

        def instrument(self, **kwargs):
            raise RuntimeError("fail")

    with mock.patch.dict(
        sys.modules,
        {
            "opentelemetry.instrumentation.litellm": types.ModuleType(
                "opentelemetry.instrumentation.litellm"
            )
        },
    ):
        sys.modules[
            "opentelemetry.instrumentation.litellm"
        ].LiteLLMInstrumentor = FakeLiteLLMInstrumentor
        inst = OpenHandsInstrumentor()
        # Should not raise
        inst._maybe_enable_litellm(tracer_provider=tracer_provider)


def test_maybe_enable_litellm_already_instrumented(tracer_provider):
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    class FakeLiteLLMInstrumentor:
        _is_instrumented_by_opentelemetry = True
        instrument_called = False

        def instrument(self, **kwargs):
            FakeLiteLLMInstrumentor.instrument_called = True

    with mock.patch.dict(
        sys.modules,
        {
            "opentelemetry.instrumentation.litellm": types.ModuleType(
                "opentelemetry.instrumentation.litellm"
            )
        },
    ):
        sys.modules[
            "opentelemetry.instrumentation.litellm"
        ].LiteLLMInstrumentor = FakeLiteLLMInstrumentor
        inst = OpenHandsInstrumentor()
        inst._maybe_enable_litellm(tracer_provider=tracer_provider)
        assert not FakeLiteLLMInstrumentor.instrument_called


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


def test_bool_env_parsing():
    from opentelemetry.instrumentation.openhands.config import _bool_env

    with mock.patch.dict(os.environ, {"TEST_VAR": "true"}):
        assert _bool_env("TEST_VAR", False) is True
    with mock.patch.dict(os.environ, {"TEST_VAR": "1"}):
        assert _bool_env("TEST_VAR", False) is True
    with mock.patch.dict(os.environ, {"TEST_VAR": "yes"}):
        assert _bool_env("TEST_VAR", False) is True
    with mock.patch.dict(os.environ, {"TEST_VAR": "on"}):
        assert _bool_env("TEST_VAR", False) is True
    with mock.patch.dict(os.environ, {"TEST_VAR": "false"}):
        assert _bool_env("TEST_VAR", True) is False
    with mock.patch.dict(os.environ, {"TEST_VAR": "0"}):
        assert _bool_env("TEST_VAR", True) is False
    # Not set — use default
    assert _bool_env("UNSET_TEST_VAR_XYZ", True) is True
    assert _bool_env("UNSET_TEST_VAR_XYZ", False) is False


# ---------------------------------------------------------------------------
# Instrument disabled — patch the module-level binding
# Covers: __init__.py lines 152-154
# ---------------------------------------------------------------------------


def test_instrument_disabled_via_module_patch(tracer_provider):
    """Patch the module-level OTEL_INSTRUMENTATION_OPENHANDS_ENABLED to False."""
    import opentelemetry.instrumentation.openhands as oh_mod
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    original = oh_mod.OTEL_INSTRUMENTATION_OPENHANDS_ENABLED
    oh_mod.OTEL_INSTRUMENTATION_OPENHANDS_ENABLED = False
    try:
        inst = OpenHandsInstrumentor()
        inst.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        # Should be a no-op — no wrapping applied
    finally:
        oh_mod.OTEL_INSTRUMENTATION_OPENHANDS_ENABLED = original
        try:
            inst.uninstrument()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# _safe_unwrap — setattr failure
# Covers: __init__.py lines 141-142
# ---------------------------------------------------------------------------


def test_safe_unwrap_setattr_error():
    """When setattr on the parent fails, _safe_unwrap swallows the error."""
    from opentelemetry.instrumentation.openhands import _safe_unwrap

    mod_name = "_test_safe_unwrap_setattr_err"
    mod = types.ModuleType(mod_name)

    original_fn = lambda: None  # noqa: E731
    wrapped_fn = lambda: None  # noqa: E731
    wrapped_fn.__wrapped__ = original_fn

    class ReadOnlyContainer:
        """A container where setattr raises."""

        func = wrapped_fn

        def __setattr__(self, name, value):
            raise AttributeError("read-only")

    container = ReadOnlyContainer()
    mod.container = container  # type: ignore[attr-defined]
    sys.modules[mod_name] = mod
    try:
        # Should not raise despite setattr failure
        _safe_unwrap(mod_name, "container.func")
    finally:
        del sys.modules[mod_name]
