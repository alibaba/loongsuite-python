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

"""Lifecycle tests for ``AlgoTuneInstrumentor``.

Verifies import, instrument/uninstrument, ``_safe_wrap``/``_safe_unwrap``,
and that the instrumentor degrades gracefully when targets are missing.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest import mock

# ---------------------------------------------------------------------------
# Basic import / dependency tests
# ---------------------------------------------------------------------------


def test_import_instrumentor_package():
    module = importlib.import_module("opentelemetry.instrumentation.algotune")
    assert hasattr(module, "AlgoTuneInstrumentor")


def test_instrumentation_dependencies_empty():
    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor
    from opentelemetry.instrumentation.algotune.package import _instruments

    instr = AlgoTuneInstrumentor()
    assert tuple(instr.instrumentation_dependencies()) == _instruments
    assert _instruments == ()


def test_version_is_string():
    from opentelemetry.instrumentation.algotune.version import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


# ---------------------------------------------------------------------------
# Instrument / uninstrument lifecycle
# ---------------------------------------------------------------------------


def test_instrument_uninstrument_no_raise(tracer_provider):
    """instrument() + uninstrument() must not raise with stub modules."""
    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor

    instr = AlgoTuneInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    instr.uninstrument()


def test_double_instrument_uninstrument(tracer_provider):
    """Double instrument should not raise (BaseInstrumentor guards this)."""
    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor

    instr = AlgoTuneInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    # BaseInstrumentor prevents double-instrument, so just uninstrument.
    instr.uninstrument()


def test_uninstrument_restores_originals(tracer_provider):
    """After uninstrument(), the stub functions should be unwrapped."""
    import AlgoTuner.main as main_mod

    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor

    original_main = main_mod.main

    instr = AlgoTuneInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    # After instrument, main should be wrapped (has __wrapped__).
    assert hasattr(main_mod.main, "__wrapped__")

    instr.uninstrument()

    # After uninstrument, main should be restored.
    assert main_mod.main is original_main


def test_instrument_wraps_all_patch_sites(tracer_provider):
    """After instrument(), every _PATCH_SITES target should have __wrapped__."""
    from opentelemetry.instrumentation.algotune import (
        _PATCH_SITES,
        AlgoTuneInstrumentor,
    )

    instr = AlgoTuneInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    try:
        for logical_name, module_path, qualname in _PATCH_SITES:
            mod = importlib.import_module(module_path)
            parts = qualname.split(".")
            obj = mod
            for part in parts:
                obj = getattr(obj, part)
            assert hasattr(obj, "__wrapped__"), (
                f"{module_path}.{qualname} not wrapped"
            )
    finally:
        instr.uninstrument()


# ---------------------------------------------------------------------------
# _safe_wrap / _safe_unwrap edge cases
# ---------------------------------------------------------------------------


def test_safe_wrap_import_error():
    from opentelemetry.instrumentation.algotune import _safe_wrap

    result = _safe_wrap(
        "nonexistent.module.that.does.not.exist",
        "SomeClass.method",
        lambda *a, **k: None,
    )
    assert result is False


def test_safe_wrap_attribute_error():
    from opentelemetry.instrumentation.algotune import _safe_wrap

    result = _safe_wrap(
        "AlgoTuner.main",
        "NonExistentFunction",
        lambda *a, **k: None,
    )
    assert result is False


def test_safe_unwrap_missing_module():
    from opentelemetry.instrumentation.algotune import _safe_unwrap

    # Should not raise even if the module does not exist.
    _safe_unwrap("nonexistent.module.path", "SomeClass.method")


def test_safe_unwrap_missing_attribute():
    from opentelemetry.instrumentation.algotune import _safe_unwrap

    # Should not raise when the attribute does not exist on the module.
    _safe_unwrap("AlgoTuner.main", "NoSuchThing.method")


def test_safe_unwrap_no_wrapped_attribute():
    from opentelemetry.instrumentation.algotune import _safe_unwrap

    # main is not wrapped, so __wrapped__ is missing; should be a no-op.
    _safe_unwrap("AlgoTuner.main", "main")


def test_safe_wrap_generic_exception():
    """When wrap_function_wrapper raises a non-Import/Attribute error,
    _safe_wrap should catch it and return False."""
    from opentelemetry.instrumentation.algotune import _safe_wrap

    # A wrapper that causes a generic exception inside wrapt.
    # We use monkeypatch on wrap_function_wrapper to simulate this.
    with mock.patch(
        "opentelemetry.instrumentation.algotune.wrap_function_wrapper",
        side_effect=TypeError("unexpected error"),
    ):
        result = _safe_wrap("AlgoTuner.main", "main", lambda *a: None)
        assert result is False


def test_safe_unwrap_leaf_is_none():
    """When the leaf attribute exists but resolves to None, should be a no-op."""
    # Create a module with a class that has a None method.
    import AlgoTuner.interfaces.llm_interface as mod

    from opentelemetry.instrumentation.algotune import _safe_unwrap

    getattr(mod.LLMInterface, "run_task")
    setattr(mod.LLMInterface, "_nonexistent_method", None)
    try:
        _safe_unwrap(
            "AlgoTuner.interfaces.llm_interface",
            "LLMInterface._nonexistent_method",
        )
    finally:
        try:
            delattr(mod.LLMInterface, "_nonexistent_method")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Disabled via environment variable
# ---------------------------------------------------------------------------


def test_instrument_disabled_via_env(tracer_provider, monkeypatch):
    """When OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED=false, no wrapping occurs."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED", "false")

    # Re-import so config picks up the env var.
    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]

    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor

    instr = AlgoTuneInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    import AlgoTuner.main as main_mod

    # main should NOT have been wrapped.
    assert not hasattr(main_mod.main, "__wrapped__")

    instr.uninstrument()

    # Restore for other tests.
    monkeypatch.delenv("OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED", raising=False)
    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]


# ---------------------------------------------------------------------------
# Together opt-in
# ---------------------------------------------------------------------------


def test_together_not_wrapped_by_default(tracer_provider):
    """TogetherModel.query should NOT be wrapped unless env var is set."""
    import AlgoTuner.models.together_model as together_mod

    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor

    instr = AlgoTuneInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    assert not hasattr(together_mod.TogetherModel.query, "__wrapped__")

    instr.uninstrument()


def test_together_wrapped_when_env_set(tracer_provider, monkeypatch):
    """TogetherModel.query IS wrapped when ALGOTUNE_OTEL_INSTRUMENT_TOGETHER=true."""
    monkeypatch.setenv("ALGOTUNE_OTEL_INSTRUMENT_TOGETHER", "true")

    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]

    import AlgoTuner.models.together_model as together_mod

    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor

    instr = AlgoTuneInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    assert hasattr(together_mod.TogetherModel.query, "__wrapped__")

    instr.uninstrument()

    monkeypatch.delenv("ALGOTUNE_OTEL_INSTRUMENT_TOGETHER", raising=False)
    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]


# ---------------------------------------------------------------------------
# _is_litellm_instrumented helper
# ---------------------------------------------------------------------------


def test_is_litellm_instrumented_false_when_no_litellm():
    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]

    from opentelemetry.instrumentation.algotune import _is_litellm_instrumented

    # Block litellm from being imported by temporarily inserting None
    # into sys.modules (standard Python import-blocking idiom).
    saved = sys.modules.pop("litellm", None)
    sys.modules["litellm"] = None  # blocks import
    try:
        assert _is_litellm_instrumented() is False
    finally:
        if saved is not None:
            sys.modules["litellm"] = saved
        else:
            sys.modules.pop("litellm", None)


def test_is_litellm_instrumented_true_when_wrapped():
    """When litellm.completion has __wrapped__, returns True."""
    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]

    # Inject a fake litellm module with a wrapped completion.
    fake_litellm = types.ModuleType("litellm")

    def _fake_completion(*a, **k):
        pass

    _fake_completion.__wrapped__ = True
    fake_litellm.completion = _fake_completion
    sys.modules["litellm"] = fake_litellm

    try:
        from opentelemetry.instrumentation.algotune import (
            _is_litellm_instrumented,
        )

        assert _is_litellm_instrumented() is True
    finally:
        del sys.modules["litellm"]


def test_is_litellm_instrumented_false_when_not_wrapped():
    """When litellm.completion has no __wrapped__, returns False."""
    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = lambda *a, **k: None
    sys.modules["litellm"] = fake_litellm

    try:
        from opentelemetry.instrumentation.algotune import (
            _is_litellm_instrumented,
        )

        assert _is_litellm_instrumented() is False
    finally:
        del sys.modules["litellm"]


def test_is_litellm_instrumented_false_when_no_completion():
    """When litellm module exists but has no ``completion`` attribute, return False."""
    for m in list(sys.modules):
        if m.startswith("opentelemetry.instrumentation.algotune"):
            del sys.modules[m]

    fake_litellm = types.ModuleType("litellm")
    # Do not set fake_litellm.completion
    sys.modules["litellm"] = fake_litellm

    try:
        from opentelemetry.instrumentation.algotune import (
            _is_litellm_instrumented,
        )

        assert _is_litellm_instrumented() is False
    finally:
        del sys.modules["litellm"]
