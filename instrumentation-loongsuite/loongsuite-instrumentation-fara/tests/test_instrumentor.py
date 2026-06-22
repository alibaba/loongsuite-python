# Copyright The OpenTelemetry Authors
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

"""Tests for the FaraInstrumentor entrypoint and patch lifecycle."""

from __future__ import annotations

import fara.fara_agent as fara_agent_mod
import fara.run_fara as run_fara_mod
import pytest

from opentelemetry.instrumentation.fara import FaraInstrumentor


def test_instrumentation_dependencies():
    instr = FaraInstrumentor()
    deps = list(instr.instrumentation_dependencies())
    assert any("fara" in d for d in deps)


def test_instrument_patches_all_targets(tracer_provider):
    instr = FaraInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    try:
        # All four patch points should be wrapped.
        assert hasattr(run_fara_mod.run_fara_agent, "__wrapped__")
        assert hasattr(fara_agent_mod.FaraAgent.run, "__wrapped__")
        assert hasattr(
            fara_agent_mod.FaraAgent.generate_model_call, "__wrapped__"
        )
        assert hasattr(
            fara_agent_mod.FaraAgent.execute_action, "__wrapped__"
        )
    finally:
        instr.uninstrument()


def test_uninstrument_removes_wrappers(tracer_provider):
    instr = FaraInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    instr.uninstrument()
    assert not hasattr(run_fara_mod.run_fara_agent, "__wrapped__")
    assert not hasattr(fara_agent_mod.FaraAgent.run, "__wrapped__")
    assert not hasattr(
        fara_agent_mod.FaraAgent.generate_model_call, "__wrapped__"
    )
    assert not hasattr(
        fara_agent_mod.FaraAgent.execute_action, "__wrapped__"
    )


def test_instrument_failure_is_logged_not_raised(monkeypatch, tracer_provider):
    """A failing patch should be logged and skipped, not raised."""
    import opentelemetry.instrumentation.fara as fara_instr_pkg

    def _raise_wrap(*args, **kwargs):
        raise RuntimeError("patch failed")

    monkeypatch.setattr(
        fara_instr_pkg, "wrap_function_wrapper", _raise_wrap
    )
    instr = FaraInstrumentor()
    # Should not raise even though every patch target fails.
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    instr.uninstrument()


def test_entry_point_registered():
    """The instrumentor entry point should be discoverable."""
    import importlib.metadata as md

    eps = md.entry_points()
    if hasattr(eps, "select"):
        selected = list(
            eps.select(group="opentelemetry_instrumentor", name="fara")
        )
    else:  # pragma: no cover - py<3.10 fallback
        selected = [
            e for e in eps.get("opentelemetry_instrumentor", []) if e.name == "fara"
        ]
    assert selected, "fara entry point must be registered in pyproject.toml"
