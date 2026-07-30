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

"""Compatibility tests for QwenPaw primary and legacy import paths."""

from __future__ import annotations

import importlib.metadata
from types import ModuleType

from opentelemetry.instrumentation.qwenpaw import (
    CoPawInstrumentor,
    QwenPawInstrumentor,
)
from opentelemetry.instrumentation.qwenpaw.package import (
    get_installed_instrumentation_dependencies,
    get_installed_runner_modules,
    get_installed_runtime_targets,
)


def _fake_qwenpaw_version(name):
    if name == "qwenpaw":
        return "1.1.1"
    raise importlib.metadata.PackageNotFoundError


def _fake_copaw_version(name):
    if name == "copaw":
        return "1.0.2"
    raise importlib.metadata.PackageNotFoundError


def _fake_qwenpaw_v2_version(name):
    if name == "qwenpaw":
        return "2.0.0.post4"
    raise importlib.metadata.PackageNotFoundError


def test_runtime_detection_prefers_installed_qwenpaw(monkeypatch):
    monkeypatch.setattr(
        "opentelemetry.instrumentation.qwenpaw.package.version",
        _fake_qwenpaw_version,
    )

    assert get_installed_instrumentation_dependencies() == (
        "qwenpaw >= 1.1.0, < 2.0.0",
    )
    assert get_installed_runner_modules() == ("qwenpaw.app.runner.runner",)
    assert CoPawInstrumentor().instrumentation_dependencies() == (
        "qwenpaw >= 1.1.0, < 2.0.0",
    )


def test_runtime_detection_uses_qwenpaw_v2_runtime(monkeypatch):
    monkeypatch.setattr(
        "opentelemetry.instrumentation.qwenpaw.package.version",
        _fake_qwenpaw_v2_version,
    )

    assert get_installed_instrumentation_dependencies() == (
        "qwenpaw >= 2.0.0",
    )
    [target] = get_installed_runtime_targets()
    assert target.module_name == "qwenpaw.runtime.runtime"
    assert target.class_name == "Runtime"
    assert target.method_name == "run"
    assert target.wrapper_kind == "runtime"


def test_runtime_detection_falls_back_to_legacy_copaw(monkeypatch):
    monkeypatch.setattr(
        "opentelemetry.instrumentation.qwenpaw.package.version",
        _fake_copaw_version,
    )

    assert get_installed_instrumentation_dependencies() == (
        "copaw >= 0.1.0, <= 1.0.2",
    )
    assert get_installed_runner_modules() == ("copaw.app.runner.runner",)
    assert QwenPawInstrumentor().instrumentation_dependencies() == (
        "copaw >= 0.1.0, <= 1.0.2",
    )


def test_uninstrument_handles_qwenpaw_runner(monkeypatch):
    runner_module = ModuleType("qwenpaw.app.runner.runner")
    runner_module.AgentRunner = type("AgentRunner", (), {})
    unwrap_calls = []

    monkeypatch.setattr(
        "opentelemetry.instrumentation.qwenpaw.package.version",
        _fake_qwenpaw_version,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.qwenpaw.import_module",
        lambda name: runner_module,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.qwenpaw.unwrap",
        lambda cls, attr: unwrap_calls.append((cls, attr)),
    )

    inst = CoPawInstrumentor()
    inst._is_instrumented_by_opentelemetry = True
    inst.uninstrument()

    assert unwrap_calls == [(runner_module.AgentRunner, "query_handler")]


def test_qwenpaw_alias_points_to_same_instrumentor():
    assert QwenPawInstrumentor is CoPawInstrumentor


def test_copaw_import_path_alias():
    from opentelemetry.instrumentation.copaw import (  # noqa: PLC0415
        CoPawInstrumentor as ImportedCoPawInstrumentor,
    )

    assert ImportedCoPawInstrumentor is QwenPawInstrumentor
