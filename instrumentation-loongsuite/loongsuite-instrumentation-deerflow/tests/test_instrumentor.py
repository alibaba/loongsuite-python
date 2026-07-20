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

"""Instrumentor, version guard, and graph marker tests."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from importlib.metadata import (
    PackageNotFoundError,
)
from importlib.metadata import (
    version as distribution_version,
)
from types import ModuleType

import pytest
from packaging.version import Version

from opentelemetry.instrumentation.deerflow import (
    DeerFlowInstrumentor,
    _deerflow_runtime_supported,
)
from opentelemetry.instrumentation.deerflow.internal.constants import (
    AGENT_FLAVOR_ATTR,
    GATEWAY_RUN_AGENT_ALIASES,
)
from opentelemetry.instrumentation.deerflow.internal.patch import (
    _create_agent_alias_wrapper,
    instrument_deerflow,
    mark_deerflow_graph,
    uninstrument_deerflow,
)
from opentelemetry.instrumentation.deerflow.package import _instruments


class _Graph:
    pass


def test_has_no_pypi_target_library_dependency():
    assert _instruments == ()
    assert DeerFlowInstrumentor().instrumentation_dependencies() == ()


def test_official_source_distribution_and_gateway_signature_are_supported():
    assert Version("2") <= Version(distribution_version("deerflow-harness"))
    assert Version(distribution_version("deerflow-harness")) < Version("3")
    assert _deerflow_runtime_supported() is True

    run_agent = importlib.import_module(
        "deerflow.runtime.runs.worker"
    ).run_agent
    parameters = inspect.signature(inspect.unwrap(run_agent)).parameters
    assert list(parameters)[:3] == ["bridge", "run_manager", "record"]
    assert parameters["graph_input"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["config"].kind is inspect.Parameter.KEYWORD_ONLY


def test_probe_first_cold_start_wraps_each_deerflow_alias_once():
    script = """
import importlib
import sys

assert not any(name == "deerflow" or name.startswith("deerflow.") for name in sys.modules)

from opentelemetry.instrumentation.deerflow import DeerFlowInstrumentor
from opentelemetry.instrumentation.deerflow.internal.constants import (
    CREATE_AGENT_ALIASES,
    GATEWAY_RUN_AGENT_ALIASES,
)


def wrapper_depth(value):
    depth = 0
    seen = set()
    while hasattr(value, "__wrapped__") and id(value) not in seen:
        seen.add(id(value))
        value = value.__wrapped__
        depth += 1
    return depth


def wrapper_modules(value):
    modules = []
    seen = set()
    while hasattr(value, "__wrapped__") and id(value) not in seen:
        seen.add(id(value))
        wrapper = getattr(value, "_self_wrapper", None)
        modules.append(getattr(wrapper, "__module__", None))
        value = value.__wrapped__
    return modules


instrumentor = DeerFlowInstrumentor()
instrumentor.instrument()
assert instrumentor.is_instrumented_by_opentelemetry

aliases = (*CREATE_AGENT_ALIASES, *GATEWAY_RUN_AGENT_ALIASES)
for module_name, target in aliases:
    module = importlib.import_module(module_name)
    value = getattr(module, target)
    modules = wrapper_modules(value)
    assert modules.count(
        "opentelemetry.instrumentation.deerflow.internal.patch"
    ) == 1, (module_name, target, modules)
    expected_depth = 2 if (module_name, target) in CREATE_AGENT_ALIASES else 1
    assert wrapper_depth(value) == expected_depth, (module_name, target, modules)

instrumentor.uninstrument()
for module_name, target in aliases:
    module = importlib.import_module(module_name)
    assert wrapper_depth(getattr(module, target)) == 0, (module_name, target)
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("installed_version", "supported"),
    [
        ("1.9.9", False),
        ("2.0.0", True),
        ("2.1.0", True),
        ("3.0.0", False),
    ],
)
def test_runtime_version_guard(
    installed_version,
    supported,
    monkeypatch,
):
    real_import = importlib.import_module

    def import_module(name):
        if name == "deerflow":
            return object()
        return real_import(name)

    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.importlib.import_module",
        import_module,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.version",
        lambda _distribution: installed_version,
    )

    assert _deerflow_runtime_supported() is supported


def test_runtime_guard_silently_skips_failed_deerflow_import(monkeypatch):
    missing_dependency = ModuleNotFoundError("missing DeerFlow dependency")
    missing_dependency.name = "deerflow_optional_dependency"

    def fail_import(_name):
        raise missing_dependency

    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.importlib.import_module",
        fail_import,
    )

    assert _deerflow_runtime_supported() is False


def test_runtime_guard_silently_skips_missing_distribution_metadata(
    monkeypatch,
):
    real_import = importlib.import_module

    def import_module(name):
        if name == "deerflow":
            return object()
        return real_import(name)

    def missing_distribution(_distribution):
        raise PackageNotFoundError

    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.importlib.import_module",
        import_module,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.version",
        missing_distribution,
    )

    assert _deerflow_runtime_supported() is False


def test_instrument_rolls_back_started_dependency_on_setup_error(monkeypatch):
    class StartedDependency:
        def __init__(self):
            self.uninstrument_calls = 0

        def uninstrument(self):
            self.uninstrument_calls += 1

    dependency = StartedDependency()
    dependency_calls = 0
    deerflow_uninstrument_calls = 0

    def instrument_dependency(*_args, **_kwargs):
        nonlocal dependency_calls
        dependency_calls += 1
        if dependency_calls == 1:
            return dependency
        raise RuntimeError("dependency setup failed")

    def uninstrument_patches():
        nonlocal deerflow_uninstrument_calls
        deerflow_uninstrument_calls += 1

    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow._deerflow_runtime_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow._instrument_dependency",
        instrument_dependency,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.uninstrument_deerflow",
        uninstrument_patches,
    )

    instrumentor = DeerFlowInstrumentor()
    with pytest.raises(RuntimeError, match="dependency setup failed"):
        instrumentor._instrument()

    assert dependency.uninstrument_calls == 1
    assert deerflow_uninstrument_calls == 1
    assert instrumentor._dependency_instrumentors == []
    assert instrumentor._deerflow_patched is False


def test_create_agent_alias_marks_only_new_flavor():
    graph = _Graph()
    graph._loongsuite_react_agent = True
    result = _create_agent_alias_wrapper(
        lambda: graph,
        None,
        (),
        {},
    )

    assert result is graph
    assert getattr(graph, AGENT_FLAVOR_ATTR) == "deerflow"
    assert not hasattr(graph, "_loongsuite_react_agent")
    assert not hasattr(graph, "_loongsuite_deepagents_agent")

    uninstrument_deerflow()
    assert not hasattr(graph, AGENT_FLAVOR_ATTR)
    assert graph._loongsuite_react_agent is True


def test_uninstrument_restores_previous_graph_flavor():
    graph = _Graph()
    setattr(graph, AGENT_FLAVOR_ATTR, "langchain-create-agent")

    mark_deerflow_graph(graph)
    assert getattr(graph, AGENT_FLAVOR_ATTR) == "deerflow"

    uninstrument_deerflow()
    assert getattr(graph, AGENT_FLAVOR_ATTR) == "langchain-create-agent"


def test_gateway_aliases_are_wrapped_and_restored(
    handler,
    monkeypatch,
):
    real_aliases = []
    for module_name, target in GATEWAY_RUN_AGENT_ALIASES:
        module = importlib.import_module(module_name)
        real_aliases.append((module, target, getattr(module, target)))

    async def service_run_agent(*_args, **_kwargs):
        return None

    app_module = ModuleType("app")
    app_module.__path__ = []
    gateway_module = ModuleType("app.gateway")
    gateway_module.__path__ = []
    services_module = ModuleType("app.gateway.services")
    services_module.run_agent = service_run_agent
    app_module.gateway = gateway_module
    gateway_module.services = services_module
    monkeypatch.setitem(sys.modules, "app", app_module)
    monkeypatch.setitem(sys.modules, "app.gateway", gateway_module)
    monkeypatch.setitem(
        sys.modules,
        "app.gateway.services",
        services_module,
    )

    try:
        assert instrument_deerflow(handler) is True
        for module, target, original in real_aliases:
            wrapped = getattr(module, target)
            assert wrapped is not original
            assert inspect.unwrap(wrapped) is original

        assert services_module.run_agent is not service_run_agent
        assert inspect.unwrap(services_module.run_agent) is service_run_agent
    finally:
        uninstrument_deerflow()

    for module, target, original in real_aliases:
        assert getattr(module, target) is original
    assert services_module.run_agent is service_run_agent


def test_late_loaded_gateway_alias_is_restored(handler, monkeypatch):
    monkeypatch.delitem(sys.modules, "app.gateway.services", raising=False)
    runtime_module = importlib.import_module("deerflow.runtime")
    original = runtime_module.run_agent

    try:
        assert instrument_deerflow(handler) is True
        late_alias = runtime_module.run_agent
        assert late_alias is not original

        services_module = ModuleType("app.gateway.services")
        services_module.run_agent = late_alias
        monkeypatch.setitem(
            sys.modules,
            "app.gateway.services",
            services_module,
        )

        uninstrument_deerflow()

        assert runtime_module.run_agent is original
        assert services_module.run_agent is original
    finally:
        uninstrument_deerflow()
