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

"""Package-scoped bootstrap registry fragments for LoongSuite distro."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import Any

_SOURCE_SORT_ORDER = {
    "genai-renamed": 0,
    "upstream": 1,
    "loongsuite": 2,
}


def _iter_registry_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    package_path = globals()["__path__"]
    for module_info in sorted(
        iter_modules(package_path),
        key=lambda info: info.name,
    ):
        if module_info.ispkg or module_info.name.startswith("_"):
            continue

        module = import_module(f"{__name__}.{module_info.name}")
        registry = getattr(module, "REGISTRY", None)
        if registry is not None:
            entries.append(registry)

    return sorted(
        entries,
        key=lambda registry: (
            _SOURCE_SORT_ORDER.get(registry.get("source"), 99),
            registry.get("package", ""),
        ),
    )


def load_bootstrap_registry() -> tuple[list[dict[str, str]], list[str]]:
    libraries: list[dict[str, str]] = []
    default_instrumentations: list[str] = []

    for registry in _iter_registry_entries():
        instrumentation = registry["instrumentation"]
        target_libraries = registry.get("libraries") or []
        if not target_libraries:
            default_instrumentations.append(instrumentation)
            continue

        for target_library in target_libraries:
            libraries.append(
                {
                    "library": target_library,
                    "instrumentation": instrumentation,
                }
            )

    return libraries, default_instrumentations


libraries, default_instrumentations = load_bootstrap_registry()

__all__ = [
    "default_instrumentations",
    "libraries",
    "load_bootstrap_registry",
]
