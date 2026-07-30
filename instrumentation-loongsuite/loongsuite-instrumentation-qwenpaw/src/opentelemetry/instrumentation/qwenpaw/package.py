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

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Collection, Iterator

from packaging.requirements import Requirement

_instruments = ()
_instruments_copaw = "copaw >= 0.1.0, <= 1.0.2"
_instruments_qwenpaw_v1 = "qwenpaw >= 1.1.0, < 2.0.0"
_instruments_qwenpaw_v2 = "qwenpaw >= 2.0.0"
_instruments_qwenpaw = "qwenpaw >= 1.1.0"
_instruments_any = (_instruments_qwenpaw, _instruments_copaw)


@dataclass(frozen=True)
class RuntimeTarget:
    requirement: str
    distribution_name: str
    module_name: str
    class_name: str
    method_name: str
    wrapper_kind: str


_runtime_targets = (
    RuntimeTarget(
        _instruments_qwenpaw_v2,
        "qwenpaw",
        "qwenpaw.runtime.runtime",
        "Runtime",
        "run",
        "runtime",
    ),
    RuntimeTarget(
        _instruments_qwenpaw_v1,
        "qwenpaw",
        "qwenpaw.app.runner.runner",
        "AgentRunner",
        "query_handler",
        "query_handler",
    ),
    RuntimeTarget(
        _instruments_copaw,
        "copaw",
        "copaw.app.runner.runner",
        "AgentRunner",
        "query_handler",
        "query_handler",
    ),
)


def get_installed_runtime_targets() -> Iterator[RuntimeTarget]:
    for runtime_target in _runtime_targets:
        try:
            installed_version = version(runtime_target.distribution_name)
        except PackageNotFoundError:
            continue
        if Requirement(runtime_target.requirement).specifier.contains(
            installed_version
        ):
            yield runtime_target


def get_installed_instrumentation_dependencies() -> Collection[str]:
    return tuple(
        target.requirement for target in get_installed_runtime_targets()
    )


def get_installed_runner_modules() -> Collection[str]:
    """Return matched module names for compatibility with older callers."""

    return tuple(
        target.module_name for target in get_installed_runtime_targets()
    )
