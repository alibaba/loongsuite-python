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

from importlib.metadata import PackageNotFoundError, version

from packaging.requirements import Requirement

_instruments_v1 = ("agentscope >= 1.0.0, < 2.0.0",)
_instruments_v2 = ("agentscope >= 2.0.0, < 3.0.0",)
_instruments = ("agentscope >= 1.0.0, < 3.0.0",)

_supports_metrics = False


def get_installed_instrumentation_dependencies():
    """Return the AgentScope dependency range matching the installed major."""
    try:
        installed_version = version("agentscope")
    except PackageNotFoundError:
        return _instruments

    for requirement in (_instruments_v2[0], _instruments_v1[0]):
        if Requirement(requirement).specifier.contains(
            installed_version,
            prereleases=True,
        ):
            return (requirement,)

    return _instruments
