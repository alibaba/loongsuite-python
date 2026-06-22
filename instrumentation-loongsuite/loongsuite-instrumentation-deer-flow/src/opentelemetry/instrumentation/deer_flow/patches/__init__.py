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

"""DeerFlow wrapt patch modules.

Each submodule wraps a DeerFlow public API with an ``ExtendedTelemetryHandler``
-powered span. ``instrument(handler)`` returns a list of ``(module_name,
attr_path)`` tuples that ``DeerFlowInstrumentor._uninstrument`` can pass to
``unwrap``.
"""

from __future__ import annotations

from . import entry, memory, sandbox, subagent, task_tool

__all__ = ["entry", "memory", "sandbox", "subagent", "task_tool"]
