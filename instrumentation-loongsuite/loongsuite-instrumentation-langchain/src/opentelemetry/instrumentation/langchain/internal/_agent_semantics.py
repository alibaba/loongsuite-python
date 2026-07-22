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

"""Internal metadata contract for LangGraph-based agent harnesses."""

from __future__ import annotations

from typing import Any

AGENT_FRAMEWORK_METADATA_KEY = "_loongsuite_agent_framework"
AGENT_STEP_NODE_METADATA_KEY = "_loongsuite_agent_step_node"
DEERFLOW_FRAMEWORK = "deerflow"


def get_agent_semantics(run: Any) -> tuple[str, str] | None:
    """Return validated framework and decision-node metadata for a run."""

    metadata = getattr(run, "metadata", None) or {}
    framework = metadata.get(AGENT_FRAMEWORK_METADATA_KEY)
    step_node = metadata.get(AGENT_STEP_NODE_METADATA_KEY)
    if not isinstance(framework, str) or not framework.strip():
        return None
    if not isinstance(step_node, str) or not step_node.strip():
        return None
    return framework.strip(), step_node.strip()
