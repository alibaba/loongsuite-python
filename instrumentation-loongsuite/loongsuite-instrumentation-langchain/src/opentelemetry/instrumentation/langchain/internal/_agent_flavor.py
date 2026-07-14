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

"""Internal contract for distinguishing LangGraph-based agent harnesses."""

from __future__ import annotations

from typing import Any

AGENT_FLAVOR_METADATA_KEY = "_loongsuite_agent_flavor"
REACT_AGENT_METADATA_KEY = "_loongsuite_react_agent"
DEEPAGENTS_METADATA_KEY = "_loongsuite_deepagents_agent"

LANGGRAPH_PREBUILT_AGENT_FLAVOR = "langgraph-prebuilt"
LANGCHAIN_CREATE_AGENT_FLAVOR = "langchain-create-agent"
DEEPAGENTS_AGENT_FLAVOR = "deepagents"
DEERFLOW_AGENT_FLAVOR = "deerflow"

AGENT_FLAVOR_STEP_NODES = {
    LANGGRAPH_PREBUILT_AGENT_FLAVOR: "agent",
    LANGCHAIN_CREATE_AGENT_FLAVOR: "model",
    DEEPAGENTS_AGENT_FLAVOR: "model",
    DEERFLOW_AGENT_FLAVOR: "model",
}
SUPPORTED_AGENT_FLAVORS = frozenset(AGENT_FLAVOR_STEP_NODES)


def get_agent_flavor(run: Any) -> str | None:
    """Return the single agent flavor carried by a callback run.

    The explicit flavor is authoritative. The boolean markers are read only as
    compatibility fallbacks for graphs produced by older instrumentation.
    """

    metadata = getattr(run, "metadata", None) or {}
    explicit_flavor = metadata.get(AGENT_FLAVOR_METADATA_KEY)
    if explicit_flavor is not None:
        return (
            explicit_flavor
            if isinstance(explicit_flavor, str)
            and explicit_flavor in SUPPORTED_AGENT_FLAVORS
            else None
        )

    if metadata.get(DEEPAGENTS_METADATA_KEY):
        return DEEPAGENTS_AGENT_FLAVOR
    if metadata.get(REACT_AGENT_METADATA_KEY):
        return LANGGRAPH_PREBUILT_AGENT_FLAVOR
    return None
