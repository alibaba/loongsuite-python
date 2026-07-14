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

from types import SimpleNamespace

from opentelemetry.instrumentation.langchain.internal._agent_flavor import (
    AGENT_FLAVOR_METADATA_KEY,
    AGENT_FLAVOR_STEP_NODES,
    DEEPAGENTS_AGENT_FLAVOR,
    DEEPAGENTS_METADATA_KEY,
    DEERFLOW_AGENT_FLAVOR,
    LANGCHAIN_CREATE_AGENT_FLAVOR,
    LANGGRAPH_PREBUILT_AGENT_FLAVOR,
    REACT_AGENT_METADATA_KEY,
    get_agent_flavor,
)


def test_explicit_agent_flavor_overrides_legacy_markers():
    run = SimpleNamespace(
        metadata={
            AGENT_FLAVOR_METADATA_KEY: LANGCHAIN_CREATE_AGENT_FLAVOR,
            REACT_AGENT_METADATA_KEY: True,
            DEEPAGENTS_METADATA_KEY: True,
        }
    )

    assert get_agent_flavor(run) == LANGCHAIN_CREATE_AGENT_FLAVOR


def test_deepagents_marker_precedes_legacy_react_marker():
    run = SimpleNamespace(
        metadata={
            REACT_AGENT_METADATA_KEY: True,
            DEEPAGENTS_METADATA_KEY: True,
        }
    )

    assert get_agent_flavor(run) == DEEPAGENTS_AGENT_FLAVOR


def test_legacy_react_marker_maps_to_langgraph_prebuilt():
    run = SimpleNamespace(metadata={REACT_AGENT_METADATA_KEY: True})

    assert get_agent_flavor(run) == LANGGRAPH_PREBUILT_AGENT_FLAVOR


def test_unknown_explicit_agent_flavor_does_not_fall_back():
    run = SimpleNamespace(
        metadata={
            AGENT_FLAVOR_METADATA_KEY: "unknown",
            REACT_AGENT_METADATA_KEY: True,
        }
    )

    assert get_agent_flavor(run) is None


def test_non_string_explicit_agent_flavor_does_not_fall_back():
    run = SimpleNamespace(
        metadata={
            AGENT_FLAVOR_METADATA_KEY: ["deerflow"],
            REACT_AGENT_METADATA_KEY: True,
        }
    )

    assert get_agent_flavor(run) is None


def test_deerflow_explicit_flavor_uses_model_step_node():
    run = SimpleNamespace(
        metadata={AGENT_FLAVOR_METADATA_KEY: DEERFLOW_AGENT_FLAVOR}
    )

    assert get_agent_flavor(run) == DEERFLOW_AGENT_FLAVOR
    assert AGENT_FLAVOR_STEP_NODES[DEERFLOW_AGENT_FLAVOR] == "model"
