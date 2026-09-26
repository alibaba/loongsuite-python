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

import pytest

from opentelemetry.instrumentation.langchain.internal._agent_semantics import (
    AGENT_FRAMEWORK_METADATA_KEY,
    AGENT_STEP_NODE_METADATA_KEY,
    get_agent_semantics,
)


def test_agent_semantics_requires_both_scalar_fields():
    run = SimpleNamespace(
        metadata={
            AGENT_FRAMEWORK_METADATA_KEY: "deerflow",
            AGENT_STEP_NODE_METADATA_KEY: "model",
        }
    )

    assert get_agent_semantics(run) == ("deerflow", "model")


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {AGENT_FRAMEWORK_METADATA_KEY: "deerflow"},
        {AGENT_STEP_NODE_METADATA_KEY: "model"},
        {
            AGENT_FRAMEWORK_METADATA_KEY: ["deerflow"],
            AGENT_STEP_NODE_METADATA_KEY: "model",
        },
        {
            AGENT_FRAMEWORK_METADATA_KEY: "deerflow",
            AGENT_STEP_NODE_METADATA_KEY: "",
        },
    ],
)
def test_agent_semantics_rejects_incomplete_or_non_string_metadata(metadata):
    assert get_agent_semantics(SimpleNamespace(metadata=metadata)) is None
