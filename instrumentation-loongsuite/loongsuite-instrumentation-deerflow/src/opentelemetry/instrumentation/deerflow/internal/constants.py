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

"""Constants shared by DeerFlow patch helpers."""

AGENT_FLAVOR_ATTR = "_loongsuite_agent_flavor"
DEERFLOW_AGENT_FLAVOR = "deerflow"

GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_SESSION_ID = "gen_ai.session.id"
GEN_AI_USER_ID = "gen_ai.user.id"

DEERFLOW_ASSISTANT_ID = "deerflow.assistant.id"
DEERFLOW_RUN_ID = "deerflow.run.id"
DEERFLOW_RUN_STATUS = "deerflow.run.status"
DEERFLOW_TRACE_ID = "deerflow.trace.id"

DEERFLOW_TRACE_METADATA_KEY = "deerflow_trace_id"

CREATE_AGENT_ALIASES = (
    ("deerflow.agents.factory", "create_agent"),
    ("deerflow.agents.lead_agent.agent", "create_agent"),
    ("deerflow.client", "create_agent"),
    ("deerflow.subagents.executor", "create_agent"),
)

GATEWAY_RUN_AGENT_ALIASES = (
    ("deerflow.runtime.runs.worker", "run_agent"),
    ("deerflow.runtime.runs", "run_agent"),
    ("deerflow.runtime", "run_agent"),
)
GATEWAY_LOADED_RUN_AGENT_ALIASES = (("app.gateway.services", "run_agent"),)
CLIENT_STREAM = (
    "deerflow.client",
    "DeerFlowClient.stream",
)
