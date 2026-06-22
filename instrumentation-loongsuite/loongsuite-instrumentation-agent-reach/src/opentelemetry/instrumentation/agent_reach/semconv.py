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

# LoongSuite GenAI semantic-convention constants for Agent-Reach instrumentation.
# Reference: /home/admin/semantic-conventions/arms_docs/trace/gen-ai.md

GEN_AI_FRAMEWORK = "agent-reach"

# gen_ai.span.kind values
SPAN_KIND_ENTRY = "ENTRY"
SPAN_KIND_TOOL = "TOOL"

# gen_ai.operation.name values
OPERATION_ENTER = "enter"
OPERATION_EXECUTE_TOOL = "execute_tool"

# Common gen_ai.* attribute keys
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_FRAMEWORK_ATTR = "gen_ai.framework"
GEN_AI_SESSION_ID = "gen_ai.session.id"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

# Agent-Reach specific custom attributes (non-PII metadata for diagnostics)
AGENT_REACH_CHANNEL_TIER = "agent_reach.channel.tier"
AGENT_REACH_CHANNEL_ACTIVE_BACKEND = "agent_reach.channel.active_backend"
AGENT_REACH_CHANNEL_STATUS = "agent_reach.channel.status"
AGENT_REACH_PROBE_CMD = "agent_reach.probe.cmd"
AGENT_REACH_PROBE_STATUS = "agent_reach.probe.status"
AGENT_REACH_TRANSCRIBE_PROVIDER = "agent_reach.transcribe.provider"
AGENT_REACH_TRANSCRIBE_MODEL = "agent_reach.transcribe.model"
AGENT_REACH_TRANSCRIBE_SOURCE = "agent_reach.transcribe.source"
AGENT_REACH_TRANSCRIBE_CHUNK_COUNT = "agent_reach.transcribe.chunk_count"

# Truncation thresholds (chars). Sensitive outputs (probe stderr, channel
# messages) are truncated before going onto span attributes.
MESSAGE_TRUNCATE_CHARS = 200

# Channel check status values (per agent_reach base.Channel contract)
CHANNEL_STATUS_VALUES = ("ok", "warn", "off", "error")
# Probe status values (per agent_reach.probe.ProbeResult)
PROBE_STATUS_VALUES = ("ok", "missing", "broken", "timeout", "error")
