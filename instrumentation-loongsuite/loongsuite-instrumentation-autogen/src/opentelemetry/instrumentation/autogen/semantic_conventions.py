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

from typing import Final


class GenAISpanKind:
    AGENT = "AGENT"
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    ENTRY = "ENTRY"


class GenAIOperation:
    CHAT = "chat"
    CREATE_AGENT = "create_agent"
    EXECUTE_TOOL = "execute_tool"
    GENERATE_CONTENT = "generate_content"
    INVOKE_AGENT = "invoke_agent"
    TEXT_COMPLETION = "text_completion"


AUTOGEN_PROVIDER_NAME: Final = "autogen"

GEN_AI_AGENT_DESCRIPTION: Final = "gen_ai.agent.description"
GEN_AI_AGENT_ID: Final = "gen_ai.agent.id"
GEN_AI_AGENT_NAME: Final = "gen_ai.agent.name"
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME: Final = "gen_ai.provider.name"
GEN_AI_SPAN_KIND: Final = "gen_ai.span.kind"
GEN_AI_SYSTEM: Final = "gen_ai.system"
GEN_AI_TOOL_CALL_ID: Final = "gen_ai.tool.call.id"
GEN_AI_TOOL_DESCRIPTION: Final = "gen_ai.tool.description"
GEN_AI_TOOL_NAME: Final = "gen_ai.tool.name"
