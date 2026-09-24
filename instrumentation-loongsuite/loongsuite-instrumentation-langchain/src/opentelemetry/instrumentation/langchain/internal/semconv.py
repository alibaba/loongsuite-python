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

"""
Semantic convention attributes for LangChain instrumentation.

Re-exports attributes from ``util-genai`` extended semconv so that the
plugin and its tests have a single import source.
"""

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (  # noqa: E501
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY_TEXT,
    GEN_AI_SPAN_KIND,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
)

GEN_AI_OPERATION_NAME = GenAI.GEN_AI_OPERATION_NAME

GEN_AI_TOOL_CALL_ID = GenAI.GEN_AI_TOOL_CALL_ID

__all__ = [
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_RETRIEVAL_DOCUMENTS",
    "GEN_AI_RETRIEVAL_QUERY_TEXT",
    "GEN_AI_SPAN_KIND",
    "GEN_AI_TOOL_CALL_ARGUMENTS",
    "GEN_AI_TOOL_CALL_RESULT",
    "INPUT_VALUE",
    "OUTPUT_VALUE",
]

# Input/Output attributes (used for Chain spans)
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
