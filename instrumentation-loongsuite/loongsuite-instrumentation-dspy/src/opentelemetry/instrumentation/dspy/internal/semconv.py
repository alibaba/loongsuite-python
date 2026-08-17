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
Semantic convention attributes for DSPy instrumentation.

Re-exports attributes from ``util-genai`` extended semconv so that the
plugin and its tests have a single import source.
"""

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (  # noqa: E501
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GenAiSpanKindValues,
)

GEN_AI_OPERATION_NAME = GenAI.GEN_AI_OPERATION_NAME
GEN_AI_REQUEST_MODEL = GenAI.GEN_AI_REQUEST_MODEL
GEN_AI_USAGE_INPUT_TOKENS = GenAI.GEN_AI_USAGE_INPUT_TOKENS
GEN_AI_USAGE_OUTPUT_TOKENS = GenAI.GEN_AI_USAGE_OUTPUT_TOKENS

GEN_AI_FRAMEWORK = "gen_ai.framework"
FRAMEWORK_NAME = "dspy"

# Input/Output attributes (used for Chain spans)
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"

# ``GenAiSpanKindValues`` has no CHAIN member, so the literal is required.
SPAN_KIND_CHAIN = "CHAIN"

# ``gen_ai.operation.name`` for Chain spans: composite programs are workflows,
# leaf predictors are tasks.
CHAIN_OPERATION_WORKFLOW = "workflow"
CHAIN_OPERATION_TASK = "task"

__all__ = [
    "CHAIN_OPERATION_TASK",
    "CHAIN_OPERATION_WORKFLOW",
    "FRAMEWORK_NAME",
    "GEN_AI_FRAMEWORK",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REACT_FINISH_REASON",
    "GEN_AI_REACT_ROUND",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_SPAN_KIND",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_TOTAL_TOKENS",
    "GenAiSpanKindValues",
    "INPUT_VALUE",
    "OUTPUT_VALUE",
    "SPAN_KIND_CHAIN",
]
