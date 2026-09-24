#!/usr/bin/env bash

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

# Run one VitaBench delivery task with LoongSuite instrumentation.
set -euo pipefail

export OTEL_SEMCONV_STABILITY_OPT_IN="${OTEL_SEMCONV_STABILITY_OPT_IN:-gen_ai_latest_experimental}"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="${OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT:-SPAN_ONLY}"

VITA_ROOT=/work/upstream/vitabench
if [ ! -d "$VITA_ROOT" ]; then
  echo "[vita-cmd] vitabench not found, run setup.sh first" >&2
  exit 1
fi

cd "$VITA_ROOT"
export VITA_MODEL_CONFIG_PATH=/work/upstream/vitabench/models.yaml

echo "[vita-cmd] invoking vita run --domain delivery --num-tasks 1"
loongsuite-instrument vita run \
  --domain delivery \
  --user-llm qwen3.6-plus \
  --agent-llm qwen3.6-plus \
  --evaluator-llm qwen3.6-plus \
  --num-tasks 1 \
  --num-trials 1
