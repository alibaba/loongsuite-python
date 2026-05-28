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

# Prepare VitaBench and write a DashScope-backed model config.
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

mkdir -p /work/upstream
cd /work/upstream

if [ ! -d vitabench ]; then
  echo "[vita-setup] cloning vitabench"
  git clone --depth=1 https://github.com/meituan-longcat/vitabench.git
fi

cd vitabench
pip install --quiet --no-deps -e . || pip install --no-deps -e .
pip install --quiet "openai>=1.0" "pydantic>=2" pyyaml "loguru" "anthropic" \
  "litellm" "tenacity" "tiktoken" pandas toml addict deepdiff thefuzz \
  json_repair holidays || true

cat > /work/upstream/vitabench/models.yaml <<YAML
default:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
  temperature: 0.0
  max_input_tokens: 8192
  headers:
    Content-Type: "application/json"
    Authorization: "Bearer ${OPENAI_API_KEY}"
models:
  - name: qwen3.6-plus
    max_tokens: 1024
    max_input_tokens: 8192
YAML

echo "[vita-setup] done. config at /work/upstream/vitabench/models.yaml"
