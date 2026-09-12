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

"""Constant attribute keys & framework identity used across wrappers."""

from __future__ import annotations

GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_SPAN_KIND = "gen_ai.span.kind"

FRAMEWORK_NAME = "openhands"

# OpenHands-specific span attributes (namespaced to avoid clashing with the
# generic GenAI semconv attributes already provided by upstream).
OH_INITIAL_MESSAGE_PREVIEW = "openhands.initial_message.preview"
