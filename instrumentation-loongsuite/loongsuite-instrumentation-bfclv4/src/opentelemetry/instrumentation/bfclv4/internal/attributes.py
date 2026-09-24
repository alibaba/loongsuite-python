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

"""Constant attribute keys used by the BFCL v4 instrumentation."""

from __future__ import annotations

from typing import Final

FRAMEWORK_NAME: Final = "bfclv4"

# gen_ai.* attribute keys that are not exported by
# opentelemetry-semantic-conventions today.
GEN_AI_FRAMEWORK: Final = "gen_ai.framework"
GEN_AI_PROVIDER_NAME: Final = "gen_ai.provider.name"

# BFCL-specific (vendor) attribute keys.
BFCL_TEST_CATEGORY: Final = "bfcl.test_category"
BFCL_NUM_THREADS: Final = "bfcl.num_threads"
BFCL_TEST_CASE_COUNT: Final = "bfcl.test_case_count"
BFCL_RUN_IDS: Final = "bfcl.run_ids"
BFCL_TEST_ENTRY_ID: Final = "bfcl.test_entry_id"
BFCL_TURN_IDX: Final = "bfcl.turn_idx"
BFCL_QUERY_MODE: Final = "bfcl.query_mode"
BFCL_OSS_BACKEND: Final = "bfcl.oss.backend"
BFCL_TOOL_DURATION_IS_ESTIMATED: Final = "bfcl.tool.duration_is_estimated"
BFCL_TOOL_INDEX: Final = "bfcl.tool.index"
