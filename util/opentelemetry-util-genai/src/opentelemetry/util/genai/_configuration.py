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

import os
from functools import lru_cache

OTEL_SEMCONV_STABILITY_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"


@lru_cache(maxsize=1)
def is_experimental_mode() -> bool:
    """Snapshot the GenAI opt-in on first use; configure before SDK startup.

    Keep the instrumentation opt-in token and comma-separated parsing without
    importing its private initialization state. Runtime changes are unsupported.
    """
    return "gen_ai_latest_experimental" in {
        value.strip()
        for value in os.environ.get(OTEL_SEMCONV_STABILITY_OPT_IN, "").split(
            ","
        )
    }
