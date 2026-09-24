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

"""Utility functions for WildToolBench instrumentation."""

import json
from typing import Any, Optional


def safe_json_dumps(obj: Any, max_length: int = 10000) -> Optional[str]:
    """Safely serialize object to JSON string with length limit."""
    if obj is None:
        return None
    try:
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) > max_length:
            return s[:max_length] + "...(truncated)"
        return s
    except (TypeError, ValueError):
        return str(obj)[:max_length]
