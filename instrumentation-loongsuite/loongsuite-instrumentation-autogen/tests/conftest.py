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

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTIL_GENAI_SRC = _REPO_ROOT / "util" / "opentelemetry-util-genai" / "src"
if _UTIL_GENAI_SRC.is_dir() and str(_UTIL_GENAI_SRC) not in sys.path:
    sys.path.insert(0, str(_UTIL_GENAI_SRC))
    for _module_name in list(sys.modules):
        if (
            _module_name == "opentelemetry.util.genai"
            or _module_name.startswith("opentelemetry.util.genai.")
        ):
            del sys.modules[_module_name]

_AUTOGEN_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
if _AUTOGEN_PLUGIN_SRC.is_dir() and str(_AUTOGEN_PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(_AUTOGEN_PLUGIN_SRC))
