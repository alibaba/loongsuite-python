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

"""OpenTelemetry Hermes Agent instrumentation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from .instrumentor import HermesAgentInstrumentor
    from .metrics import HermesMetrics as _HermesMetrics
    from .wrappers import (
        LLMCallWrapper as _LLMCallWrapper,
    )
    from .wrappers import (
        RunConversationWrapper as _RunConversationWrapper,
    )
    from .wrappers import (
        ToolBatchWrapper as _ToolBatchWrapper,
    )
    from .wrappers import (
        ToolCallWrapper as _ToolCallWrapper,
    )
    from .wrappers import (
        ToolDispatchWrapper as _ToolDispatchWrapper,
    )
    from .wrappers import (
        ToolExecutionWrapper as _ToolExecutionWrapper,
    )
    from .wrappers import (
        finish_step as _finish_step,
    )
except ImportError:
    _PACKAGE_DIR = Path(__file__).resolve().parent

    def _load_local_module(module_name: str):
        full_name = f"{__name__}.{module_name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        spec = importlib.util.spec_from_file_location(
            full_name,
            _PACKAGE_DIR / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    _instrumentor = _load_local_module("instrumentor")
    _metrics = _load_local_module("metrics")
    _wrappers = _load_local_module("wrappers")

    HermesAgentInstrumentor = _instrumentor.HermesAgentInstrumentor
    _HermesMetrics = _metrics.HermesMetrics
    _LLMCallWrapper = _wrappers.LLMCallWrapper
    _RunConversationWrapper = _wrappers.RunConversationWrapper
    _ToolBatchWrapper = _wrappers.ToolBatchWrapper
    _ToolCallWrapper = _wrappers.ToolCallWrapper
    _ToolDispatchWrapper = _wrappers.ToolDispatchWrapper
    _ToolExecutionWrapper = _wrappers.ToolExecutionWrapper
    _finish_step = _wrappers.finish_step

__all__ = [
    "HermesAgentInstrumentor",
]
