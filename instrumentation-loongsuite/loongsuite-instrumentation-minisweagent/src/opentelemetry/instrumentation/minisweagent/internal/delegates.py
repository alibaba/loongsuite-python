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

"""Tracing delegates for Environment (factory-injected wrappers).

LLM-call spans remain with LiteLLM/OpenAI instrumentation; this emits execute_tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)


def _sanitize_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(payload, default=str))
    except (TypeError, ValueError):
        logger.debug("tool result not JSON-normalizable", exc_info=True)
        try:
            return {"repr": repr(payload)}
        except Exception:
            return {"error": "unserializable_tool_result"}


class TracingEnvironment:
    """Delegates to inner Environment and emits ARMS-aligned TOOL (execute_tool) spans."""

    __slots__ = ("_inner", "_tracer")

    def __init__(self, inner: Any, tracer: Tracer):  # noqa: ARG002
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tracer", tracer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def execute(
        self, action: dict, cwd: str = "", **kwargs: Any
    ) -> dict[str, Any]:
        from minisweagent.exceptions import InterruptAgentFlow  # noqa: PLC0415

        from opentelemetry.util.genai.extended_handler import (
            get_extended_telemetry_handler,  # noqa: PLC0415
        )
        from opentelemetry.util.genai.extended_types import (
            ExecuteToolInvocation,  # noqa: PLC0415
        )
        from opentelemetry.util.genai.types import (
            Error as GenAIError,  # noqa: PLC0415
        )

        command = action.get("command", "") if isinstance(action, dict) else ""
        tool_call_id = (
            action.get("tool_call_id") if isinstance(action, dict) else None
        )
        han = get_extended_telemetry_handler()
        inv = ExecuteToolInvocation(
            tool_name="bash",
            provider="minisweagent",
            tool_type="function",
            tool_call_id=tool_call_id
            if isinstance(tool_call_id, str)
            else None,
            tool_description="Execute a bash command",
            tool_call_arguments={"command": command},
        )

        han.start_execute_tool(inv, context=context_api.get_current())
        try:
            result = self._inner.execute(action, cwd, **kwargs)
        except InterruptAgentFlow:
            inv.tool_call_result = {"interrupted": "InterruptAgentFlow"}
            han.stop_execute_tool(inv)
            raise
        except Exception as exc:
            inv.tool_call_result = {"error": str(exc)}
            han.fail_execute_tool(
                inv, GenAIError(message=str(exc), type=type(exc))
            )
            raise

        if isinstance(result, dict):
            payload_out = dict(result)
        else:
            payload_out = {"value": result}
        inv.tool_call_result = _sanitize_tool_result(payload_out)
        han.stop_execute_tool(inv)
        return result
