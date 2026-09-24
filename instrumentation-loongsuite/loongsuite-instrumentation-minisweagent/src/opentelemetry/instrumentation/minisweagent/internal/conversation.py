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

"""Map mini-swe-agent trajectory dicts → OpenTelemetry GenAI message / tool-definition types."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)

_TRAJ_MAX_BYTES = 8_000_000


def bash_tool_definition() -> FunctionToolDefinition:
    """Single bash tool (same schema mini uses via LiteLLM)."""
    from minisweagent.models.utils.actions_toolcall import (
        BASH_TOOL,  # noqa: PLC0415
    )

    fn = BASH_TOOL["function"]
    return FunctionToolDefinition(
        name=fn["name"],
        description=fn.get("description"),
        parameters=fn.get("parameters") or {},
    )


def _text_parts(content: str | None) -> list[Text]:
    if content is None or str(content).strip() == "":
        return []
    return [Text(content=str(content))]


def _normalized_tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    parts: list[ToolCall] = []
    raw = msg.get("tool_calls")
    if raw:
        for tc in raw:
            fn_obj = getattr(tc, "function", None)
            if fn_obj is None and isinstance(tc, dict):
                fn_obj = tc.get("function")

            tc_id = getattr(tc, "id", None)
            if tc_id is None and isinstance(tc, dict):
                tc_id = tc.get("id")

            name = "bash"
            raw_args: Any = "{}"
            if fn_obj is not None:
                name = getattr(fn_obj, "name", None) or (
                    fn_obj.get("name") if isinstance(fn_obj, dict) else name
                )
                raw_args = getattr(fn_obj, "arguments", None)
                if raw_args is None and isinstance(fn_obj, dict):
                    raw_args = fn_obj.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args_obj = json.loads(raw_args)
                except json.JSONDecodeError:
                    args_obj = {"raw": raw_args}
            else:
                args_obj = raw_args if raw_args is not None else {}
            parts.append(
                ToolCall(
                    id=tc_id, name=str(name or "bash"), arguments=args_obj
                )
            )

    extra = msg.get("extra") or {}
    actions = extra.get("actions") or []
    if not raw and actions:
        for act in actions:
            cmd = act.get("command") if isinstance(act, dict) else None
            if cmd is None:
                continue
            parts.append(
                ToolCall(
                    id=act.get("tool_call_id")
                    if isinstance(act, dict)
                    else None,
                    name="bash",
                    arguments={"command": cmd},
                )
            )

    return parts


def split_system_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[Text], list[dict[str, Any]]]:
    sys_parts: list[Text] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            sys_parts.append(Text(content=str(m.get("content", ""))))
        else:
            rest.append(m)
    return sys_parts, rest


def _message_to_semconv_messages(
    msg: dict[str, Any],
) -> list[InputMessage | OutputMessage]:
    role = msg.get("role")
    if role == "user":
        return [
            InputMessage(role="user", parts=_text_parts(msg.get("content")))
        ]
    if role == "tool":
        tid = msg.get("tool_call_id")
        return [
            InputMessage(
                role="tool",
                parts=[
                    ToolCallResponse(
                        id=tid if isinstance(tid, str) else None,
                        response=msg.get("content", ""),
                    )
                ],
            )
        ]
    if role == "assistant":
        parts: list[Any] = []
        parts.extend(_text_parts(msg.get("content")))
        parts.extend(_normalized_tool_calls(msg))
        if not parts:
            parts = [Text(content="")]
        extra = msg.get("extra") or {}
        finish = (
            "tool_calls"
            if extra.get("actions") or msg.get("tool_calls")
            else "stop"
        )
        return [
            OutputMessage(
                role="assistant",
                parts=parts,
                finish_reason=finish,  # type: ignore[arg-type]
            )
        ]
    if role == "exit":
        return [
            InputMessage(
                role="user",
                parts=_text_parts(f"EXIT: {msg.get('content', '')}"),
            )
        ]
    return [
        InputMessage(
            role=str(role or "unknown"),
            parts=_text_parts(str(msg.get("content"))),
        ),
    ]


def build_invoke_payload_from_messages(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Core conversion: trajectory message dicts → invoke_agent / ENTRY payload."""
    sys_inst, rest = split_system_messages(messages)
    input_messages: list[InputMessage] = []
    output_messages: list[OutputMessage] = []

    try:
        for m in rest:
            for converted in _message_to_semconv_messages(m):
                if isinstance(converted, OutputMessage):
                    output_messages.append(converted)
                else:
                    input_messages.append(converted)
    except Exception:
        logger.debug("conversation serialization failed", exc_info=True)

    return {
        "system_instruction": sys_inst,
        "input_messages": input_messages,
        "output_messages": output_messages,
        "tool_definitions": [bash_tool_definition()],
    }


def build_invoke_agent_payload(agent: Any) -> dict[str, Any]:
    """Produce semantic fields from a DefaultAgent (or duck-typed agent) trajectory."""
    raw_messages = list(getattr(agent, "messages", None) or [])
    messages = [m for m in raw_messages if isinstance(m, dict)]
    return build_invoke_payload_from_messages(messages)


def try_fill_entry_payload_from_mini_trajectory() -> dict[str, Any] | None:
    """Read default mini trajectory file and build ENTRY / invoke payloads."""
    try:
        from minisweagent import global_config_dir  # noqa: PLC0415
    except Exception:
        return None

    path = Path(global_config_dir) / "last_mini_run.traj.json"
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > _TRAJ_MAX_BYTES:
            logger.warning(
                "trajectory too large for telemetry snapshot: %s", path
            )
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("failed to read mini trajectory %s", path, exc_info=True)
        return None

    msgs = data.get("messages")
    if not isinstance(msgs, list):
        return None
    dict_msgs = [m for m in msgs if isinstance(m, dict)]
    if not dict_msgs:
        return None
    try:
        return build_invoke_payload_from_messages(dict_msgs)
    except Exception:
        logger.debug("trajectory payload build failed", exc_info=True)
        return None


def apply_payload_to_entry_invocation(
    entry_inv: Any, payload: dict[str, Any]
) -> None:
    entry_inv.input_messages = payload["input_messages"]
    entry_inv.output_messages = payload["output_messages"]
    entry_inv.system_instruction = payload["system_instruction"]
    entry_inv.tool_definitions = payload["tool_definitions"]
