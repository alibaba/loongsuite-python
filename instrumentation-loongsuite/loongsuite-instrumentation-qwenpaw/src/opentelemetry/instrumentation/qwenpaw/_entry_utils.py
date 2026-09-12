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

"""Build QwenPaw ``EntryInvocation`` objects and message attributes."""

from __future__ import annotations

from typing import Any

from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text


def _non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _attribute(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def parse_query_handler_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Return ``(msgs, request)`` from ``query_handler`` positional/kwargs."""
    msgs: Any = None
    request: Any = None
    if args:
        msgs = args[0]
        if len(args) > 1:
            request = args[1]
    if msgs is None and "msgs" in kwargs:
        msgs = kwargs["msgs"]
    if request is None:
        request = kwargs.get("request")
    return msgs, request


def parse_runtime_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Return the QwenPaw 2 ``AgentRequest`` passed to ``Runtime.run``."""

    if args:
        return args[0]
    return kwargs.get("request")


def _message_text(message: Any) -> str | None:
    if hasattr(message, "get_text_content"):
        return _non_empty_str(message.get_text_content())

    content = _attribute(message, "content")
    if isinstance(content, str):
        return _non_empty_str(content)
    if not isinstance(content, (list, tuple)):
        return None

    texts: list[str] = []
    for part in content:
        text = _non_empty_str(_attribute(part, "text"))
        if text:
            texts.append(text)
    return "\n".join(texts) or None


def input_messages_from_msgs(msgs: Any) -> list[InputMessage]:
    """Turn AgentScope / runtime message list into ``InputMessage`` entries."""
    if not msgs:
        return []
    if not isinstance(msgs, (list, tuple)):
        msgs = [msgs]
    out: list[InputMessage] = []
    for m in msgs:
        role = _enum_value(_attribute(m, "role")) or "user"
        text = _message_text(m)
        if text:
            out.append(
                InputMessage(
                    role=str(role),
                    parts=[Text(content=text)],
                )
            )
    return out


def input_messages_from_runtime_request(request: Any) -> list[InputMessage]:
    """Map QwenPaw 2 ``AgentRequest.input`` into GenAI input messages."""

    return input_messages_from_msgs(_attribute(request, "input"))


def output_message_from_yield_item(item: Any) -> OutputMessage | None:
    """If *item* is ``(Msg, last)`` with an assistant text message, map to output."""
    if not isinstance(item, tuple) or not item:
        return None
    msg = item[0]
    if msg is None:
        return None
    if getattr(msg, "role", None) != "assistant":
        return None
    if not hasattr(msg, "get_text_content"):
        return None
    text = msg.get_text_content()
    if not text:
        return None
    return OutputMessage(
        role="assistant",
        parts=[Text(content=text)],
        finish_reason="stop",
    )


def output_message_from_runtime_item(item: Any) -> OutputMessage | None:
    """Map a completed QwenPaw 2 response/message envelope."""

    candidates: list[Any]
    if _attribute(item, "object") == "response":
        output = _attribute(item, "output")
        candidates = list(output) if isinstance(output, (list, tuple)) else []
    else:
        candidates = [item]

    for message in reversed(candidates):
        role = _enum_value(_attribute(message, "role"))
        status = _enum_value(_attribute(message, "status"))
        if role != "assistant" or status != "completed":
            continue
        text = _message_text(message)
        if text:
            return OutputMessage(
                role="assistant",
                parts=[Text(content=text)],
                finish_reason="stop",
            )
    return None


def _entry_attributes(agent_id: str | None, channel: str | None) -> dict:
    extra_attrs: dict[str, Any] = {}
    if agent_id:
        extra_attrs["qwenpaw.agent_id"] = agent_id
        extra_attrs["copaw.agent_id"] = agent_id
    if channel:
        extra_attrs["qwenpaw.channel"] = channel
        extra_attrs["copaw.channel"] = channel
    return extra_attrs


def build_entry_invocation(
    instance: Any,
    msgs: Any,
    request: Any,
) -> EntryInvocation:
    """Populate ``EntryInvocation`` from runner instance and query_handler args."""
    session_id = None
    user_id = None
    channel = None
    if request is not None:
        session_id = _non_empty_str(_attribute(request, "session_id"))
        user_id = _non_empty_str(_attribute(request, "user_id"))
        channel = _non_empty_str(_attribute(request, "channel"))

    agent_id = _non_empty_str(getattr(instance, "agent_id", None))

    return EntryInvocation(
        session_id=session_id,
        user_id=user_id,
        input_messages=input_messages_from_msgs(msgs),
        attributes=_entry_attributes(agent_id, channel),
    )


def build_runtime_entry_invocation(
    instance: Any,
    request: Any,
) -> EntryInvocation:
    """Build an Entry invocation for QwenPaw 2 ``Runtime.run``."""

    workspace = getattr(instance, "workspace", None)
    agent_id = _non_empty_str(
        _attribute(request, "agent_id") or getattr(workspace, "agent_id", None)
    )
    channel = _non_empty_str(_attribute(request, "channel"))
    return EntryInvocation(
        session_id=_non_empty_str(_attribute(request, "session_id")),
        user_id=_non_empty_str(_attribute(request, "user_id")),
        input_messages=input_messages_from_runtime_request(request),
        attributes=_entry_attributes(agent_id, channel),
    )
