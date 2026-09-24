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

"""Wrappers for the OpenHands **V0** (Legacy CodeAct) architecture.

Trace tree
----------

::

    ENTRY enter openhands                       (openhands.core.main.run_controller)
    `-- AGENT invoke_agent codeact              (openhands.core.loop.run_agent_until_done)
        |-- STEP react step [×N]                (openhands.controller.agent_controller.AgentController._step)
        |   `-- LLM chat {model}                (litellm — covered by litellm instrumentor)
        `-- TOOL execute_tool {tool_name}       (openhands.runtime.base.Runtime.run_action)

Context propagation across threads
----------------------------------

OpenHands V0's ``EventStream`` delivers events via ``ThreadPoolExecutor``,
and ``AgentController.on_event`` then runs the actual handler with a
*brand-new* asyncio loop in a worker thread:

.. code:: python

    asyncio.get_event_loop().run_until_complete(self._on_event(event))

Python ``contextvars`` do NOT propagate from the main coroutine into these
worker threads, so ``AgentController._step`` and ``Runtime.run_action``
would otherwise start *root* spans with fresh ``trace_id``s, fragmenting
the trace into many disconnected pieces.

To fix that, we use :mod:`session_context` as a process-wide bridge: the
ENTRY wrapper stashes the OTel context (carrying the ENTRY+AGENT span
chain) keyed by session id, and STEP / TOOL wrappers re-attach it before
opening their span. The result is one trace per session id with the
correct parent-child links.

I/O capture
-----------

STEP spans set:

* ``input.value`` and ``output.value`` (OpenInference convention)
* ``input.mime_type`` / ``output.mime_type``
* ``gen_ai.input.messages`` / ``gen_ai.output.messages`` where the GenAI
  semconv applies (LLM-style messages + assistant tool calls)

ENTRY and AGENT spans set GenAI message attributes only — they never
emit OpenInference ``input.value`` / ``output.value`` mirrors.

TOOL spans set ``gen_ai.tool.call.arguments`` (always, including ``"{}"``
when empty) and ``gen_ai.tool.call.result`` for observations. They do
not set OpenInference ``input.value`` / ``output.value``.

Capture is always on and content is emitted untruncated.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.openhands.config import (
    OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS,
)
from opentelemetry.instrumentation.openhands.internal.constants import (
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_SPAN_KIND,
    OH_INITIAL_MESSAGE_PREVIEW,
)
from opentelemetry.instrumentation.openhands.internal.session_context import (
    AttachedSession,
    clear_context,
    get_context,
    get_tool_definition,
    store_context,
    store_tool_registry,
)
from opentelemetry.instrumentation.openhands.internal.utils import (
    action_to_genai_output,
    maybe_preview,
    safe_get_attr,
    safe_str,
    serialize_message,
    to_json_str,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
)

logger = logging.getLogger(__name__)


# Constants -----------------------------------------------------------------

OH_AGENT_NAME = "openhands.agent.name"
OH_REACT_ROUND = "gen_ai.react.round"
OH_AGENT_STATE = "openhands.agent.state"
OH_RUNTIME_NAME = "openhands.runtime.name"
OH_ACTION_TYPE = "openhands.action.type"
OH_OBSERVATION_TYPE = "openhands.observation.type"
OH_HISTORY_LENGTH = "openhands.history.length"

# OpenInference / GenAI common I/O attribute keys
INPUT_VALUE = "input.value"
INPUT_MIME = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME = "output.mime_type"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_SESSION_ID = "gen_ai.session.id"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

# Tool span attributes per ARMS GenAI semconv (gen-ai.md §Tool).
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

# Stash slots on AgentController instances (set by AgentControllerInitWrapper).
_OWNS_FLAG = "_otel_oh_owns_lifecycle"
_ENTRY_SPAN_ATTR = "_otel_oh_entry_span"
_AGENT_SPAN_ATTR = "_otel_oh_agent_span"
_ENTRY_TOKEN_ATTR = "_otel_oh_entry_token"
_AGENT_TOKEN_ATTR = "_otel_oh_agent_token"
# STEP persistence — keeps the *most-recent* STEP span alive across the
# return of ``_step`` so that ``Runtime.run_action`` (which fires *later*
# in a thread-pool executor via ``call_sync_from_async``) can re-attach
# the STEP context and become its child rather than a sibling.
#
# IMPORTANT: we deliberately do **not** stash an OTel attach-token across
# the return of ``_step``. ``otel_context.attach()`` returns a Token that
# is bound to the ``contextvars.Context`` it was created in; calling
# ``detach(token)`` from a *different* context raises ``ValueError`` (and
# in production the Aliyun OTel SDK floods the log with
# "Token was created in a different Context" errors).  Attach/detach
# always happen as a balanced pair *inside the same async task*; cross-
# task / cross-thread propagation goes through the ``Context`` *object*
# stashed in :mod:`session_context` and re-attached on the consumer side.
_STEP_SPAN_ATTR = "_otel_oh_step_span"
_AGENT_CTX_ATTR = "_otel_oh_agent_ctx"  # restore target when STEP closes


def _set_common(span: trace_api.Span, kind: str) -> None:
    span.set_attribute(GEN_AI_SPAN_KIND, kind)
    span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
    span.set_attribute(GEN_AI_SYSTEM, FRAMEWORK_NAME)


def _set_io(
    span: trace_api.Span,
    *,
    input_value: str = "",
    output_value: str = "",
    input_messages: str = "",
    output_messages: str = "",
    mime: str = "application/json",
) -> None:
    if input_value:
        span.set_attribute(INPUT_VALUE, input_value)
        span.set_attribute(INPUT_MIME, mime)
    if output_value:
        span.set_attribute(OUTPUT_VALUE, output_value)
        span.set_attribute(OUTPUT_MIME, mime)
    if input_messages:
        span.set_attribute(GEN_AI_INPUT_MESSAGES, input_messages)
    if output_messages:
        span.set_attribute(GEN_AI_OUTPUT_MESSAGES, output_messages)


def _extract_model_from_config(config: Any) -> str:
    if config is None:
        return ""
    try:
        llms = safe_get_attr(config, "llms")
        if isinstance(llms, dict) and llms:
            llm = next(iter(llms.values()))
            model = safe_get_attr(llm, "model")
            if model:
                return safe_str(model)
    except Exception:
        pass
    try:
        llm = safe_get_attr(config, "llm")
        model = safe_get_attr(llm, "model")
        if model:
            return safe_str(model)
    except Exception:
        pass
    return ""


def _extract_input_message_text(initial_user_action: Any) -> str:
    """Pull human-readable text out of an ``initial_user_action`` argument."""
    return serialize_message(initial_user_action)


def _state_to_input_messages(state: Any, max_messages: int = 10) -> str:
    """Best-effort extract a chat-style messages list from a controller State.

    The actual messages sent to the LLM are built inside ``CodeActAgent.step``
    and not stored on the controller, so this is a coarse summary derived
    from ``state.history`` which is reliably available.
    """
    history = safe_get_attr(state, "history") or []
    if not isinstance(history, list):
        return ""
    items: list[dict[str, str]] = []
    # Keep the most recent ``max_messages`` events for size budget.
    for ev in history[-max_messages:]:
        cls_name = type(ev).__name__
        # Map common event types to roles
        if cls_name in ("MessageAction", "SystemMessageAction"):
            role = (
                "user"
                if str(safe_get_attr(ev, "source")) == "user"
                else "assistant"
            )
            content = (
                safe_get_attr(ev, "content")
                or safe_get_attr(ev, "message")
                or ""
            )
        elif cls_name.endswith("Action"):
            role = "assistant"
            content = (
                safe_get_attr(ev, "thought")
                or safe_get_attr(ev, "command")
                or safe_get_attr(ev, "code")
                or safe_str(ev)
            )
        elif cls_name.endswith("Observation"):
            role = "tool"
            content = safe_get_attr(ev, "content") or safe_str(ev)
        else:
            role = "system"
            content = safe_str(ev)
        items.append(
            {"role": role, "content": safe_str(content), "event": cls_name}
        )
    return to_json_str(items)


def _final_state_to_output(state: Any) -> str:
    """Serialize the controller's final state for output.value."""
    if state is None:
        return ""
    payload: dict[str, Any] = {}
    agent_state = safe_get_attr(state, "agent_state")
    if agent_state is not None:
        payload["agent_state"] = safe_get_attr(
            agent_state, "value"
        ) or safe_str(agent_state)
    last_error = safe_get_attr(state, "last_error")
    if last_error:
        payload["last_error"] = safe_str(last_error)
    iteration = safe_get_attr(state, "iteration")
    if iteration is not None:
        payload["iteration"] = safe_str(iteration)
    history = safe_get_attr(state, "history") or []
    if isinstance(history, list) and history:
        payload["history_length"] = len(history)
        # Find the last AgentFinishAction or last assistant content for a final answer summary.
        for ev in reversed(history):
            if type(ev).__name__ == "AgentFinishAction":
                payload["final_thought"] = safe_str(
                    safe_get_attr(ev, "final_thought")
                    or safe_get_attr(ev, "thought")
                    or ""
                )
                payload["outputs"] = safe_str(
                    safe_get_attr(ev, "outputs") or {}
                )
                break
    return to_json_str(payload)


def _entry_input_messages_from_initial(initial_user_action: Any) -> str:
    """Return ARMS gen_ai.input.messages for the ENTRY span."""
    text = _extract_input_message_text(initial_user_action)
    if not text:
        return ""
    return to_json_str(
        [{"role": "user", "parts": [{"type": "text", "content": text}]}]
    )


def _entry_io_from_state(state: Any) -> tuple[str, str]:
    """Return (input_messages, output_messages) for ENTRY from final state."""
    history = safe_get_attr(state, "history") or []
    input_messages = ""
    output_messages = ""
    if isinstance(history, list) and history:
        input_payload = _history_to_input_messages_schema(history)
        if input_payload:
            input_messages = to_json_str(input_payload)
        output_payload = _history_to_output_messages_schema(history)
        if output_payload:
            output_messages = to_json_str(output_payload)
    if not output_messages:
        final_state = _final_state_to_output(state)
        if final_state:
            output_messages = to_json_str(
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": final_state}],
                        "finish_reason": "stop",
                    }
                ]
            )
    return input_messages, output_messages


# ---------------------------------------------------------------------------
# ARMS GenAI semconv message-schema converters.
#
# Per gen-ai.md §LLM/§Agent, gen_ai.input.messages / gen_ai.output.messages
# / gen_ai.system_instructions follow a "parts"-based structure:
#
#     [{"role": "user|assistant|tool|system",
#       "parts": [{"type": "text|tool_call|tool_call_response|...",
#                  "content": "...", "name": "...", "id": "...",
#                  "arguments": {...}, "result": "..."}],
#       "finish_reason": "stop|...",        # output only
#     }]
#
# The system instructions schema is a flat list of parts:
#
#     [{"type": "text", "content": "..."}]
# ---------------------------------------------------------------------------


def _action_event_to_parts(ev: Any) -> list[dict[str, Any]]:
    """Convert an Action event into a list of ``parts`` for AGENT messages.

    Captures both the model's "thought" text and any ``tool_call`` part
    derived from ``tool_call_metadata``.
    """
    parts: list[dict[str, Any]] = []
    thought = safe_get_attr(ev, "thought")
    if thought:
        parts.append({"type": "text", "content": safe_str(thought)})
    tcm = safe_get_attr(ev, "tool_call_metadata")
    if tcm is not None:
        fn_name = safe_str(safe_get_attr(tcm, "function_name") or "")
        tcid = safe_str(safe_get_attr(tcm, "tool_call_id") or "")
        # Best-effort harvest the original LLM-emitted JSON arguments.
        args: Any = {}
        try:
            mr = safe_get_attr(tcm, "model_response")
            choices = (
                getattr(mr, "choices", None) if mr is not None else None
            ) or []
            for choice in choices:
                msg = getattr(choice, "message", None) or (
                    choice.get("message") if isinstance(choice, dict) else None
                )
                tool_calls = (
                    getattr(msg, "tool_calls", None)
                    if msg is not None
                    else None
                ) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
                if not tool_calls:
                    continue
                for tc in tool_calls:
                    tc_id = (
                        getattr(tc, "id", None)
                        if not isinstance(tc, dict)
                        else tc.get("id")
                    )
                    if tcid and safe_str(tc_id) != tcid:
                        continue
                    fn = (
                        getattr(tc, "function", None)
                        if not isinstance(tc, dict)
                        else tc.get("function")
                    )
                    raw = (
                        getattr(fn, "arguments", None)
                        if not isinstance(fn, dict)
                        else fn.get("arguments")
                    )
                    if isinstance(raw, str):
                        try:
                            import json as _json

                            args = _json.loads(raw)
                        except Exception:
                            args = {"raw": raw}
                    elif isinstance(raw, dict):
                        args = raw
        except Exception:
            args = {}
        if not args:
            for key in (
                "command",
                "code",
                "path",
                "url",
                "content",
                "task_list",
                "old_str",
                "new_str",
                "file_text",
            ):
                v = safe_get_attr(ev, key)
                if v not in (None, "", [], {}):
                    args[key] = v
        if fn_name or tcid or args:
            parts.append(
                {
                    "type": "tool_call",
                    "id": tcid,
                    "name": fn_name
                    or safe_str(safe_get_attr(ev, "action") or ""),
                    "arguments": args,
                }
            )
    if not parts:
        # Minimal fallback when nothing else could be extracted.
        action_type = safe_str(safe_get_attr(ev, "action") or "")
        if action_type:
            parts.append(
                {"type": "tool_call", "name": action_type, "arguments": {}}
            )
    return parts


def _observation_event_to_parts(ev: Any) -> list[dict[str, Any]]:
    """Convert an Observation event into ``parts`` for tool-response messages."""
    tcm = safe_get_attr(ev, "tool_call_metadata")
    tcid = safe_str(safe_get_attr(tcm, "tool_call_id") or "") if tcm else ""
    result_payload: dict[str, Any] = {}
    for key in ("content", "exit_code", "error", "stdout", "stderr", "url"):
        v = safe_get_attr(ev, key)
        if v not in (None, "", [], {}):
            result_payload[key] = v
    return [
        {
            "type": "tool_call_response",
            "id": tcid,
            "result": result_payload or safe_str(ev),
        }
    ]


def _history_to_input_messages_schema(
    history: list, max_events: int = 200
) -> list[dict[str, Any]]:
    """Convert ``state.history`` into the ARMS gen_ai.input.messages schema.

    Folds adjacent same-role events into a single message with multiple
    ``parts``, mirroring how the messages were assembled when sent to
    the LLM.
    """
    if not history:
        return []
    items = history[-max_events:]
    messages: list[dict[str, Any]] = []
    for ev in items:
        cls = type(ev).__name__
        # Determine role + parts for this event.
        if cls == "SystemMessageAction":
            # System is reported separately under gen_ai.system_instructions.
            continue
        if cls == "MessageAction":
            src = str(safe_get_attr(ev, "source") or "").lower()
            role = "user" if src == "user" else "assistant"
            content = safe_str(safe_get_attr(ev, "content") or "")
            parts = [{"type": "text", "content": content}]
        elif cls.endswith("Observation"):
            role = "tool"
            parts = _observation_event_to_parts(ev)
        elif cls.endswith("Action"):
            role = "assistant"
            parts = _action_event_to_parts(ev)
        else:
            role = "system"
            parts = [{"type": "text", "content": safe_str(ev)}]
        # Fold consecutive same-role messages.
        if messages and messages[-1]["role"] == role:
            messages[-1]["parts"].extend(parts)
        else:
            messages.append({"role": role, "parts": parts})
    return messages


def _history_to_output_messages_schema(history: list) -> list[dict[str, Any]]:
    """Pull the *final* assistant turn from history per ARMS gen_ai.output.messages.

    Walks back from the end of history and collects assistant-side events
    (Actions) up to the previous user/tool boundary. Includes a
    ``finish_reason`` derived from the last AgentFinishAction / state.
    """
    if not history:
        return []
    finish_reason = "stop"
    tail_actions: list[Any] = []
    for ev in reversed(history):
        cls = type(ev).__name__
        if cls == "AgentFinishAction":
            finish_reason = safe_str(
                safe_get_attr(ev, "final_thought") and "stop" or "stop"
            )
            tail_actions.insert(0, ev)
            continue
        if cls.endswith("Observation") or cls == "MessageAction":
            # Stop once we cross back into user-input or tool-result territory.
            if (
                cls == "MessageAction"
                and str(safe_get_attr(ev, "source") or "").lower() == "user"
            ):
                break
            if cls.endswith("Observation"):
                break
        if cls.endswith("Action") or (
            cls == "MessageAction"
            and str(safe_get_attr(ev, "source") or "").lower() != "user"
        ):
            tail_actions.insert(0, ev)
    if not tail_actions:
        # Fallback: at least include the very last event as the assistant turn.
        tail_actions = [history[-1]]
    parts: list[dict[str, Any]] = []
    for ev in tail_actions:
        cls = type(ev).__name__
        if cls == "MessageAction":
            content = safe_str(safe_get_attr(ev, "content") or "")
            if content:
                parts.append({"type": "text", "content": content})
        elif cls == "AgentFinishAction":
            ft = safe_str(safe_get_attr(ev, "final_thought") or "")
            if ft:
                parts.append({"type": "text", "content": ft})
            outputs = safe_get_attr(ev, "outputs")
            if outputs:
                parts.append({"type": "text", "content": safe_str(outputs)})
        else:
            parts.extend(_action_event_to_parts(ev))
    if not parts:
        parts = [{"type": "text", "content": ""}]
    return [
        {"role": "assistant", "parts": parts, "finish_reason": finish_reason}
    ]


def _agent_to_system_instructions(
    agent: Any, state: Any
) -> list[dict[str, Any]]:
    """Return ARMS gen_ai.system_instructions for the controller's agent.

    Tries the explicit ``agent.get_system_message()`` API first (most
    accurate), then falls back to scanning ``state.history`` for a
    ``SystemMessageAction``.
    """
    content = ""
    try:
        gsm = safe_get_attr(agent, "get_system_message")
        if callable(gsm):
            sm = gsm()
            content = safe_str(safe_get_attr(sm, "content") or "")
    except Exception:
        content = ""
    if not content:
        history = safe_get_attr(state, "history") or []
        if isinstance(history, list):
            for ev in history:
                if type(ev).__name__ == "SystemMessageAction":
                    content = safe_str(safe_get_attr(ev, "content") or "")
                    if content:
                        break
    if not content:
        return []
    return [{"type": "text", "content": content}]


# ---------------------------------------------------------------------------
# ENTRY: openhands.core.main.run_controller
# ---------------------------------------------------------------------------


class RunControllerWrapper:
    """ENTRY span around the V0 CLI/headless ``run_controller`` coroutine.

    Stashes the active OTel Context (with the ENTRY span attached) keyed
    by ``sid`` so STEP / TOOL spans firing in worker threads can re-attach
    it and remain in the same trace.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        return self._impl(wrapped, instance, args, kwargs)

    async def _impl(self, wrapped, instance, args, kwargs):
        if not OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS:
            return await wrapped(*args, **kwargs)

        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        initial_user_action = kwargs.get("initial_user_action")
        if initial_user_action is None and len(args) >= 2:
            initial_user_action = args[1]
        sid = kwargs.get("sid")
        if sid is None and len(args) >= 3:
            sid = args[2]
        # When sid wasn't passed, we don't yet know the auto-generated one;
        # the controller will publish ``controller.id`` later. We update
        # the stash again from inside the AGENT wrapper.

        span = self._tracer.start_span(
            "enter openhands", kind=SpanKind.INTERNAL
        )
        _set_common(span, "ENTRY")
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "enter")
        if sid:
            span.set_attribute(GEN_AI_SESSION_ID, safe_str(sid))
            span.set_attribute(GEN_AI_CONVERSATION_ID, safe_str(sid))
        model = _extract_model_from_config(config)
        if model:
            span.set_attribute(GEN_AI_REQUEST_MODEL, model)

        input_text = _extract_input_message_text(initial_user_action)
        preview = maybe_preview(input_text)
        if preview:
            span.set_attribute(OH_INITIAL_MESSAGE_PREVIEW, preview)
        if input_text:
            entry_input_messages = _entry_input_messages_from_initial(
                initial_user_action
            )
            if entry_input_messages:
                _set_io(
                    span,
                    input_messages=entry_input_messages,
                )

        ctx = set_span_in_context(span)
        token = otel_context.attach(ctx)
        if sid:
            store_context(sid, ctx)
        try:
            try:
                result = await wrapped(*args, **kwargs)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, type(exc).__qualname__)
                )
                raise
            try:
                entry_input_messages, entry_output_messages = (
                    _entry_io_from_state(result)
                )
                if entry_input_messages or entry_output_messages:
                    _set_io(
                        span,
                        input_messages=entry_input_messages,
                        output_messages=entry_output_messages,
                    )
                agent_state = safe_get_attr(result, "agent_state")
                if agent_state is not None:
                    span.set_attribute(
                        OH_AGENT_STATE,
                        safe_get_attr(agent_state, "value")
                        or safe_str(agent_state),
                    )
            except Exception:
                pass
            return result
        finally:
            try:
                otel_context.detach(token)
            except Exception:
                pass
            if sid:
                clear_context(sid)
            span.end()


# ---------------------------------------------------------------------------
# AGENT: openhands.core.loop.run_agent_until_done
# ---------------------------------------------------------------------------


class RunAgentUntilDoneWrapper:
    """AGENT span around the V0 polling loop.

    Re-attaches the ENTRY context (in case asyncio task creation didn't
    propagate it for some reason) and re-stashes a fresh context that now
    also includes the AGENT span — that's what STEP / TOOL re-attach.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        return self._impl(wrapped, instance, args, kwargs)

    async def _impl(self, wrapped, instance, args, kwargs):
        if not OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS:
            return await wrapped(*args, **kwargs)

        controller = kwargs.get("controller")
        if controller is None and args:
            controller = args[0]
        agent = safe_get_attr(controller, "agent")
        agent_name = safe_get_attr(agent, "name") or "codeact"
        agent_class = (
            f"{type(agent).__module__}.{type(agent).__name__}" if agent else ""
        )
        sid = safe_str(safe_get_attr(controller, "id") or "")
        llm = safe_get_attr(agent, "llm")
        llm_config = safe_get_attr(llm, "config")
        model = safe_get_attr(llm_config, "model") or safe_get_attr(
            llm, "model"
        )

        # If AgentController.__init__ already opened lifecycle-bound ENTRY+AGENT
        # spans, do not create a second AGENT here. Just run the loop with the
        # existing AGENT context current so STEP/LLM/TOOL remain descendants.
        lifecycle_agent_span = getattr(controller, _AGENT_SPAN_ATTR, None)
        lifecycle_agent_ctx = getattr(controller, _AGENT_CTX_ATTR, None)
        if (
            lifecycle_agent_span is not None
            and lifecycle_agent_ctx is not None
        ):
            try:
                _capture_agent_io_attributes(
                    lifecycle_agent_span,
                    controller,
                    agent,
                    safe_get_attr(controller, "state"),
                )
            except Exception:
                pass
            lifecycle_token = otel_context.attach(lifecycle_agent_ctx)
            try:
                return await wrapped(*args, **kwargs)
            except BaseException as exc:
                try:
                    lifecycle_agent_span.record_exception(exc)
                    lifecycle_agent_span.set_status(
                        Status(StatusCode.ERROR, type(exc).__qualname__)
                    )
                except Exception:
                    pass
                raise
            finally:
                try:
                    state = safe_get_attr(controller, "state")
                    _capture_agent_io_attributes(
                        lifecycle_agent_span, controller, agent, state
                    )
                    history = safe_get_attr(state, "history") or []
                    if isinstance(history, list):
                        lifecycle_agent_span.set_attribute(
                            OH_HISTORY_LENGTH, len(history)
                        )
                except Exception:
                    pass
                try:
                    otel_context.detach(lifecycle_token)
                except Exception:
                    pass

        # Bridge: re-attach whatever the ENTRY wrapper stashed (works even
        # if asyncio.create_task somehow lost the context, and is the only
        # way for the worker-thread STEP / TOOL spans to find us).
        attach_ctx = get_context(sid)
        fallback_entry_span: trace_api.Span | None = None
        if attach_ctx is None:
            fallback_entry_span = self._tracer.start_span(
                "enter openhands", kind=SpanKind.INTERNAL
            )
            _set_common(fallback_entry_span, "ENTRY")
            fallback_entry_span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME, "enter"
            )
            if sid:
                fallback_entry_span.set_attribute(GEN_AI_SESSION_ID, sid)
                fallback_entry_span.set_attribute(GEN_AI_CONVERSATION_ID, sid)
            if agent_class:
                fallback_entry_span.set_attribute(OH_AGENT_NAME, agent_class)
            if model:
                fallback_entry_span.set_attribute(
                    GEN_AI_REQUEST_MODEL, safe_str(model)
                )
            try:
                state = safe_get_attr(controller, "state")
                entry_input_messages, _ = _entry_io_from_state(state)
                if entry_input_messages:
                    _set_io(
                        fallback_entry_span,
                        input_messages=entry_input_messages,
                    )
            except Exception:
                pass
            attach_ctx = set_span_in_context(fallback_entry_span)
            if sid:
                store_context(sid, attach_ctx)
        if attach_ctx is not None:
            attach_token = otel_context.attach(attach_ctx)
        else:
            attach_token = None

        try:
            span = self._tracer.start_span(
                f"invoke_agent {agent_name}",
                kind=SpanKind.INTERNAL,
                context=attach_ctx,
            )
            _set_common(span, "AGENT")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
            )
            span.set_attribute(GenAI.GEN_AI_AGENT_NAME, safe_str(agent_name))
            if agent_class:
                span.set_attribute(OH_AGENT_NAME, agent_class)
            if sid:
                span.set_attribute(GEN_AI_SESSION_ID, sid)
                span.set_attribute(GEN_AI_CONVERSATION_ID, sid)
                span.set_attribute(GEN_AI_AGENT_ID, sid)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, safe_str(model))

            # Capture the agent's tool registry so the TOOL wrapper (which
            # only sees a Runtime instance) can resolve tool descriptions
            # and produce ``gen_ai.tool.description``. Also emit
            # ``gen_ai.tool.definitions`` on this AGENT span itself per the
            # ARMS GenAI semconv §Agent — minimal {type,name} entries by
            # default; full definitions only when content capture is on.
            try:
                tools = safe_get_attr(agent, "tools") or []
                if sid:
                    store_tool_registry(sid, tools)
                tool_defs_summary: list[dict[str, Any]] = []
                for t in tools:
                    if isinstance(t, dict):
                        kind = t.get("type") or "function"
                        fn = t.get("function") or {}
                        name = fn.get("name") if isinstance(fn, dict) else None
                    else:
                        kind = safe_get_attr(t, "type") or "function"
                        fn = safe_get_attr(t, "function")
                        name = safe_get_attr(fn, "name")
                    if not name:
                        continue
                    item: dict[str, Any] = {
                        "type": safe_str(kind),
                        "name": safe_str(name),
                    }
                    if isinstance(fn, dict):
                        desc = fn.get("description")
                        params = fn.get("parameters")
                    else:
                        desc = safe_get_attr(fn, "description")
                        params = safe_get_attr(fn, "parameters")
                    if desc:
                        item["description"] = safe_str(desc)
                    if params:
                        item["parameters"] = params
                    tool_defs_summary.append(item)
                if tool_defs_summary:
                    span.set_attribute(
                        GEN_AI_TOOL_DEFINITIONS, to_json_str(tool_defs_summary)
                    )
            except Exception:
                pass

            # Capture initial user/system context for AGENT using the same
            # ARMS message schema as the lifecycle-bound AGENT path.
            try:
                state = safe_get_attr(controller, "state")
                _capture_agent_io_attributes(span, controller, agent, state)
            except Exception:
                pass

            # Stash the context that now includes the AGENT span so STEP /
            # TOOL re-attach correctly even when running in worker threads.
            ctx_with_agent = set_span_in_context(span)
            if sid:
                store_context(sid, ctx_with_agent)
            # Mirror onto the controller too — STEP wrapper uses this when
            # closing a STEP to restore the session stash to AGENT instead
            # of leaving a dangling closed-STEP context behind.
            if controller is not None:
                try:
                    setattr(controller, _AGENT_CTX_ATTR, ctx_with_agent)
                    setattr(controller, _AGENT_SPAN_ATTR, span)
                except Exception:
                    pass
                if getattr(controller, _STEP_SPAN_ATTR, None) is None:
                    try:
                        warmup_step = self._tracer.start_span(
                            "react step",
                            kind=SpanKind.INTERNAL,
                            context=ctx_with_agent,
                        )
                        _set_common(warmup_step, "STEP")
                        warmup_step.set_attribute(
                            GenAI.GEN_AI_OPERATION_NAME, "react"
                        )
                        warmup_step.set_attribute(OH_REACT_ROUND, 1)
                        warmup_step.set_attribute(
                            GenAI.GEN_AI_AGENT_NAME, safe_str(agent_name)
                        )
                        if sid:
                            warmup_step.set_attribute(GEN_AI_SESSION_ID, sid)
                            warmup_step.set_attribute(
                                GEN_AI_CONVERSATION_ID, sid
                            )
                            warmup_step.set_attribute(GEN_AI_AGENT_ID, sid)
                        setattr(controller, _STEP_SPAN_ATTR, warmup_step)
                        setattr(controller, "_otel_oh_round", 1)
                        setattr(controller, "_otel_oh_step_consumed", False)
                        if sid:
                            store_context(
                                sid, set_span_in_context(warmup_step)
                            )
                    except Exception:
                        pass
            agent_token = otel_context.attach(ctx_with_agent)
            try:
                try:
                    result = await wrapped(*args, **kwargs)
                except BaseException as exc:
                    span.record_exception(exc)
                    span.set_status(
                        Status(StatusCode.ERROR, type(exc).__qualname__)
                    )
                    raise
                # Capture final AGENT I/O using ARMS gen_ai.* message attrs.
                try:
                    state = safe_get_attr(controller, "state")
                    _capture_agent_io_attributes(
                        span, controller, agent, state
                    )
                    if state is not None:
                        agent_state = safe_get_attr(state, "agent_state")
                        if agent_state is not None:
                            span.set_attribute(
                                OH_AGENT_STATE,
                                safe_get_attr(agent_state, "value")
                                or safe_str(agent_state),
                            )
                        history = safe_get_attr(state, "history") or []
                        if isinstance(history, list):
                            span.set_attribute(OH_HISTORY_LENGTH, len(history))
                except Exception:
                    pass
                return result
            finally:
                try:
                    otel_context.detach(agent_token)
                except Exception:
                    pass
                if controller is not None:
                    try:
                        if getattr(controller, _AGENT_SPAN_ATTR, None) is span:
                            setattr(controller, _AGENT_SPAN_ATTR, None)
                    except Exception:
                        pass
                    try:
                        _close_open_step(controller)
                    except Exception:
                        pass
                span.end()
        finally:
            if attach_token is not None:
                try:
                    otel_context.detach(attach_token)
                except Exception:
                    pass
            if fallback_entry_span is not None:
                try:
                    state = safe_get_attr(controller, "state")
                    entry_input_messages, entry_output_messages = (
                        _entry_io_from_state(state)
                    )
                    if entry_input_messages or entry_output_messages:
                        _set_io(
                            fallback_entry_span,
                            input_messages=entry_input_messages,
                            output_messages=entry_output_messages,
                        )
                    history = safe_get_attr(state, "history") or []
                    if isinstance(history, list):
                        fallback_entry_span.set_attribute(
                            OH_HISTORY_LENGTH, len(history)
                        )
                except Exception:
                    pass
                try:
                    fallback_entry_span.end()
                except Exception:
                    pass
                if sid:
                    try:
                        clear_context(sid)
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# STEP: AgentController._step
# ---------------------------------------------------------------------------


def _close_open_step(controller: Any) -> None:
    """End the controller's currently-open STEP span, if any.

    Restores the session-context stash to the controller's AGENT context
    (kept under ``_AGENT_CTX_ATTR``) so subsequent TOOL spans are still
    parented correctly even after the last STEP closes.

    Crucially, this function only ends the *span* — it never touches an
    attach-token. The STEP wrapper attaches/detaches the STEP context
    in a balanced pair *inside* the ``_step`` coroutine; cross-task
    propagation happens via the ``Context`` object stashed in
    :mod:`session_context`, which can be re-attached safely from any
    task / thread because every attach is paired with a detach inside
    its creating context.
    """
    span = getattr(controller, _STEP_SPAN_ATTR, None)
    if span is None:
        return
    try:
        span.end()
    except Exception:
        pass
    try:
        setattr(controller, _STEP_SPAN_ATTR, None)
    except Exception:
        pass
    sid = safe_str(safe_get_attr(controller, "id") or "")
    agent_ctx = getattr(controller, _AGENT_CTX_ATTR, None)
    if sid and agent_ctx is not None:
        store_context(sid, agent_ctx)


class AgentControllerStepWrapper:
    """STEP span around one ReAct iteration of the V0 controller.

    The STEP span is intentionally **kept open across the return of
    ``_step``**. Why: ``Runtime.run_action`` runs *later*, in a thread-pool
    executor (``call_sync_from_async`` inside ``_handle_action``), so by
    the time TOOL fires the STEP coroutine has already returned. Closing
    STEP at end of ``_step`` would make every TOOL a sibling of STEP
    (parented under AGENT) instead of a child.

    Lifecycle:

    1. New ``_step`` invoked → close *previous* STEP if any → open new
       STEP (child of AGENT) → stash STEP context under ``sid`` so that
       TOOL / LLM spans firing on worker threads re-attach STEP.
    2. ``_step`` body runs to completion. We do **not** close STEP here.
    3. The next ``_step`` (or ``AgentController.close``) closes the
       still-open STEP.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        return self._impl(wrapped, instance, args, kwargs)

    @staticmethod
    def _will_step_be_noop(instance: Any) -> bool:
        """Return True if this ``_step`` call will short-circuit without
        producing real work (state != RUNNING, or a pending action is
        already queued). We skip span emission for these so the round
        counter stays sequential (1, 2, 3, ...) instead of inflating to
        (1, 3, 5, ...) with empty 0.5ms STEP spans cluttering the trace.

        This mirrors the early-return checks at the top of
        ``AgentController._step`` (state-check + ``_pending_action``).
        We read ``_pending_action_info`` directly rather than going
        through the ``_pending_action`` *property* — the property has
        logging side effects (it can emit a "pending action active for
        Xs" log line at warn-level) that we don't want to trigger from
        an instrumentation hot path.
        """
        try:
            state = safe_get_attr(instance, "state")
            agent_state = safe_get_attr(state, "agent_state")
            # AgentState enum value is 'running' (case-insensitive).
            agent_state_str = safe_str(
                safe_get_attr(agent_state, "value") or agent_state
            ).lower()
            if agent_state_str != "running":
                return True
            # Check the underlying tuple slot, not the property — the
            # property's getter is non-trivial in OpenHands.
            if getattr(instance, "_pending_action_info", None) is not None:
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _snapshot_for_work_detection(instance: Any) -> tuple[int, Any]:
        """Snapshot the bits we need to tell whether ``_step`` body did
        anything. Returned tuple is (history_length, pending_action_id).
        Used by ``_impl`` to detect "empty" STEP invocations that get
        through ``_will_step_be_noop`` (e.g. ``state_tracker`` raised,
        ``_is_stuck`` early-returned, ``agent.step`` returned ``None``)
        and shouldn't show up in the trace as 0.3ms placeholder spans.
        """
        try:
            state = safe_get_attr(instance, "state")
            history = safe_get_attr(state, "history")
            history_len = len(history) if isinstance(history, list) else 0
        except Exception:
            history_len = 0
        try:
            info = getattr(instance, "_pending_action_info", None)
            pending_id = id(info) if info is not None else None
        except Exception:
            pending_id = None
        return history_len, pending_id

    async def _impl(self, wrapped, instance, args, kwargs):
        if not OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS:
            return await wrapped(*args, **kwargs)

        # Skip no-op _step invocations entirely so the trace shows only
        # the rounds that actually do work (LLM call + tool dispatch).
        if self._will_step_be_noop(instance):
            return await wrapped(*args, **kwargs)

        sid = safe_str(safe_get_attr(instance, "id") or "")
        agent = safe_get_attr(instance, "agent")
        agent_name = safe_get_attr(agent, "name") or "codeact"

        # Snapshot the AGENT context if we don't already have one so
        # ``_close_open_step`` can restore the session stash to AGENT
        # after STEP ends.
        if (
            not hasattr(instance, _AGENT_CTX_ATTR)
            or getattr(instance, _AGENT_CTX_ATTR, None) is None
        ):
            try:
                setattr(instance, _AGENT_CTX_ATTR, get_context(sid))
            except Exception:
                pass

        # ----- Reuse warmup STEP if not yet consumed -----
        # The init wrapper opens a warmup STEP (round 1) so pre-step
        # actions like RECALL parent under STEP 1. The first real
        # ``_step`` reuses that STEP (without bumping the round) so the
        # LLM call + first LLM-driven tool also nest under STEP 1. From
        # the second real ``_step`` onward, we close the previous STEP
        # and open a new one with round = previous + 1.
        existing_step = getattr(instance, _STEP_SPAN_ATTR, None)
        consumed = bool(getattr(instance, "_otel_oh_step_consumed", True))
        reused_warmup = False
        is_new_span = False
        if existing_step is not None and not consumed:
            span = existing_step
            round_num = int(getattr(instance, "_otel_oh_round", 1) or 1)
            reused_warmup = True
            try:
                setattr(instance, "_otel_oh_step_consumed", True)
            except Exception:
                pass
        else:
            # Close any still-open consumed STEP from the previous round
            # before opening a new one.
            _close_open_step(instance)
            # Tentative round number — only committed if body does work.
            round_num = int(getattr(instance, "_otel_oh_round", 0) or 0) + 1

            # Open the new STEP as a child of AGENT. Prefer the explicit
            # AGENT context (more reliable than relying on contextvars
            # propagation across asyncio task / thread boundaries).
            agent_ctx = getattr(instance, _AGENT_CTX_ATTR, None)
            if agent_ctx is None and sid:
                agent_ctx = get_context(sid)
            try:
                span = self._tracer.start_span(
                    "react step",
                    kind=SpanKind.INTERNAL,
                    context=agent_ctx,
                )
            except Exception:
                # Fall back to current-context-based parenting if explicit
                # context= isn't accepted (older OTel SDKs).
                with AttachedSession(sid):
                    span = self._tracer.start_span(
                        "react step", kind=SpanKind.INTERNAL
                    )
            _set_common(span, "STEP")
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "react")
            span.set_attribute(OH_REACT_ROUND, round_num)
            span.set_attribute(GenAI.GEN_AI_AGENT_NAME, safe_str(agent_name))
            if sid:
                span.set_attribute(GEN_AI_SESSION_ID, sid)
                span.set_attribute(GEN_AI_CONVERSATION_ID, sid)
                span.set_attribute(GEN_AI_AGENT_ID, sid)
            is_new_span = True
            try:
                setattr(instance, _STEP_SPAN_ATTR, span)
                setattr(instance, "_otel_oh_step_consumed", True)
            except Exception:
                try:
                    span.end()
                except Exception:
                    pass
                return await wrapped(*args, **kwargs)

        # Capture INPUT: messages going into this step.
        try:
            state = safe_get_attr(instance, "state")
            history = safe_get_attr(state, "history") or []
            if isinstance(history, list):
                span.set_attribute(OH_HISTORY_LENGTH, len(history))
            input_messages = _state_to_input_messages(state)
            if input_messages:
                _set_io(
                    span,
                    input_value=input_messages,
                    input_messages=input_messages,
                )
        except Exception:
            pass

        # Build the STEP context object. Cross-thread propagation goes
        # through this Context object stashed in session_context (TOOL /
        # LLM wrappers re-attach it inside their own scopes with paired
        # attach/detach so no token ever crosses a context boundary).
        step_ctx = set_span_in_context(span)
        if sid:
            store_context(sid, step_ctx)

        # Snapshot pre-body state so we can detect "empty" body that
        # got through ``_will_step_be_noop`` (e.g. ``state_tracker``
        # raised inside ``_step``, ``_is_stuck`` early-returned, or
        # ``agent.step`` returned ``None`` / raised handled error).
        pre_history_len, pre_pending_id = self._snapshot_for_work_detection(
            instance
        )

        # Attach STEP for the *body's* contextvars propagation only.
        # Both attach and the matching detach happen in this coroutine's
        # own context, so the Aliyun SDK's strict token check is happy.
        step_token = otel_context.attach(step_ctx)
        body_error: BaseException | None = None
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            body_error = exc
        finally:
            try:
                otel_context.detach(step_token)
            except Exception:
                pass

        if body_error is not None:
            try:
                span.set_attribute(
                    "gen_ai.react.finish_reason", type(body_error).__qualname__
                )
                span.record_exception(body_error)
                span.set_status(
                    Status(StatusCode.ERROR, type(body_error).__qualname__)
                )
            except Exception:
                pass
            # On error, close STEP now so the failure surfaces cleanly
            # rather than waiting for the next _step / controller close.
            _close_open_step(instance)
            # Make sure the round counter we *tentatively* assigned for
            # this STEP gets committed so subsequent rounds renumber
            # past it instead of overlapping.
            if is_new_span:
                try:
                    instance._otel_oh_round = round_num
                except Exception:
                    pass
            raise body_error

        # Detect post-body "empty" STEP — the wrapper passed the
        # ``_will_step_be_noop`` pre-check but the body still produced
        # zero observable work (no new history events, no new pending
        # action). The user has explicitly asked us not to clutter the
        # trace with sub-millisecond placeholder STEP spans, so:
        #
        # * If we *opened* a fresh span this round, end it immediately,
        #   mark it ``openhands.step.empty=true``, and DO NOT bump the
        #   committed round counter. Next real _step opens a fresh STEP
        #   with the same round number — the empty span still appears
        #   in the trace (we have no way to suppress export from inside
        #   a wrapper), but with a clear ``empty=true`` marker so it's
        #   trivially filterable in the dashboard.
        # * If we *reused* a warmup / persisted STEP that was already
        #   meaningful (had earlier RECALL/TOOL children), keep it open
        #   and don't mark it empty — the children give it value.
        post_history_len, post_pending_id = self._snapshot_for_work_detection(
            instance
        )
        did_work = post_history_len > pre_history_len or (
            post_pending_id is not None and post_pending_id != pre_pending_id
        )

        if not did_work and is_new_span:
            try:
                span.set_attribute("openhands.step.empty", True)
                span.set_attribute(
                    "gen_ai.react.finish_reason", "noop_step_body"
                )
                span.end()
            except Exception:
                pass
            # Forget this empty STEP so the next _step opens a fresh one
            # without trying to close-or-reuse this one.
            try:
                if getattr(instance, _STEP_SPAN_ATTR, None) is span:
                    setattr(instance, _STEP_SPAN_ATTR, None)
            except Exception:
                pass
            try:
                # Roll back to the previous committed round (don't
                # advance the counter for an empty STEP).
                instance._otel_oh_round = round_num - 1
                instance._otel_oh_step_consumed = True
            except Exception:
                pass
            # Restore session stash to AGENT so subsequent TOOLs land
            # under AGENT (not under a now-ended STEP).
            if sid:
                agent_ctx = getattr(instance, _AGENT_CTX_ATTR, None)
                if agent_ctx is not None:
                    try:
                        store_context(sid, agent_ctx)
                    except Exception:
                        pass
            return result

        # Body did work — commit the round counter (we only update it
        # *after* we're sure the STEP is meaningful).
        if is_new_span:
            try:
                instance._otel_oh_round = round_num
            except Exception:
                pass

        # Capture OUTPUT: the freshly-decided pending action.
        try:
            pending = getattr(instance, "_pending_action", None)
            state = safe_get_attr(instance, "state")
            agent_state = safe_get_attr(state, "agent_state")
            if agent_state is not None:
                span.set_attribute(
                    OH_AGENT_STATE,
                    safe_get_attr(agent_state, "value")
                    or safe_str(agent_state),
                )
            if pending is not None:
                action_type = _action_type_value(pending)
                if action_type:
                    span.set_attribute(OH_ACTION_TYPE, action_type)
                out = action_to_genai_output(pending)
                if out:
                    _set_io(span, output_value=out, output_messages=out)
        except Exception:
            pass

        # Mirror the latest history snapshot back up to the AGENT span
        # so AGENT's GenAI message attributes stay current during the run
        # (not just at close-time). Downstream dashboards may read AGENT
        # before the controller actually closes.
        try:
            agent_span = getattr(instance, _AGENT_SPAN_ATTR, None)
            if agent_span is not None:
                _capture_agent_io_attributes(
                    agent_span,
                    instance,
                    agent,
                    safe_get_attr(instance, "state"),
                )
        except Exception:
            pass

        # Mark the warmup STEP (round 1) the moment we know it carries
        # real work — it now contains LLM/TOOL children and matters.
        if reused_warmup:
            try:
                span.set_attribute("openhands.step.warmup_consumed", True)
            except Exception:
                pass

        # STEP span stays open here — it lives until the next _step (or
        # AgentController.close) ends it. Until then any TOOL fired by
        # Runtime.run_action on a thread-pool worker will re-attach the
        # STEP context object stashed above and become its child.
        return result


# ---------------------------------------------------------------------------
# TOOL: Runtime.run_action
# ---------------------------------------------------------------------------


_TOOL_KIND_TO_NAME: dict[str, str] = {
    "run": "bash",
    "run_ipython": "ipython",
    "browse_interactive": "browser",
    "browse": "browser",
    "edit": "str_replace_editor",
    "read": "file_read",
    "write": "file_write",
    "delegate": "delegate",
    "finish": "finish",
    "think": "think",
    "task_tracking": "task_tracker",
    "mcp": "mcp",
    "send_message": "send_message",
    # ``recall`` is a real (non-LLM-initiated) tool: the controller posts
    # a RecallAction and the memory subsystem runs it just like any other
    # action via ``Runtime.run_action``. Worth a TOOL span.
    "recall": "recall",
}

# Action types that are *not* real tool calls — they're internal control
# events posted by the controller / event-stream itself (system prompt,
# user message, agent-state transition, no-ops). Emitting TOOL spans for
# these clutters the trace tree and confuses the GenAI semconv (these
# aren't things the LLM "called").
_INTERNAL_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "message",
        "system",
        "change_agent_state",
        "agent_state_changed",
        "null",
        "noop",
    }
)


def _action_type_value(action: Any) -> str:
    """Best-effort extract the canonical action-type string for ``action``.

    OpenHands declares ``ActionType`` as ``class ActionType(str, Enum)``
    with members like ``MESSAGE = 'message'``. Each Action subclass sets
    ``action: str = ActionType.MESSAGE``. ``str(ActionType.MESSAGE)``
    returns ``'ActionType.MESSAGE'`` (Python's default Enum.__str__),
    *not* the value ``'message'`` we want for filtering / lookup. This
    helper prefers ``.value`` when the attribute is enum-like, else the
    raw string.
    """
    raw = safe_get_attr(action, "action")
    if raw is None:
        return ""
    val = safe_get_attr(raw, "value")
    if val is not None:
        return safe_str(val).lower()
    text = safe_str(raw).lower()
    # ``str(ActionType.MESSAGE)`` → "actiontype.message"; strip the prefix.
    prefix = "actiontype."
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _is_real_tool_call(action: Any) -> bool:
    """Return True iff ``action`` represents a meaningful tool execution.

    Filtering rules (in order):

    1. **Internal action types are *always* dropped** even when the
       action carries ``tool_call_metadata``. OpenHands lets the LLM
       produce ``MessageAction`` (via the ``send_message`` "tool"),
       ``SystemMessageAction``, ``ChangeAgentStateAction`` etc. — those
       are coordination signals, not real tool executions, and they
       clutter the trace with sub-millisecond noise spans that the user
       has explicitly asked us to suppress.
    2. Otherwise, an action qualifies if it has ``tool_call_metadata``
       (i.e. it was produced from an LLM ``tool_calls`` response — e.g.
       ``execute_bash``, ``str_replace_editor``), or
    3. Its action-type is in the executable-tool whitelist
       (``_TOOL_KIND_TO_NAME``) — this catches synthesized actions like
       ``RECALL`` that don't come from the LLM but are still worth
       tracing as TOOL spans (memory retrieval, microagent loading,
       etc.).
    """
    action_type = _action_type_value(action)
    # Always drop internal/system actions regardless of how they were
    # produced — see rule 1 above.
    if action_type and action_type in _INTERNAL_ACTION_TYPES:
        return False
    if safe_get_attr(action, "tool_call_metadata") is not None:
        return True
    if not action_type:
        return False
    return action_type in _TOOL_KIND_TO_NAME


def _extract_tool_name(action: Any) -> tuple[str, str]:
    """Return (tool_name, action_type).

    Prefers the function name carried on ``action.tool_call_metadata``
    (set when the action came from an LLM tool call) — that's what the
    LLM and our LLM-side instrumentation know it as. Falls back to the
    canonical action-type string (``ActionType.RECALL`` → ``"recall"``)
    mapped through ``_TOOL_KIND_TO_NAME``.
    """
    action_type = _action_type_value(action)
    tcm = safe_get_attr(action, "tool_call_metadata")
    if tcm is not None:
        fn = safe_get_attr(tcm, "function_name")
        if fn:
            return safe_str(fn), action_type
    tool_name = _TOOL_KIND_TO_NAME.get(
        action_type, action_type or "agent.action"
    )
    return tool_name, action_type


def _extract_tool_call_id(action: Any) -> str:
    tcm = safe_get_attr(action, "tool_call_metadata")
    if tcm is None:
        return ""
    return safe_str(safe_get_attr(tcm, "tool_call_id") or "")


def _runtime_sid(instance: Any) -> str:
    """Best-effort discover the session id from a Runtime instance."""
    sid = safe_get_attr(instance, "sid")
    if sid:
        return safe_str(sid)
    es = safe_get_attr(instance, "event_stream")
    es_sid = safe_get_attr(es, "sid")
    if es_sid:
        return safe_str(es_sid)
    return ""


class RuntimeRunActionWrapper:
    """TOOL span around ``Runtime.run_action``.

    Bridges the session context across worker threads, then opens a TOOL
    span with GenAI tool-call attributes. Arguments are always recorded
    in ``gen_ai.tool.call.arguments`` (``"{}"`` when none); results go to
    ``gen_ai.tool.call.result``. No ``input.value`` / ``output.value``.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        if not OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS:
            return wrapped(*args, **kwargs)

        action = args[0] if args else kwargs.get("action")
        # Skip internal control events — system prompts, user messages,
        # memory recalls, agent-state transitions etc. aren't tool calls
        # and shouldn't appear as TOOL spans alongside the real ones.
        if not _is_real_tool_call(action):
            return wrapped(*args, **kwargs)

        tool_name, action_type = _extract_tool_name(action)
        tool_call_id = _extract_tool_call_id(action)
        runtime_class = (
            f"{type(instance).__module__}.{type(instance).__name__}"
            if instance
            else ""
        )
        sid = _runtime_sid(instance)

        # Look up the session-stashed context (STEP if a step is open,
        # AGENT otherwise) and use it as the *explicit* parent context
        # for the TOOL span. Explicit context= is more robust than
        # relying on contextvars propagation across worker threads — it
        # always parents under the latest STEP/AGENT no matter what
        # thread/loop the runtime is running on.
        parent_ctx = get_context(sid)
        try:
            span = self._tracer.start_span(
                f"execute_tool {tool_name}",
                kind=SpanKind.INTERNAL,
                context=parent_ctx,
            )
        except Exception:
            with AttachedSession(sid):
                span = self._tracer.start_span(
                    f"execute_tool {tool_name}", kind=SpanKind.INTERNAL
                )
        # The TOOL span itself is parented *explicitly* via context=
        # above. We additionally attach the session context throughout
        # the wrapped call so any nested spans created by the runtime
        # (e.g. a retried LLM call) that go through the contextvars
        # propagation path also inherit the right session — and the
        # ``otel_context.attach(set_span_in_context(span))`` below makes
        # the TOOL itself current so retry-spawned child spans nest
        # under TOOL, not under its parent STEP.
        with AttachedSession(sid):
            # ARMS GenAI semconv (Tool):
            #   gen_ai.span.kind=TOOL, gen_ai.operation.name=execute_tool,
            #   gen_ai.tool.name, gen_ai.tool.type
            #   gen_ai.tool.call.id, gen_ai.tool.description    [recommended]
            #   gen_ai.tool.call.arguments, gen_ai.tool.call.result
            #     [optional, gated on capture-message-content]
            _set_common(span, "TOOL")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(GenAI.GEN_AI_TOOL_TYPE, "function")
            if tool_call_id:
                span.set_attribute(GEN_AI_TOOL_CALL_ID, tool_call_id)
            if action_type:
                # ``action_type`` from ``_extract_tool_name`` is the
                # canonical lowercased value (e.g. ``"recall"``), suitable
                # for ``openhands.action.type``.
                span.set_attribute(OH_ACTION_TYPE, action_type)
            if runtime_class:
                span.set_attribute(OH_RUNTIME_NAME, runtime_class)
            if sid:
                span.set_attribute(GEN_AI_SESSION_ID, sid)
                span.set_attribute(GEN_AI_CONVERSATION_ID, sid)

            # gen_ai.tool.description — looked up via the per-sid registry
            # populated by the AGENT wrapper from ``controller.agent.tools``.
            try:
                tool_def = get_tool_definition(sid, tool_name)
                if tool_def is not None:
                    if isinstance(tool_def, dict):
                        fn = tool_def.get("function") or {}
                        desc = (
                            fn.get("description")
                            if isinstance(fn, dict)
                            else None
                        )
                    else:
                        fn = safe_get_attr(tool_def, "function")
                        desc = safe_get_attr(fn, "description")
                    if desc:
                        span.set_attribute(
                            GEN_AI_TOOL_DESCRIPTION, safe_str(desc)
                        )
            except Exception:
                pass

            # gen_ai.tool.call.arguments — always emit (empty object as "{}" ).
            # No OpenInference input.value / output.value on TOOL spans.
            arguments_dict = _tool_call_arguments(action)
            try:
                args_json = to_json_str(arguments_dict)
                if not args_json:
                    args_json = "{}"
                span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, args_json)
                preview_field, preview_text = _first_preview_field(action)
                if preview_text:
                    span.set_attribute(
                        f"openhands.action.{preview_field}", preview_text
                    )
            except Exception:
                span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, "{}")

            ctx = set_span_in_context(span)
            token = otel_context.attach(ctx)
            try:
                try:
                    observation = wrapped(*args, **kwargs)
                except BaseException as exc:
                    span.record_exception(exc)
                    span.set_status(
                        Status(StatusCode.ERROR, type(exc).__qualname__)
                    )
                    raise
                try:
                    _annotate_observation(span, observation)
                except Exception:
                    pass
                return observation
            finally:
                try:
                    otel_context.detach(token)
                except Exception:
                    pass
                span.end()


def _first_preview_field(action: Any) -> tuple[str, str]:
    for attr in ("command", "code", "path", "url", "content"):
        v = safe_get_attr(action, attr)
        if v:
            return attr, safe_str(v)
    return "", ""


_TOOL_ARG_FIELDS: tuple[str, ...] = (
    "command",
    "code",
    "path",
    "url",
    "content",
    "task_list",
    "name",
    "arguments",
    "thought",
    "is_input",
    "blocking",
    "keep_prompt",
    "translated_ipython_code",
    "browser_actions",
    "agent_state",
    "outputs",
    "final_thought",
    "old_str",
    "new_str",
    "view_range",
    "file_text",
    "insert_line",
    "start_line",
    "end_line",
)


def _coerce_tool_arguments(value: Any) -> dict[str, Any]:
    """Normalize tool call arguments to a JSON-object-compatible dict."""
    if value in (None, "", [], {}):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {"raw": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {"value": value}


def _tool_call_arguments(action: Any) -> dict[str, Any]:
    """Return the bare arguments dict for ``gen_ai.tool.call.arguments``.

    Per ARMS GenAI semconv the value is a JSON string of *just* the call
    arguments — e.g. ``{"location": "San Francisco", "date": "2025-10-01"}``
    — not the wrapping ``{"tool": ..., "arguments": ...}`` envelope.
    """
    if action is None:
        return {}
    # When the action came from an LLM tool call, prefer the original
    # JSON arguments the model emitted (most faithful to what the LLM
    # actually requested).
    tcm = safe_get_attr(action, "tool_call_metadata")
    if tcm is not None:
        direct_args = _coerce_tool_arguments(safe_get_attr(tcm, "arguments"))
        if direct_args:
            return direct_args
    model_response = safe_get_attr(tcm, "model_response") if tcm else None
    if model_response is not None:
        try:
            choices = (
                model_response.choices
                if hasattr(model_response, "choices")
                else None
            ) or []
            for choice in choices:
                msg = getattr(choice, "message", None) or (
                    choice.get("message") if isinstance(choice, dict) else None
                )
                tool_calls = (
                    getattr(msg, "tool_calls", None)
                    if msg is not None
                    else None
                ) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
                if not tool_calls:
                    continue
                want_id = safe_str(safe_get_attr(tcm, "tool_call_id") or "")
                for tc in tool_calls:
                    tc_id = (
                        getattr(tc, "id", None)
                        if not isinstance(tc, dict)
                        else tc.get("id")
                    )
                    if want_id and safe_str(tc_id) != want_id:
                        continue
                    fn = (
                        getattr(tc, "function", None)
                        if not isinstance(tc, dict)
                        else tc.get("function")
                    )
                    raw_args = (
                        getattr(fn, "arguments", None)
                        if not isinstance(fn, dict)
                        else fn.get("arguments")
                    )
                    parsed_args = _coerce_tool_arguments(raw_args)
                    if parsed_args:
                        return parsed_args
        except Exception:
            pass
    # Fallback: harvest known argument-bearing fields off the Action object.
    args: dict[str, Any] = {}
    for key in _TOOL_ARG_FIELDS:
        v = safe_get_attr(action, key)
        if v not in (None, "", [], {}):
            args[key] = v
    return args


def _observation_to_result(observation: Any) -> dict[str, Any]:
    """Return a dict suitable for ``gen_ai.tool.call.result``."""
    if observation is None:
        return {}
    payload: dict[str, Any] = {}
    for key in (
        "content",
        "exit_code",
        "error",
        "interpreter_details",
        "command",
        "stdout",
        "stderr",
        "url",
        "screenshot",
        "outputs",
    ):
        v = safe_get_attr(observation, key)
        if v not in (None, "", [], {}):
            payload[key] = v
    return payload


def _annotate_observation(span: trace_api.Span, observation: Any) -> None:
    if observation is None:
        return
    obs_type = safe_str(
        safe_get_attr(observation, "observation") or type(observation).__name__
    )
    if obs_type:
        span.set_attribute(OH_OBSERVATION_TYPE, obs_type)
    exit_code = safe_get_attr(observation, "exit_code")
    if exit_code is not None:
        try:
            ec = int(exit_code)
            span.set_attribute("openhands.action.exit_code", ec)
            if ec != 0:
                span.set_status(Status(StatusCode.ERROR, f"exit_code={ec}"))
        except (TypeError, ValueError):
            pass
    error = safe_get_attr(observation, "error")
    if error:
        span.set_attribute("openhands.observation.error", safe_str(error))
        span.set_status(Status(StatusCode.ERROR, safe_str(error)))
    # TOOL spans do not emit OpenInference output.value; the result lives in
    # the GenAI tool-call result attribute.
    try:
        result_payload = _observation_to_result(observation)
        result_payload.setdefault("observation", obs_type)
        out = to_json_str(result_payload)
        if out:
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, out)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ENTRY + AGENT (controller-lifecycle bound)
#
# Why this exists in addition to RunControllerWrapper / RunAgentUntilDoneWrapper:
#
# When OpenHands V0 is launched via ``python -m openhands.core.main``, Python
# executes ``main.py`` *as ``__main__``*. The ``from openhands.core.loop
# import run_agent_until_done`` (and other from-imports) at the top of
# ``main.py`` bind those symbols into ``__main__``'s namespace **before**
# our instrumentor patches ``openhands.core.main.run_controller`` /
# ``openhands.core.loop.run_agent_until_done``. The ``__main__`` block's
# ``asyncio.run(run_controller(...))`` call uses the *unpatched* local
# reference, so the wrappers above never fire — and the trace appears
# without an ENTRY span.
#
# STEP / TOOL spans work because ``_step`` and ``run_action`` are *class
# methods*: patching ``AgentController._step`` updates the class object
# that both ``__main__.AgentController`` and
# ``openhands.controller.agent_controller.AgentController`` reference, so
# every method lookup at call time finds the wrapped version.
#
# ENTRY+AGENT here exploit the same principle — they hook
# ``AgentController.__init__`` and ``AgentController.close``, both class
# methods, so the spans bracket the controller's lifecycle reliably no
# matter how ``run_controller`` was invoked. They no-op when a session
# context is already stashed for this sid (i.e. ``RunControllerWrapper``
# fired successfully — the API/test-suite code path).
# ---------------------------------------------------------------------------


def _capture_agent_io_attributes(
    span: trace_api.Span, controller: Any, agent: Any, state: Any
) -> None:
    """Set gen_ai.system_instructions / input.messages / output.messages on
    the AGENT span, following the ARMS GenAI semconv schema."""
    try:
        sys_instr = _agent_to_system_instructions(agent, state)
        if sys_instr:
            payload = to_json_str(sys_instr)
            if payload:
                span.set_attribute(GEN_AI_SYSTEM_INSTRUCTIONS, payload)
    except Exception:
        pass
    try:
        history = safe_get_attr(state, "history") or []
        if isinstance(history, list) and history:
            input_msgs = _history_to_input_messages_schema(history)
            if input_msgs:
                payload = to_json_str(input_msgs)
                if payload:
                    span.set_attribute(GEN_AI_INPUT_MESSAGES, payload)
            output_msgs = _history_to_output_messages_schema(history)
            if output_msgs:
                payload = to_json_str(output_msgs)
                if payload:
                    span.set_attribute(GEN_AI_OUTPUT_MESSAGES, payload)
    except Exception:
        pass


def _open_entry_and_agent_for_controller(
    tracer: Tracer, controller: Any
) -> None:
    """Open ENTRY (parent) + AGENT (child) + warmup STEP for ``controller``.

    Opening a *warmup STEP* (round 1) right after AGENT means that any
    pre-step actions like RECALL — which are dispatched to the runtime
    *before* the first ``_step`` invocation — become children of STEP 1
    instead of dangling siblings under AGENT. The first real ``_step``
    call detects that the warmup STEP isn't yet "consumed" and reuses
    it (without bumping the round counter) so the LLM call + first
    LLM-driven tool also nest under STEP 1.

    All inner span creations use the explicit ``context=`` argument
    (instead of relying on ``contextvars`` propagation through
    ``otel_context.attach``) — this is the most deterministic way to
    parent a child span and avoids the entire class of "Token was
    created in a different Context" failures we used to chase across
    asyncio-task / thread boundaries.

    Idempotent on ``_OWNS_FLAG`` — safe to call multiple times for the
    same controller. Deliberately does **not** check whether a session
    context is already stashed: under ``python -m openhands.core.main``
    the from-import binding bypasses ``RunControllerWrapper`` and
    ``RunAgentUntilDoneWrapper``, so the init wrapper is the only
    reliable source of ENTRY+AGENT and must always run.
    """
    if not OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS:
        return
    if getattr(controller, _OWNS_FLAG, False):
        # Already opened (e.g. RunControllerWrapper fired first) — log
        # and bail. We don't want to double-emit ENTRY/AGENT.
        logger.debug(
            "OpenHands instrumentation: ENTRY+AGENT already open on "
            "controller %s — skipping init-wrapper open",
            id(controller),
        )
        return

    sid = safe_str(safe_get_attr(controller, "id") or "")
    agent = safe_get_attr(controller, "agent")
    agent_name = safe_get_attr(agent, "name") or "codeact"
    agent_class = (
        f"{type(agent).__module__}.{type(agent).__name__}" if agent else ""
    )
    llm = safe_get_attr(agent, "llm")
    llm_config = safe_get_attr(llm, "config")
    model = safe_get_attr(llm_config, "model") or safe_get_attr(llm, "model")

    # ----- ENTRY -----
    # If RunControllerWrapper already stashed an ENTRY context, parent AGENT
    # directly under it. Otherwise create the lifecycle-owned ENTRY here.
    entry: trace_api.Span | None = None
    entry_ctx = get_context(sid)
    if entry_ctx is None:
        try:
            entry = tracer.start_span(
                "enter openhands", kind=SpanKind.INTERNAL
            )
        except Exception as exc:
            logger.error(
                "OpenHands instrumentation: failed to start ENTRY span for "
                "sid=%r: %s",
                sid,
                exc,
                exc_info=True,
            )
            return

        try:
            _set_common(entry, "ENTRY")
            entry.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "enter")
            if sid:
                entry.set_attribute(GEN_AI_SESSION_ID, sid)
                entry.set_attribute(GEN_AI_CONVERSATION_ID, sid)
            if agent_class:
                entry.set_attribute(OH_AGENT_NAME, agent_class)
            if model:
                entry.set_attribute(GEN_AI_REQUEST_MODEL, safe_str(model))
            state = safe_get_attr(controller, "state")
            entry_input_messages, _ = _entry_io_from_state(state)
            if entry_input_messages:
                _set_io(
                    entry,
                    input_messages=entry_input_messages,
                )
        except Exception as exc:
            logger.debug(
                "OpenHands instrumentation: ENTRY attr setup: %s", exc
            )

        entry_ctx = set_span_in_context(entry)

    # ----- AGENT (child of ENTRY) -----
    # Pass ``context=entry_ctx`` *explicitly* so AGENT inherits ENTRY
    # as parent regardless of what the surrounding contextvars look
    # like (some 3rd-party SDKs reset contextvars between calls).
    try:
        agent_span = tracer.start_span(
            f"invoke_agent {agent_name}",
            kind=SpanKind.INTERNAL,
            context=entry_ctx,
        )
    except Exception as exc:
        logger.error(
            "OpenHands instrumentation: failed to start AGENT span for "
            "sid=%r: %s",
            sid,
            exc,
            exc_info=True,
        )
        if entry is not None:
            try:
                entry.end()
            except Exception:
                pass
        return

    try:
        _set_common(agent_span, "AGENT")
        agent_span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME,
            GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
        )
        agent_span.set_attribute(GenAI.GEN_AI_AGENT_NAME, safe_str(agent_name))
        if agent_class:
            agent_span.set_attribute(OH_AGENT_NAME, agent_class)
        if sid:
            agent_span.set_attribute(GEN_AI_SESSION_ID, sid)
            agent_span.set_attribute(GEN_AI_CONVERSATION_ID, sid)
            agent_span.set_attribute(GEN_AI_AGENT_ID, sid)
        if model:
            agent_span.set_attribute(GEN_AI_REQUEST_MODEL, safe_str(model))
    except Exception as exc:
        logger.debug("OpenHands instrumentation: AGENT attr setup: %s", exc)

    # Tool registry + gen_ai.tool.definitions — same logic as
    # RunAgentUntilDoneWrapper, since this path also needs the
    # registry for downstream TOOL spans.
    try:
        tools = safe_get_attr(agent, "tools") or []
        if sid:
            store_tool_registry(sid, tools)
        defs_summary: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                kind = t.get("type") or "function"
                fn = t.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
            else:
                kind = safe_get_attr(t, "type") or "function"
                fn = safe_get_attr(t, "function")
                name = safe_get_attr(fn, "name")
            if not name:
                continue
            item: dict[str, Any] = {
                "type": safe_str(kind),
                "name": safe_str(name),
            }
            if isinstance(fn, dict):
                desc = fn.get("description")
                params = fn.get("parameters")
            else:
                desc = safe_get_attr(fn, "description")
                params = safe_get_attr(fn, "parameters")
            if desc:
                item["description"] = safe_str(desc)
            if params:
                item["parameters"] = params
            defs_summary.append(item)
        if defs_summary:
            agent_span.set_attribute(
                GEN_AI_TOOL_DEFINITIONS, to_json_str(defs_summary)
            )
    except Exception:
        pass

    # Best-effort INPUT + system_instructions capture on AGENT at open
    # time. ``_capture_agent_io_attributes`` will run again at close to
    # overwrite these with the *final* state, but having them now means
    # an in-flight read of the AGENT span (e.g. live dashboards) sees
    # at least the system prompt + initial user message.
    try:
        state = safe_get_attr(controller, "state")
        _capture_agent_io_attributes(agent_span, controller, agent, state)
    except Exception as exc:
        logger.debug(
            "OpenHands instrumentation: AGENT initial I/O capture: %s", exc
        )

    agent_ctx = set_span_in_context(agent_span)
    if sid:
        # Stash ctx-with-AGENT so STEP / TOOL re-attach correctly even
        # when fired from worker threads with brand-new asyncio loops.
        # The downstream consumers (STEP / TOOL / LLM bridge) all do
        # their own paired attach/detach, so it's safe to share this
        # ``Context`` object across asyncio tasks and threads.
        store_context(sid, agent_ctx)

    # ----- WARMUP STEP (round 1) -----
    # Open right after AGENT so any pre-_step actions (RECALL, etc.) that
    # the controller dispatches to the runtime become children of STEP 1
    # rather than dangling siblings under AGENT. The first real ``_step``
    # call detects this open STEP isn't yet "consumed" and reuses it
    # (preserving the round number) so the LLM call + first LLM-driven
    # tool also nest under STEP 1 — giving the trace tree:
    #
    #   ENTRY > AGENT > STEP 1 > [RECALL, LLM, execute_bash]
    #                  STEP 2 > [LLM, finish]
    #                  ...
    warmup_step_ctx: object | None = None
    warmup_step_span: trace_api.Span | None = None
    try:
        warmup_step_span = tracer.start_span(
            "react step",
            kind=SpanKind.INTERNAL,
            context=agent_ctx,
        )
        _set_common(warmup_step_span, "STEP")
        warmup_step_span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "react")
        warmup_step_span.set_attribute(OH_REACT_ROUND, 1)
        warmup_step_span.set_attribute(
            GenAI.GEN_AI_AGENT_NAME, safe_str(agent_name)
        )
        if sid:
            warmup_step_span.set_attribute(GEN_AI_SESSION_ID, sid)
            warmup_step_span.set_attribute(GEN_AI_CONVERSATION_ID, sid)
            warmup_step_span.set_attribute(GEN_AI_AGENT_ID, sid)
        warmup_step_ctx = set_span_in_context(warmup_step_span)
        if sid and warmup_step_ctx is not None:
            store_context(sid, warmup_step_ctx)
    except Exception as exc:
        logger.debug("Failed to open warmup STEP span: %s", exc)
        warmup_step_span = None

    # Stash everything we need to tear down in close().
    try:
        setattr(controller, _OWNS_FLAG, True)
        setattr(controller, _ENTRY_SPAN_ATTR, entry)
        setattr(controller, _AGENT_SPAN_ATTR, agent_span)
        # Save the AGENT context so the STEP wrapper can restore the
        # session stash to AGENT every time it closes a STEP — that way
        # any TOOL fired between rounds re-attaches AGENT (not a closed
        # STEP).
        setattr(controller, _AGENT_CTX_ATTR, agent_ctx)
        # Stash warmup STEP so the first real ``_step`` reuses it.
        setattr(controller, _STEP_SPAN_ATTR, warmup_step_span)
        setattr(
            controller,
            "_otel_oh_round",
            1 if warmup_step_span is not None else 0,
        )
        setattr(controller, "_otel_oh_step_consumed", False)
    except Exception:
        # If we can't attach to the instance (slots, etc.), close the
        # spans down so we don't leak them.
        if warmup_step_span is not None:
            try:
                warmup_step_span.end()
            except Exception:
                pass
        try:
            agent_span.end()
        except Exception:
            pass
        if entry is not None:
            try:
                entry.end()
            except Exception:
                pass
        return

    # Log at INFO so the user can verify in their app logs that the
    # ENTRY+AGENT spans were actually opened (and which trace/span IDs
    # they got). When a user reports "no ENTRY span" in their backend,
    # the first thing to check is whether this log line appeared.
    try:
        entry_sc = entry.get_span_context() if entry is not None else None
        agent_sc = agent_span.get_span_context()
        warmup_sc = (
            warmup_step_span.get_span_context()
            if warmup_step_span is not None
            else None
        )
        logger.info(
            "OpenHands instrumentation: opened ENTRY+AGENT for sid=%r "
            "(trace_id=%032x entry_span=%016x agent_span=%016x "
            "warmup_step=%s agent_name=%s model=%s)",
            sid,
            entry_sc.trace_id if entry_sc is not None else agent_sc.trace_id,
            entry_sc.span_id if entry_sc is not None else 0,
            agent_sc.span_id,
            f"{warmup_sc.span_id:016x}" if warmup_sc is not None else "none",
            agent_name,
            model or "",
        )
    except Exception:
        pass


def _close_entry_and_agent_for_controller(
    controller: Any, *, error: BaseException | None = None
) -> None:
    """Tear down the ENTRY+AGENT spans previously opened for ``controller``.

    Also closes any STEP span left open from the last ``_step`` invocation
    (STEP spans are intentionally persisted across the return of ``_step``
    so that thread-pooled TOOL / LLM calls fire as their children).
    """
    if not getattr(controller, _OWNS_FLAG, False):
        logger.debug(
            "OpenHands instrumentation: close called on controller %s "
            "without an open ENTRY/AGENT — nothing to do",
            id(controller),
        )
        return
    sid = safe_str(safe_get_attr(controller, "id") or "")
    agent = safe_get_attr(controller, "agent")
    state = safe_get_attr(controller, "state")
    entry_span: trace_api.Span | None = getattr(
        controller, _ENTRY_SPAN_ATTR, None
    )
    agent_span: trace_api.Span | None = getattr(
        controller, _AGENT_SPAN_ATTR, None
    )
    # Legacy slots — kept for back-compat with already-instrumented
    # instances created before we stopped persisting attach-tokens.
    # If they're set we simply ignore them (any detach attempt across
    # asyncio task boundaries would raise ``ValueError`` in the Aliyun
    # SDK; spans alone carry all the parentage info we need).
    _ = getattr(controller, _AGENT_TOKEN_ATTR, None)
    _ = getattr(controller, _ENTRY_TOKEN_ATTR, None)

    # Close any STEP span still hanging from the last round before tearing
    # down AGENT/ENTRY. Restores the session stash to AGENT context so any
    # in-flight TOOL re-attaches AGENT (not a closed STEP).
    try:
        _close_open_step(controller)
    except Exception:
        pass

    # Capture I/O attributes on the AGENT span before ending it.
    if agent_span is not None:
        try:
            _capture_agent_io_attributes(agent_span, controller, agent, state)
        except Exception:
            pass
        try:
            history = safe_get_attr(state, "history") or []
            if isinstance(history, list):
                agent_span.set_attribute(OH_HISTORY_LENGTH, len(history))
            agent_state = safe_get_attr(state, "agent_state")
            if agent_state is not None:
                agent_span.set_attribute(
                    OH_AGENT_STATE,
                    safe_get_attr(agent_state, "value")
                    or safe_str(agent_state),
                )
        except Exception:
            pass
        if error is not None:
            try:
                agent_span.record_exception(error)
                agent_span.set_status(
                    Status(StatusCode.ERROR, type(error).__qualname__)
                )
            except Exception:
                pass

    # End AGENT (no detach — the token (if any) was attached in the
    # ``__init__`` task's contextvars context and detaching here would
    # cross a context boundary, raising ``ValueError`` in the Aliyun
    # SDK. Legacy code may have set ``agent_token`` on older instances;
    # we simply leave it alone — detaching is unnecessary because the
    # span carries its own parentage and contextvars naturally unwind
    # when the task that attached them exits).
    if agent_span is not None:
        try:
            agent_span.end()
        except Exception:
            pass

    # Mirror the most-useful bits onto ENTRY before closing it.
    if entry_span is not None:
        try:
            agent_state = safe_get_attr(state, "agent_state")
            if agent_state is not None:
                entry_span.set_attribute(
                    OH_AGENT_STATE,
                    safe_get_attr(agent_state, "value")
                    or safe_str(agent_state),
                )
            history = safe_get_attr(state, "history") or []
            if isinstance(history, list):
                entry_span.set_attribute(OH_HISTORY_LENGTH, len(history))
            entry_input_messages, entry_output_messages = _entry_io_from_state(
                state
            )
            if entry_input_messages or entry_output_messages:
                _set_io(
                    entry_span,
                    input_messages=entry_input_messages,
                    output_messages=entry_output_messages,
                )
        except Exception:
            pass
        if error is not None:
            try:
                entry_span.record_exception(error)
                entry_span.set_status(
                    Status(StatusCode.ERROR, type(error).__qualname__)
                )
            except Exception:
                pass

    # Same as AGENT: end the span; never touch a possibly-leftover token
    # from an older instrumentation run.
    if entry_span is not None:
        try:
            entry_span.end()
        except Exception:
            pass

    # Mirror the open-time INFO log so the user can confirm the spans
    # actually closed and exported.
    try:
        agent_sc = (
            agent_span.get_span_context() if agent_span is not None else None
        )
        entry_sc = (
            entry_span.get_span_context() if entry_span is not None else None
        )
        logger.info(
            "OpenHands instrumentation: closed ENTRY+AGENT for sid=%r "
            "(entry_span=%s agent_span=%s rounds=%s error=%s)",
            sid,
            f"{entry_sc.span_id:016x}" if entry_sc is not None else "none",
            f"{agent_sc.span_id:016x}" if agent_sc is not None else "none",
            getattr(controller, "_otel_oh_round", 0),
            type(error).__qualname__ if error is not None else "none",
        )
    except Exception:
        pass

    if sid:
        try:
            clear_context(sid)
        except Exception:
            pass

    # Wipe stash slots so a re-used controller instance doesn't double-emit.
    for attr in (
        _OWNS_FLAG,
        _ENTRY_SPAN_ATTR,
        _AGENT_SPAN_ATTR,
        _ENTRY_TOKEN_ATTR,
        _AGENT_TOKEN_ATTR,
        _STEP_SPAN_ATTR,
        _AGENT_CTX_ATTR,
        "_otel_oh_step_consumed",
        "_otel_oh_round",
    ):
        try:
            setattr(controller, attr, None)
        except Exception:
            pass
    try:
        setattr(controller, _OWNS_FLAG, False)
    except Exception:
        pass


class AgentControllerInitWrapper:
    """Open ENTRY + AGENT spans at the end of ``AgentController.__init__``.

    Always reliable under ``python -m openhands.core.main`` because it
    hooks a class method (immune to from-import binding).
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        try:
            result = wrapped(*args, **kwargs)
        except BaseException:
            raise
        try:
            # Skip delegate sub-controllers — they shouldn't open another
            # ENTRY span; they live within the parent controller's trace.
            is_delegate = bool(safe_get_attr(instance, "is_delegate"))
            if is_delegate:
                logger.debug(
                    "OpenHands instrumentation: skipping delegate "
                    "controller %s for ENTRY/AGENT",
                    id(instance),
                )
            else:
                _open_entry_and_agent_for_controller(self._tracer, instance)
        except Exception as exc:
            # Promote to ERROR — if this fails the user will see "no
            # ENTRY span" in their backend and we want a loud signal in
            # the app logs to point at the cause.
            logger.error(
                "OpenHands instrumentation: AgentController init wrapper "
                "failed to open ENTRY/AGENT for controller %s: %s",
                id(instance),
                exc,
                exc_info=True,
            )
        return result


class AgentControllerCloseWrapper:
    """End the ENTRY + AGENT spans previously opened in ``__init__``."""

    __slots__ = ()

    def __init__(self, _tracer: Tracer):
        # Tracer arg unused (we only need the spans we previously opened)
        # but kept for symmetry with the other factories.
        pass

    def __call__(self, wrapped, instance, args, kwargs):
        return self._impl(wrapped, instance, args, kwargs)

    async def _impl(self, wrapped, instance, args, kwargs):
        err: BaseException | None = None
        try:
            return await wrapped(*args, **kwargs)
        except BaseException as exc:
            err = exc
            raise
        finally:
            try:
                _close_entry_and_agent_for_controller(instance, error=err)
            except Exception as exc:
                logger.error(
                    "OpenHands instrumentation: AgentController close "
                    "wrapper failed to end spans for controller %s: %s",
                    id(instance),
                    exc,
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# LLM context bridge: openhands.llm.llm.LLM.__init__
# ---------------------------------------------------------------------------


# Sentinel used to mark already-bridged completion callables so we don't
# wrap them more than once if ``LLM.__init__`` runs again on the same
# completion partial (e.g. live config reload).
_LLM_BRIDGE_FLAG = "_otel_oh_ctx_bridged"


class LLMInitWrapper:
    """Make sure ``LLM.completion`` runs with the current STEP context attached.

    Why this exists
    ---------------
    The LLM call inside ``AgentController._step`` is synchronous and *should*
    inherit our STEP context via ``contextvars`` — but in real OpenHands
    deployments LiteLLM ends up creating its span with a *different*
    ``trace_id`` than the surrounding STEP/AGENT/ENTRY tree. Two known ways
    that can happen:

    * a 3rd-party auto-instrumentation injected before ours stashes the
      LLM call onto a thread-pool worker (no contextvars propagation);
    * the call is made from outside any of our wrappers (e.g. a condenser
      / summarizer worker) where no OTel context is current.

    The fix: at the end of ``LLM.__init__`` we monkey-patch ``self._completion``
    with a tiny shim that re-attaches the latest sid-stashed context (which,
    while a STEP is open, is the STEP context — see ``AgentControllerStepWrapper``).
    The downstream ``opentelemetry-instrumentation-litellm`` (or the Aliyun
    GenAI auto-instrumentation) will then create the LLM span as a child
    of STEP and the ``trace_id`` finally lines up.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        # Tracer arg unused — we only re-attach an existing OTel context
        # so the *real* LLM instrumentor (litellm / aliyun) emits the
        # span under it. We don't create our own LLM span here.
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)
        try:
            self._patch_completion(instance)
        except Exception as exc:
            logger.debug(
                "LLM init wrapper failed to bridge completion: %s", exc
            )
        return result

    @staticmethod
    def _patch_completion(instance: Any) -> None:
        completion = getattr(instance, "_completion", None)
        if completion is None:
            return
        if getattr(completion, _LLM_BRIDGE_FLAG, False):
            return

        def bridged(*a: Any, **kw: Any) -> Any:
            # ``AttachedSession(None)`` re-attaches whatever context the
            # most recent v0 wrapper stashed (STEP if a step is open,
            # AGENT otherwise). When no OpenHands session is active the
            # context manager is a no-op.
            with AttachedSession(None):
                return completion(*a, **kw)

        try:
            setattr(bridged, _LLM_BRIDGE_FLAG, True)
        except Exception:
            pass
        try:
            instance._completion = bridged
        except Exception:
            return
        # Mirror onto the unwrapped slot too — some OpenHands codepaths
        # call ``_completion_unwrapped`` directly when retries are
        # disabled, and we want them to inherit the same parent context.
        unwrapped = getattr(instance, "_completion_unwrapped", None)
        if unwrapped is not None and not getattr(
            unwrapped, _LLM_BRIDGE_FLAG, False
        ):

            def bridged_unwrapped(*a: Any, **kw: Any) -> Any:
                with AttachedSession(None):
                    return unwrapped(*a, **kw)

            try:
                setattr(bridged_unwrapped, _LLM_BRIDGE_FLAG, True)
            except Exception:
                pass
            try:
                instance._completion_unwrapped = bridged_unwrapped
            except Exception:
                pass
