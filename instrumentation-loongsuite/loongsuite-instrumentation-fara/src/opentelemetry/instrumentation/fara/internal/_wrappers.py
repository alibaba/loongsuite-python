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

"""``wrapt`` hooks that emit Fara GenAI spans.

Span hierarchy (per task)::

    ENTRY  enter_ai_application_system             (run_fara_agent)
    └── AGENT  invoke_agent FaraAgent              (FaraAgent.run)
        ├── STEP  react step (round=1)             (generate_model_call)
        │   ├── LLM  chat {model}                  (AsyncCompletions.create — OpenAI instr)
        │   └── TOOL  execute_tool {action}         (execute_action)
        ├── STEP  react step (round=2)
        │   ├── LLM  chat {model}
        │   └── TOOL  execute_tool {action}
        └── ...

The LLM span is produced by ``opentelemetry-instrumentation-openai-v2``
which wraps ``openai.resources.chat.completions.AsyncCompletions.create``
(the exact method Fara calls in ``FaraAgent._make_model_call``). Fara
instrumentation intentionally does **not** create a second LLM span.

All wrapped Fara functions are ``async``, so the wrappers are
``async def`` as well — this lets us ``await wrapped(...)`` so the
span lifecycle brackets the actual coroutine execution rather than
just its creation.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from opentelemetry.instrumentation.fara.config import (
    capture_message_content,
)
from opentelemetry.instrumentation.fara.internal import _state as state
from opentelemetry.instrumentation.fara.internal._attrs import (
    FRAMEWORK_NAME,
    safe_json_dumps,
    tool_description,
    tool_definitions,
    truncate_content,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    OutputMessage,
    Text,
)

logger = logging.getLogger(__name__)


def _make_input_messages(text: str | None) -> list[InputMessage]:
    if not text:
        return []
    return [
        InputMessage(
            role="user",
            parts=[Text(content=truncate_content(str(text)))],
        )
    ]


def _make_output_messages(text: str | None) -> list[OutputMessage]:
    if not text:
        return []
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=truncate_content(str(text)))],
            finish_reason="stop",
        )
    ]


def _close_active_step(handler: ExtendedTelemetryHandler) -> None:
    """Close the currently active STEP span, if any."""
    step_inv = state.get_step_invocation()
    if step_inv is None:
        return
    try:
        handler.stop_react_step(step_inv)
    except Exception as exc:  # noqa: BLE001
        logger.debug("FaraInstrumentor: failed to close STEP span: %s", exc)
    state.set_step_invocation(None)


# ---------------------------------------------------------------------------
# ENTRY: wrap run_fara_agent
# ---------------------------------------------------------------------------


class EntryWrapper:
    """Wrap ``fara.run_fara.run_fara_agent`` as an ENTRY span.

    ``run_fara_agent`` is the CLI / programmatic entry point. We
    generate a ``gen_ai.session.id`` (UUID) here and stash it in a
    ContextVar so the AGENT span can reuse it as ``gen_ai.conversation.id``.
    """

    __slots__ = ("_handler",)

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        initial_task = kwargs.get("initial_task")
        if initial_task is None and args:
            initial_task = args[0]
        session_id = str(uuid.uuid4())

        inv = EntryInvocation(
            session_id=session_id,
            attributes={"gen_ai.framework": FRAMEWORK_NAME},
        )
        if capture_message_content() and initial_task:
            inv.input_messages = _make_input_messages(str(initial_task))

        session_token = state.set_entry_session_id(session_id)
        self._handler.start_entry(inv)
        try:
            result = await wrapped(*args, **kwargs)
            self._handler.stop_entry(inv)
            return result
        except Exception as exc:
            self._handler.fail_entry(
                inv, Error(message=str(exc), type=type(exc))
            )
            raise
        finally:
            state.reset_entry_session_id(session_token)


# ---------------------------------------------------------------------------
# AGENT: wrap FaraAgent.run
# ---------------------------------------------------------------------------


class AgentRunWrapper:
    """Wrap ``FaraAgent.run`` as an AGENT (invoke_agent) span.

    The AGENT span opens before ``run()`` executes and closes after.
    STEP rotation happens inside ``generate_model_call`` (see
    ``GenerateModelCallWrapper``); any still-open STEP is closed in the
    ``finally`` block here so we never leak STEP spans even when
    ``run()`` raises mid-iteration.
    """

    __slots__ = ("_handler",)

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        user_message = kwargs.get("user_message")
        if user_message is None and args:
            user_message = args[0]

        agent_name = type(instance).__name__ if instance is not None else "FaraAgent"
        client_config = getattr(instance, "client_config", None) or {}
        request_model = client_config.get("model") if isinstance(client_config, dict) else None
        conversation_id = state.get_entry_session_id()

        inv = InvokeAgentInvocation(
            provider="openai",
            agent_name=agent_name,
            agent_description="Fara-7B Computer Use Agent",
            conversation_id=conversation_id,
            request_model=request_model,
            attributes={"gen_ai.framework": FRAMEWORK_NAME},
        )
        if capture_message_content() and user_message:
            inv.input_messages = _make_input_messages(str(user_message))

        # Reset STEP rotation state for this agent run.
        step_inv_token = state.set_step_invocation(None)
        step_ctr_token = state.set_step_counter(0)

        self._handler.start_invoke_agent(inv)
        try:
            result = await wrapped(*args, **kwargs)
            # ``run()`` returns (final_answer, all_actions, all_observations).
            if (
                capture_message_content()
                and isinstance(result, tuple)
                and result
            ):
                inv.output_messages = _make_output_messages(str(result[0]))
            self._handler.stop_invoke_agent(inv)
            return result
        except Exception as exc:
            # Stamp the exception type on the still-open STEP so its
            # finish_reason reflects the failure before we close it.
            step_inv = state.get_step_invocation()
            if step_inv is not None and step_inv.finish_reason is None:
                step_inv.finish_reason = type(exc).__qualname__
            self._handler.fail_invoke_agent(
                inv, Error(message=str(exc), type=type(exc))
            )
            raise
        finally:
            # Close any still-open STEP (max_rounds exit, mid-loop
            # exception in non-wrapped code, or normal terminate path
            # that ExecuteActionWrapper already marked).
            step_inv = state.get_step_invocation()
            if step_inv is not None:
                if step_inv.finish_reason is None:
                    # Loop completed without terminate and without
                    # exception -> hit max_rounds.
                    step_inv.finish_reason = "max_rounds"
                _close_active_step(self._handler)
            state.reset_step_invocation(step_inv_token)
            state.reset_step_counter(step_ctr_token)


# ---------------------------------------------------------------------------
# STEP: rotate on generate_model_call
# ---------------------------------------------------------------------------


class GenerateModelCallWrapper:
    """Wrap ``FaraAgent.generate_model_call`` to rotate the STEP span.

    ``generate_model_call`` is called exactly once per ReAct round (at
    the top of each iteration of the for-loop in ``FaraAgent.run``).
    Each call closes the previous STEP (if any) and opens a new one as
    a child of the AGENT span. ``finish_reason`` for the closed STEP is
    derived from its state:

    * Already set (``terminate`` / exception type) -> honoured.
    * Still None -> ``action_complete`` (round ended normally; loop
      continues).
    """

    __slots__ = ("_handler",)

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Close previous STEP (if any). If it has no finish_reason,
        # that means it ended normally and the loop is continuing.
        prev = state.get_step_invocation()
        if prev is not None:
            if prev.finish_reason is None:
                prev.finish_reason = "action_complete"
            _close_active_step(self._handler)

        round_no = state.increment_step_counter()
        step_inv = ReactStepInvocation(round=round_no)
        self._handler.start_react_step(step_inv)
        state.set_step_invocation(step_inv)
        try:
            return await wrapped(*args, **kwargs)
        except Exception as exc:
            # STEP remains open so the AGENT wrapper can record the
            # exception type on it; mark finish_reason now.
            if step_inv.finish_reason is None:
                step_inv.finish_reason = type(exc).__qualname__
            # We do not stop_react_step here — the AGENT wrapper's
            # finally closes it. Re-raise so Fara's own error handling
            # (and our AGENT wrapper) sees the exception.
            raise


# ---------------------------------------------------------------------------
# TOOL: wrap execute_action
# ---------------------------------------------------------------------------


class ToolWrapper:
    """Wrap ``FaraAgent.execute_action`` as a TOOL (execute_tool) span.

    The TOOL span is created as a child of the currently active STEP
    span (automatically via the OTel context set by
    ``GenerateModelCallWrapper``).
    """

    __slots__ = ("_handler",)

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        function_call = args[0] if args else kwargs.get("function_call")
        fc0 = function_call[0] if function_call else None
        action_name = "unknown"
        tool_call_id = None
        tool_args_raw: Any = None
        if fc0 is not None:
            tool_call_id = getattr(fc0, "id", None)
            fc_args = getattr(fc0, "arguments", None)
            if isinstance(fc_args, dict):
                action_name = str(fc_args.get("action") or "unknown")
                tool_args_raw = fc_args

        capture = capture_message_content()
        tool_args_str = (
            safe_json_dumps(tool_args_raw) if capture and tool_args_raw is not None else None
        )

        inv = ExecuteToolInvocation(
            tool_name=action_name,
            tool_call_id=tool_call_id,
            tool_type="browser_action",
            tool_description=tool_description(action_name),
            tool_call_arguments=tool_args_str,
            attributes={"gen_ai.framework": FRAMEWORK_NAME},
        )
        self._handler.start_execute_tool(inv)
        try:
            result = await wrapped(*args, **kwargs)
            # ``execute_action`` returns
            # (is_stop_action, new_screenshot, action_description).
            is_stop = False
            action_desc: str | None = None
            if isinstance(result, tuple) and len(result) >= 3:
                is_stop = bool(result[0])
                action_desc = result[2]
                if isinstance(action_desc, bytes):
                    try:
                        action_desc = action_desc.decode("utf-8", "replace")
                    except Exception:  # noqa: BLE001
                        action_desc = str(action_desc)
                elif not isinstance(action_desc, str):
                    action_desc = str(action_desc) if action_desc is not None else None
            if capture and action_desc:
                inv.tool_call_result = truncate_content(action_desc)

            if is_stop:
                step_inv = state.get_step_invocation()
                if step_inv is not None and step_inv.finish_reason is None:
                    step_inv.finish_reason = "terminate"
            self._handler.stop_execute_tool(inv)
            return result
        except Exception as exc:
            self._handler.fail_execute_tool(
                inv, Error(message=str(exc), type=type(exc))
            )
            step_inv = state.get_step_invocation()
            if step_inv is not None and step_inv.finish_reason is None:
                step_inv.finish_reason = type(exc).__qualname__
            raise


__all__ = [
    "AgentRunWrapper",
    "EntryWrapper",
    "GenerateModelCallWrapper",
    "ToolWrapper",
]
