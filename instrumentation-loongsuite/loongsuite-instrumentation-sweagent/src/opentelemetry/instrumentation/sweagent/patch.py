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

"""SWE-agent hook and agent method wrappers using ExtendedTelemetryHandler."""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace
from typing import Any

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

SWEAGENT_PROVIDER = "sweagent"
SWEAGENT_AGENT_NAME = "swe-agent"
SWEAGENT_AGENT_DESCRIPTION = (
    "SWE-agent autonomous software engineering agent (bash/tools loop)"
)
SWEAGENT_BASH_TOOL_NAME = "sweagent_bash"
_PROBLEM_TEXT_MAX_LEN = 4096

# Links CombinedRunHooks.on_instance_start(problem_statement=...) to
# CombinedAgentHook.on_run_start() on the same thread (no PS in hook kwargs).
_instance_tls = threading.local()


def _tool_name_from_tool_call_item(call: Any) -> str | None:
    """Extract function/tool name from one OpenAI-style tool_calls entry (dict or object)."""
    if call is None:
        return None
    if isinstance(call, dict):
        fn = call.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        name = call.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None
    fn = getattr(call, "function", None)
    if fn is not None:
        name = getattr(fn, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = getattr(call, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def tool_name_from_sweagent_step(step: Any) -> str:
    """Tool name for telemetry from the model-issued tool call list.

    In ``DefaultAgent.forward``, the dict from ``model.query()`` has separate
    fields: ``message`` (assistant text / ``content``) and, when the API uses
    function calling, ``tool_calls``. SWE-agent assigns
    ``step.tool_calls = output["tool_calls"]`` — that **is** the LLM response's
    tool call payload, not a recomputation. The registered tool name lives in
    ``tool_calls[*].function.name``, not in the free-text ``message`` string.

    With ``FunctionCallingParser``, SWE-agent allows **exactly one** tool call per
    model response; ``len(tool_calls) != 1`` raises before ``handle_action``.
    A successful ``handle_action`` therefore normally sees a single entry; the
    loop below only picks the first resolvable name for robustness.

    Without native ``tool_calls`` (e.g. thought/action parsing only), fall back
    to ``sweagent_bash`` because execution still uses bash ``communicate``.
    """
    tool_calls = (
        getattr(step, "tool_calls", None) if step is not None else None
    )
    if not tool_calls:
        return SWEAGENT_BASH_TOOL_NAME
    for call in tool_calls:
        name = _tool_name_from_tool_call_item(call)
        if name:
            return name
    return SWEAGENT_BASH_TOOL_NAME


def _select_tool_call_for_step(step: Any) -> Any | None:
    """Same entry as :func:`tool_name_from_sweagent_step` when possible, else first call."""
    tool_calls = (
        getattr(step, "tool_calls", None) if step is not None else None
    )
    if not tool_calls:
        return None
    for call in tool_calls:
        if _tool_name_from_tool_call_item(call):
            return call
    return tool_calls[0]


def _normalize_function_arguments_from_tool_call(call: Any) -> Any | None:
    """Return ``function.arguments`` parsed as JSON when a string; dict passthrough; else raw."""
    if call is None:
        return None
    if isinstance(call, dict):
        fn = call.get("function")
    else:
        fn = getattr(call, "function", None)
    if fn is None:
        return None
    if isinstance(fn, dict):
        raw = fn.get("arguments")
    else:
        raw = getattr(fn, "arguments", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    return raw


def tool_call_arguments_from_sweagent_step(step: Any) -> Any:
    """Tool call arguments for telemetry: LLM ``function.arguments`` when native ``tool_calls`` exist.

    Otherwise the parsed shell line(s) in ``step.action`` (thought/action and similar paths).
    Structured arguments are preferred when the model used function calling, since ``action`` is
    the command line already expanded by SWE-agent's parser.
    """
    if step is None:
        return None
    fallback = getattr(step, "action", None)
    selected = _select_tool_call_for_step(step)
    if selected is None:
        return fallback
    normalized = _normalize_function_arguments_from_tool_call(selected)
    if normalized is not None:
        return normalized
    return fallback


_RUN_HOOKS_MODULE = "sweagent.run.hooks.abstract"
_AGENT_HOOKS_MODULE = "sweagent.agent.hooks.abstract"
_AGENTS_MODULE = "sweagent.agent.agents"


def _truncate(text: str, max_len: int = _PROBLEM_TEXT_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...<truncated len={len(text)}>"


def _problem_statement_id_and_text(
    problem_statement: Any,
) -> tuple[str | None, str]:
    instance_id = getattr(problem_statement, "id", None)
    text = ""
    try:
        text = problem_statement.get_problem_statement()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        logger.debug("Could not read problem statement text", exc_info=True)
    return instance_id, _truncate(text or "")


def _build_agent_run_summary(info: Any, trajectory: Any) -> str:
    """Human-readable summary from SWE-agent ``info`` + ``trajectory`` (entry / invoke_agent output)."""
    if not isinstance(info, dict):
        info = {}
    traj = trajectory or []
    parts: list[str] = []
    exit_status = info.get("exit_status")
    if exit_status is not None:
        parts.append(f"exit_status={exit_status!r}")
    parts.append(f"trajectory_len={len(traj)}")
    sub = info.get("submission")
    if sub:
        parts.append(f"submission_preview={_truncate(str(sub), 512)!r}")
    ms = info.get("model_stats")
    if ms:
        parts.append(f"model_stats={ms!r}")
    return "\n".join(parts)


def _build_entry_output_summary(result: Any) -> str:
    """Build a text summary for Entry output_messages from AgentRunResult-like object."""
    info = getattr(result, "info", None)
    traj = getattr(result, "trajectory", None)
    return _build_agent_run_summary(info, traj)


def _apply_agent_info_to_invocation(
    inv: InvokeAgentInvocation, info: Any
) -> None:
    """Map SWE-agent ``AgentInfo`` to semconv-oriented invoke_agent fields when present."""
    if not isinstance(info, dict):
        return
    exit_status = info.get("exit_status")
    if exit_status is not None:
        inv.finish_reasons = [str(exit_status)]
    ms = info.get("model_stats")
    if not isinstance(ms, dict):
        return
    ts = ms.get("tokens_sent")
    tr = ms.get("tokens_received")
    if ts is not None:
        try:
            inv.input_tokens = int(ts)
        except (TypeError, ValueError):
            pass
    if tr is not None:
        try:
            inv.output_tokens = int(tr)
        except (TypeError, ValueError):
            pass


def wrap_combined_run_hooks_on_instance_start(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    instance_id, body = _problem_statement_id_and_text(
        kwargs.get("problem_statement")
    )
    inv = EntryInvocation(
        session_id=str(instance_id) if instance_id is not None else None,
        input_messages=[
            InputMessage(role="user", parts=[Text(content=body or "(empty)")])
        ],
    )
    handler.start_entry(inv)
    setattr(instance, "_loongsuite_entry_invocation", inv)
    _instance_tls.problem_statement = kwargs.get("problem_statement")
    try:
        return wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_entry(inv, Error(message=str(exc), type=type(exc)))
        delattr(instance, "_loongsuite_entry_invocation")
        _instance_tls.problem_statement = None
        raise


def wrap_combined_run_hooks_on_instance_completed(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    try:
        return wrapped(*args, **kwargs)
    finally:
        inv = getattr(instance, "_loongsuite_entry_invocation", None)
        if inv is None:
            return
        result = kwargs.get("result")
        summary = (
            _build_entry_output_summary(result)
            if result is not None
            else "(no result)"
        )
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=summary)],
                finish_reason="stop",
            )
        ]
        try:
            handler.stop_entry(inv)
        finally:
            delattr(instance, "_loongsuite_entry_invocation")
            _instance_tls.problem_statement = None


def wrap_combined_agent_hook_on_run_start(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    # Same user message shape as ``EntryInvocation`` (``on_instance_start``).
    ps = getattr(_instance_tls, "problem_statement", None)
    instance_id, body = _problem_statement_id_and_text(ps)
    inv = InvokeAgentInvocation(
        provider=SWEAGENT_PROVIDER,
        agent_name=SWEAGENT_AGENT_NAME,
        agent_description=SWEAGENT_AGENT_DESCRIPTION,
        conversation_id=str(instance_id) if instance_id is not None else None,
        input_messages=[
            InputMessage(role="user", parts=[Text(content=body or "(empty)")])
        ],
    )
    handler.start_invoke_agent(inv)
    setattr(instance, "_loongsuite_invoke_invocation", inv)
    try:
        return wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_invoke_agent(inv, Error(message=str(exc), type=type(exc)))
        delattr(instance, "_loongsuite_invoke_invocation")
        raise


def wrap_combined_agent_hook_on_run_done(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    try:
        return wrapped(*args, **kwargs)
    finally:
        inv = getattr(instance, "_loongsuite_invoke_invocation", None)
        if inv is None:
            return
        # Same summary text as entry ``on_instance_completed`` (``AgentRunResult``-like).
        result_like = SimpleNamespace(
            info=kwargs.get("info"),
            trajectory=kwargs.get("trajectory"),
        )
        summary = _build_entry_output_summary(result_like)
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=summary)],
                finish_reason="stop",
            )
        ]
        _apply_agent_info_to_invocation(inv, result_like.info)
        try:
            handler.stop_invoke_agent(inv)
        finally:
            delattr(instance, "_loongsuite_invoke_invocation")


def wrap_combined_agent_hook_on_step_start(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    round_no = getattr(instance, "_loongsuite_react_round", 0) + 1
    setattr(instance, "_loongsuite_react_round", round_no)
    inv = ReactStepInvocation(round=round_no)
    handler.start_react_step(inv)
    setattr(instance, "_loongsuite_react_invocation", inv)
    try:
        return wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_react_step(inv, Error(message=str(exc), type=type(exc)))
        delattr(instance, "_loongsuite_react_invocation")
        raise


def wrap_combined_agent_hook_on_step_done(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    try:
        return wrapped(*args, **kwargs)
    finally:
        inv = getattr(instance, "_loongsuite_react_invocation", None)
        if inv is None:
            return
        step = kwargs.get("step")
        if step is not None:
            fr = getattr(step, "exit_status", None)
            inv.finish_reason = str(fr) if fr is not None else None
        try:
            handler.stop_react_step(inv)
        finally:
            delattr(instance, "_loongsuite_react_invocation")


def wrap_default_agent_handle_action(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    """Wrap ``handle_action`` so tool spans end on error paths (not always paired hooks)."""
    step = args[0] if args else kwargs.get("step")
    resolved_tool = tool_name_from_sweagent_step(step)
    inv = ExecuteToolInvocation(
        tool_name=resolved_tool,
        provider=SWEAGENT_PROVIDER,
        tool_type="function",
    )
    if step is not None:
        inv.tool_call_arguments = tool_call_arguments_from_sweagent_step(step)
    handler.start_execute_tool(inv)
    try:
        result = wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_execute_tool(inv, Error(message=str(exc), type=type(exc)))
        raise
    if step is not None:
        inv.tool_call_result = getattr(step, "observation", None)
    handler.stop_execute_tool(inv)
    return result


def bind_extended_handler(handler: ExtendedTelemetryHandler, fn):
    return lambda wrapped, instance, args, kwargs: fn(
        handler, wrapped, instance, args, kwargs
    )
