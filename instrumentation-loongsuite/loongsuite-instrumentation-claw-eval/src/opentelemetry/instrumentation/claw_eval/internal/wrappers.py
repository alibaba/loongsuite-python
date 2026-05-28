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

"""Wrapt wrappers for claw-eval OpenTelemetry instrumentation.

Span hierarchy
--------------
ENTRY (cmd_run / cmd_batch / _run_single_task)
└── AGENT (run_task)
    ├── STEP (rotated per main-loop provider.chat call)
    │   ├── TOOL (dispatcher.dispatch / sandbox_dispatcher.dispatch)
    │   ├── CHAIN (do_auto_compact)
    └── (judge.evaluate* + per-task grader._llm_score_classifications:
         nested LLM SDK / HTTP spans suppressed, no span emitted)
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
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

try:
    from aliyun.sdk.extension.arms.semconv import _SUPPRESS_LLM_SDK_KEY
except ImportError:
    _SUPPRESS_LLM_SDK_KEY = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
# ``GEN_AI_TOOL_DEFINITIONS`` was added to the upstream semconv after the
# version vendored by some Aliyun ARMS releases, so we hardcode the spec
# string instead of reading it from ``gen_ai_attributes``.
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

# ---------------------------------------------------------------------------
# ContextVars for STEP lifecycle & compact-depth tracking
# ---------------------------------------------------------------------------

_compact_depth: ContextVar[int] = ContextVar(
    "claw_eval_compact_depth", default=0
)
_in_agent_run: ContextVar[bool] = ContextVar(
    "claw_eval_in_agent_run", default=False
)
_step_counter: ContextVar[int] = ContextVar(
    "claw_eval_step_counter", default=0
)
_current_step_span: ContextVar[Any] = ContextVar(
    "claw_eval_current_step_span", default=None
)
_current_step_token: ContextVar[Any] = ContextVar(
    "claw_eval_current_step_token", default=None
)
_in_tool_dispatch: ContextVar[bool] = ContextVar(
    "claw_eval_in_tool_dispatch", default=False
)

# Per-call capture state for the active AGENT span. ``RunTaskWrapper`` sets a
# fresh dict on entry; the lightweight ``provider.chat`` shim installed below
# pushes data into it. Using a ContextVar keeps concurrent ``run_task``
# invocations isolated even when they share the same provider instance.
_agent_capture: ContextVar["dict[str, Any] | None"] = ContextVar(
    "claw_eval_agent_capture", default=None
)

# JSON-serialized tool-definition list captured from the ``tools=`` kwarg of
# the first ``provider.chat`` call inside an AGENT run. Read by
# ``ToolDispatchWrapper`` to populate ``gen_ai.tool.definitions`` on every
# TOOL span. Stored as a pre-serialized string so each TOOL span pays only an
# attribute-set cost, not a JSON-encode cost.
_agent_tool_definitions: ContextVar[str] = ContextVar(
    "claw_eval_agent_tool_definitions", default=""
)

# Per-CLI-invocation capture for the ENTRY span. ``EntryWrapper`` /
# ``RunSingleTaskWrapper`` initialize a list on entry; each completing AGENT
# span pushes its own capture dict onto it. The first task prompt and the
# final agent response surface as ENTRY ``gen_ai.input.messages`` /
# ``gen_ai.output.messages`` so the trace root carries useful IO.
_entry_capture: ContextVar["list[dict[str, Any]] | None"] = ContextVar(
    "claw_eval_entry_capture", default=None
)

# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def _safe_json(obj: Any) -> str:
    """JSON-serialize ``obj`` for span attributes.

    Content is intentionally NOT truncated: downstream consumers (evaluators,
    SLS analytics) need the full request/response payloads.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _extract_tool_result_text(result) -> str:
    """Extract text content from a ToolResultBlock for gen_ai.tool.call.result.

    Tool output is intentionally NOT truncated so downstream consumers see the
    full payload returned to the agent.
    """
    content = getattr(result, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_system_prompt(messages) -> str:
    """Pull the text content of the first ``role=system`` message."""
    if not messages:
        return ""
    for msg in messages:
        if getattr(msg, "role", None) != "system":
            continue
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text":
                return getattr(block, "text", "") or ""
        break
    return ""


# ---------------------------------------------------------------------------
# Spec-compliant message serialization
# ---------------------------------------------------------------------------
#
# These helpers convert claw-eval's internal ``Message``/``ContentBlock``
# objects into the ARMS GenAI semantic-convention JSON shape documented in
# ``arms_docs/trace/gen-ai.md`` and the message JSON schemas:
#
# * ``gen_ai.input.messages``  — array of ``ChatMessage`` ({role, parts})
# * ``gen_ai.output.messages`` — array of ``OutputMessage``
#                                ({role, parts, finish_reason})
# * ``gen_ai.system_instructions`` — array of parts (TextPart, ...) — note
#                                    that this is *not* wrapped in a message.
#
# Each ``part`` follows the schema:
#   - TextPart:               {"type": "text", "content": ...}
#   - ToolCallRequestPart:    {"type": "tool_call", "id", "name", "arguments"}
#   - ToolCallResponsePart:   {"type": "tool_call_response", "id", "response"}


def _block_to_part(block) -> dict[str, Any]:
    """Convert a claw-eval ContentBlock to a spec-compliant message part."""
    btype = getattr(block, "type", "")
    if btype == "text":
        return {
            "type": "text",
            "content": getattr(block, "text", "") or "",
        }
    if btype == "tool_use":
        return {
            "type": "tool_call",
            "id": getattr(block, "id", "") or "",
            "name": getattr(block, "name", "") or "",
            "arguments": getattr(block, "input", None),
        }
    if btype == "tool_result":
        inner_texts: list[str] = []
        for ib in getattr(block, "content", []) or []:
            t = getattr(ib, "text", None)
            if t:
                inner_texts.append(t)
        return {
            "type": "tool_call_response",
            "id": getattr(block, "tool_use_id", "") or "",
            "response": "\n".join(inner_texts),
        }
    if btype in {"image", "audio", "video"}:
        return {"type": btype}
    return {"type": btype or "unknown"}


def _message_to_chat_message(msg) -> dict[str, Any]:
    """Convert a claw-eval ``Message`` to a spec ``ChatMessage`` dict."""
    role = getattr(msg, "role", "unknown")
    parts = [_block_to_part(b) for b in (getattr(msg, "content", None) or [])]
    return {"role": role, "parts": parts}


def _infer_finish_reason(message) -> str:
    """Infer ``finish_reason`` for an output message.

    The claw-eval ``Message`` returned from ``provider.chat`` does not carry
    the upstream ``finish_reason``; the loop relies on the presence/absence of
    ``tool_use`` blocks to decide whether to keep iterating. We mirror that
    convention here so downstream consumers get a well-formed
    ``OutputMessage``.
    """
    for b in getattr(message, "content", None) or []:
        if getattr(b, "type", "") == "tool_use":
            return "tool_call"
    return "stop"


def _serialize_input_messages(messages) -> str:
    """Serialize a list of input ``Message`` objects to JSON per the spec."""
    arr = [_message_to_chat_message(m) for m in (messages or [])]
    try:
        return json.dumps(arr, ensure_ascii=False, default=str)
    except Exception:
        return str(arr)


def _serialize_output_message(message) -> str:
    """Serialize a single response ``Message`` to a JSON ``OutputMessages`` array."""
    if message is None:
        return ""
    role = getattr(message, "role", "assistant") or "assistant"
    parts = [
        _block_to_part(b) for b in (getattr(message, "content", None) or [])
    ]
    out = {
        "role": role,
        "parts": parts,
        "finish_reason": _infer_finish_reason(message),
    }
    try:
        return json.dumps([out], ensure_ascii=False, default=str)
    except Exception:
        return str([out])


def _serialize_system_instructions(text: str) -> str:
    """Wrap a system prompt string into a JSON ``SystemInstructions`` array."""
    if not text:
        return ""
    arr = [{"type": "text", "content": text}]
    try:
        return json.dumps(arr, ensure_ascii=False, default=str)
    except Exception:
        return str(arr)


def _build_user_text_messages(text: str) -> str:
    """Build a one-message ``InputMessages`` JSON for a plain user prompt."""
    if not text:
        return ""
    arr = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": text}],
        }
    ]
    try:
        return json.dumps(arr, ensure_ascii=False, default=str)
    except Exception:
        return str(arr)


def _serialize_tool_definitions(tools) -> str:
    """Serialize a ``ToolSpec`` iterable as the ``gen_ai.tool.definitions`` JSON.

    Per the GenAI semantic convention each entry is a ``ToolDefinition`` object
    of the form ``{"type": "function", "name": ..., "description": ...,
    "parameters": ...}``. Anything not coercible to that shape is skipped so
    a malformed entry never aborts serialization for the rest of the list.
    """
    if not tools:
        return ""
    arr: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", None)
        if not name:
            continue
        entry: dict[str, Any] = {"type": "function", "name": str(name)}
        desc = getattr(t, "description", None)
        if desc:
            entry["description"] = str(desc)
        # claw-eval names it ``input_schema``; OpenAI / OTel spec uses
        # ``parameters``. Translate so consumers don't have to special-case.
        schema = getattr(t, "input_schema", None)
        if schema is None:
            schema = getattr(t, "parameters", None)
        if schema is not None:
            entry["parameters"] = schema
        arr.append(entry)
    if not arr:
        return ""
    try:
        return json.dumps(arr, ensure_ascii=False, default=str)
    except Exception:
        return str(arr)


# ---------------------------------------------------------------------------
# STEP lifecycle helpers
# ---------------------------------------------------------------------------


def _end_current_step() -> None:
    """End the active STEP span and detach its context token."""
    span = _current_step_span.get(None)
    token = _current_step_token.get(None)
    if span is not None:
        span.end()
        _current_step_span.set(None)
    if token is not None:
        otel_context.detach(token)
        _current_step_token.set(None)


def _rotate_step(tracer: Tracer) -> None:
    """End the previous STEP and start a new one under the current context."""
    _end_current_step()

    step_num = _step_counter.get(0) + 1
    _step_counter.set(step_num)

    step_span = tracer.start_span("react step", kind=SpanKind.INTERNAL)
    step_span.set_attribute(GEN_AI_SPAN_KIND, "STEP")
    step_span.set_attribute(
        GenAI.GEN_AI_OPERATION_NAME,
        GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
    )
    step_span.set_attribute(GEN_AI_FRAMEWORK, "claw-eval")
    step_span.set_attribute(GenAI.GEN_AI_AGENT_NAME, "claw-eval")
    step_span.set_attribute("gen_ai.react.round", step_num)

    _current_step_span.set(step_span)
    ctx = set_span_in_context(step_span)
    token = otel_context.attach(ctx)
    _current_step_token.set(token)


# ---------------------------------------------------------------------------
# ENTRY wrappers (cli.cmd_run / cli.cmd_batch)
# ---------------------------------------------------------------------------


def _populate_entry_span(span, captures: list[dict] | None) -> None:
    """Apply the first task prompt and the last agent response to ENTRY span.

    ENTRY is the trace root for a CLI invocation; representing it with the
    first user prompt and the final agent response gives the span a useful
    summary view without trying to merge potentially conflicting data from
    multiple trials/tasks.
    """
    if not captures:
        return

    # Input: prefer the first agent run's captured input messages (already in
    # spec format); otherwise fall back to its task prompt.
    input_msgs = ""
    for cap in captures:
        input_msgs = cap.get("input_messages_str", "") or ""
        if input_msgs:
            break
    if not input_msgs:
        for cap in captures:
            prompt = cap.get("task_prompt", "") or ""
            if prompt:
                input_msgs = _build_user_text_messages(prompt)
                break
    if input_msgs:
        span.set_attribute(GenAI.GEN_AI_INPUT_MESSAGES, input_msgs)

    # Output: last agent's last response wins (most likely the final answer
    # the user would care about).
    output_msgs = ""
    for cap in reversed(captures):
        output_msgs = cap.get("last_response_str", "") or ""
        if output_msgs:
            break
    if output_msgs:
        span.set_attribute(GenAI.GEN_AI_OUTPUT_MESSAGES, output_msgs)


class EntryWrapper:
    """Creates an ENTRY span around CLI entry-point functions."""

    __slots__ = ("_tracer", "_command")

    def __init__(self, tracer: Tracer, command: str):
        self._tracer = tracer
        self._command = command

    def __call__(self, wrapped, instance, args, kwargs):
        captures: list[dict] = []
        cap_tok = _entry_capture.set(captures)
        with self._tracer.start_as_current_span(
            f"claw-eval {self._command}", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "ENTRY")
            span.set_attribute(GEN_AI_FRAMEWORK, "claw-eval")
            span.set_attribute("claw_eval.command", self._command)
            try:
                return wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            finally:
                _populate_entry_span(span, captures)
                _entry_capture.reset(cap_tok)


class RunSingleTaskWrapper:
    """Creates an ENTRY span for batch worker ``_run_single_task``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        task_dir = args[0] if args else kwargs.get("task_dir", "")
        captures: list[dict] = []
        cap_tok = _entry_capture.set(captures)
        with self._tracer.start_as_current_span(
            "claw-eval batch_worker", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "ENTRY")
            span.set_attribute(GEN_AI_FRAMEWORK, "claw-eval")
            span.set_attribute("claw_eval.command", "batch_worker")
            if task_dir:
                span.set_attribute("claw_eval.task_dir", str(task_dir))
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                if isinstance(result, dict):
                    tid = result.get("task_id")
                    if tid:
                        span.set_attribute("claw_eval.task_id", str(tid))
                return result
            finally:
                _populate_entry_span(span, captures)
                _entry_capture.reset(cap_tok)


# ---------------------------------------------------------------------------
# AGENT wrapper (runner.loop.run_task)
# ---------------------------------------------------------------------------


class RunTaskWrapper:
    """Creates an AGENT span and aggregates per-task GenAI attributes.

    The wrapper installs a lightweight, idempotent shim on ``provider.chat``
    that records the first-call input messages, system prompt, latest response
    and accumulated token usage into a per-call ``_agent_capture`` dict. On
    exit the data is written onto the AGENT span using the OTel GenAI
    semantic conventions (``gen_ai.input.messages``,
    ``gen_ai.output.messages``, ``gen_ai.system_instructions``,
    ``gen_ai.usage.{input,output}_tokens``, ``gen_ai.request.model``).

    ``ProviderChatWrapper`` is intentionally left untouched: the shim wraps
    the *bound* method that already goes through ``ProviderChatWrapper``, so
    STEP rotation continues to work exactly as before.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        task = args[0] if args else kwargs.get("task")
        provider = args[1] if len(args) > 1 else kwargs.get("provider")
        task_id = getattr(task, "task_id", "unknown") if task else "unknown"

        with self._tracer.start_as_current_span(
            "invoke_agent claw-eval", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "AGENT")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
            )
            span.set_attribute(GEN_AI_FRAMEWORK, "claw-eval")
            span.set_attribute(GenAI.GEN_AI_AGENT_NAME, "claw-eval")
            span.set_attribute("claw_eval.task_id", str(task_id))

            model_id = ""
            if provider is not None:
                model_id = str(getattr(provider, "model_id", "") or "")
                if model_id:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, model_id)

            prompt = _get_task_prompt(task)
            if prompt:
                span.set_attribute(
                    GenAI.GEN_AI_AGENT_DESCRIPTION,
                    prompt,
                )

            capture: dict[str, Any] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "system_instructions": "",
                "input_messages_str": "",
                "last_response_str": "",
                "task_prompt": prompt,
                "first_call_done": False,
            }

            _install_provider_chat_capture_shim(provider)

            tok_agent = _in_agent_run.set(True)
            tok_cnt = _step_counter.set(0)
            tok_ss = _current_step_span.set(None)
            tok_st = _current_step_token.set(None)
            tok_cap = _agent_capture.set(capture)
            tok_tools = _agent_tool_definitions.set("")

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                total = _step_counter.get(0)
                if total > 0:
                    span.set_attribute("claw_eval.total_turns", total)
                return result
            finally:
                _populate_agent_span(span, capture, prompt)
                entry_caps = _entry_capture.get()
                if entry_caps is not None:
                    entry_caps.append(capture)
                _end_current_step()
                _in_agent_run.reset(tok_agent)
                _step_counter.reset(tok_cnt)
                _current_step_span.reset(tok_ss)
                _current_step_token.reset(tok_st)
                _agent_capture.reset(tok_cap)
                _agent_tool_definitions.reset(tok_tools)


def _install_provider_chat_capture_shim(provider) -> None:
    """Idempotently install a pass-through shim on ``provider.chat``.

    The shim reads the active capture dict from ``_agent_capture`` and
    records token usage / input messages / latest response into it. When no
    capture is active (e.g. provider used outside an AGENT span) the shim is
    a transparent no-op. Recording is skipped while ``_compact_depth > 0``
    so the AGENT totals match the framework's own ``total_usage`` accounting
    (which excludes auto-compact LLM calls).
    """
    if provider is None:
        return

    existing = provider.__dict__.get("chat")
    if existing is not None and getattr(
        existing, "_claw_eval_capture_shim", False
    ):
        return

    cls = type(provider)
    cls_chat = getattr(cls, "chat", None)
    if cls_chat is None:
        return
    try:
        bound_chat = cls_chat.__get__(provider, cls)
    except Exception:
        return
    if not callable(bound_chat):
        return

    def chat(messages, *call_args, **call_kwargs):
        # Capture the tools list *before* delegating so TOOL spans created
        # inside ``bound_chat`` (none today, but cheap insurance) still see
        # the populated ContextVar. The capture is idempotent — we only
        # serialize once per AGENT run.
        if _compact_depth.get(0) == 0 and not _agent_tool_definitions.get(""):
            tools_arg = call_kwargs.get("tools")
            if tools_arg is None and call_args:
                tools_arg = call_args[0]
            if tools_arg:
                try:
                    serialized = _serialize_tool_definitions(tools_arg)
                except Exception:
                    serialized = ""
                if serialized:
                    _agent_tool_definitions.set(serialized)

        response, usage = bound_chat(messages, *call_args, **call_kwargs)
        capture = _agent_capture.get()
        if capture is None or _compact_depth.get(0) > 0:
            return response, usage

        try:
            capture["input_tokens"] += int(
                getattr(usage, "input_tokens", 0) or 0
            )
            capture["output_tokens"] += int(
                getattr(usage, "output_tokens", 0) or 0
            )
        except Exception:
            pass

        if not capture.get("first_call_done", False):
            capture["first_call_done"] = True
            try:
                capture["system_instructions"] = _extract_system_prompt(
                    messages
                )
                non_system = [
                    m for m in messages if getattr(m, "role", None) != "system"
                ]
                if non_system:
                    capture["input_messages_str"] = _serialize_input_messages(
                        non_system
                    )
            except Exception:
                pass

        try:
            capture["last_response_str"] = _serialize_output_message(response)
        except Exception:
            pass

        return response, usage

    chat._claw_eval_capture_shim = True
    try:
        provider.chat = chat
    except Exception:
        pass


def _populate_agent_span(span, capture: dict, task_prompt: str) -> None:
    """Apply aggregated LLM/token/message data to the AGENT span on exit.

    The GenAI semantic-convention attributes (``gen_ai.input.messages``,
    ``gen_ai.output.messages``, ``gen_ai.system_instructions``,
    ``gen_ai.usage.{input,output}_tokens``) are always written when the data
    has been captured. The AGENT span is the canonical record of a task's IO
    and must surface it now that per-LLM-call spans are suppressed.
    """
    inp = int(capture.get("input_tokens", 0) or 0)
    out = int(capture.get("output_tokens", 0) or 0)
    if inp:
        span.set_attribute(GenAI.GEN_AI_USAGE_INPUT_TOKENS, inp)
    if out:
        span.set_attribute(GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, out)

    sys_prompt = capture.get("system_instructions", "") or ""
    if sys_prompt:
        span.set_attribute(
            GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
            _serialize_system_instructions(sys_prompt),
        )

    input_msgs = capture.get("input_messages_str", "") or ""
    if input_msgs:
        span.set_attribute(GenAI.GEN_AI_INPUT_MESSAGES, input_msgs)
    elif task_prompt:
        span.set_attribute(
            GenAI.GEN_AI_INPUT_MESSAGES,
            _build_user_text_messages(task_prompt),
        )

    last_response_str = capture.get("last_response_str", "") or ""
    if last_response_str:
        span.set_attribute(GenAI.GEN_AI_OUTPUT_MESSAGES, last_response_str)


def _get_task_prompt(task) -> str:
    """Safely extract the prompt text from a TaskDefinition."""
    if task is None:
        return ""
    prompt = getattr(task, "prompt", None)
    if prompt is None:
        return ""
    return getattr(prompt, "text", "") or ""


class ProviderChatWrapper:
    """Rotates STEP spans around main-loop provider chat calls.

    When ``compact_depth == 0`` and inside an agent run, each call ends
    the previous STEP and starts a new one so that subsequent TOOL spans
    become children of the latest STEP.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        compact_depth = _compact_depth.get(0)
        in_agent = _in_agent_run.get(False)

        if in_agent and compact_depth == 0:
            _rotate_step(self._tracer)

        return wrapped(*args, **kwargs)


# ---------------------------------------------------------------------------
# CHAIN wrapper (compact.do_auto_compact)
# ---------------------------------------------------------------------------


class DoAutoCompactWrapper:
    """Creates a CHAIN span and bumps ``_compact_depth``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        focus = kwargs.get("focus")
        layer = "manual" if focus is not None else "auto"

        with self._tracer.start_as_current_span(
            "compact", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "CHAIN")
            span.set_attribute(GEN_AI_FRAMEWORK, "claw-eval")
            span.set_attribute("claw_eval.compact.layer", layer)

            depth_tok = _compact_depth.set(_compact_depth.get(0) + 1)
            try:
                return wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            finally:
                _compact_depth.reset(depth_tok)


# ---------------------------------------------------------------------------
# TOOL wrapper (ToolDispatcher.dispatch / SandboxToolDispatcher.dispatch)
# ---------------------------------------------------------------------------


class ToolDispatchWrapper:
    """Creates a TOOL span for ``dispatch`` calls.

    Uses ``_in_tool_dispatch`` guard to prevent duplicate spans when
    ``SandboxToolDispatcher.dispatch`` delegates to ``ToolDispatcher.dispatch``.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        if _in_tool_dispatch.get(False):
            return wrapped(*args, **kwargs)

        tool_use = args[0] if args else kwargs.get("tool_use")
        tool_name = (
            getattr(tool_use, "name", "unknown") if tool_use else "unknown"
        )
        tool_use_id = getattr(tool_use, "id", "") if tool_use else ""
        tool_input = getattr(tool_use, "input", None) if tool_use else None
        is_sandbox = hasattr(instance, "_http")

        guard = _in_tool_dispatch.set(True)
        with self._tracer.start_as_current_span(
            f"execute_tool {tool_name}", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "TOOL")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )
            span.set_attribute(GEN_AI_FRAMEWORK, "claw-eval")
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(GenAI.GEN_AI_TOOL_TYPE, "function")
            if tool_use_id:
                span.set_attribute(GenAI.GEN_AI_TOOL_CALL_ID, tool_use_id)
            tool_defs = _agent_tool_definitions.get("")
            if tool_defs:
                span.set_attribute(GEN_AI_TOOL_DEFINITIONS, tool_defs)
            if is_sandbox:
                sandbox_url = getattr(instance, "_sandbox_url", None)
                span.set_attribute(
                    "claw_eval.sandbox.remote", sandbox_url is not None
                )
            if tool_input is not None:
                span.set_attribute(
                    GEN_AI_TOOL_CALL_ARGUMENTS,
                    _safe_json(tool_input),
                )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                _extract_dispatch_attrs(span, result)
                return result
            finally:
                _in_tool_dispatch.reset(guard)


def _extract_dispatch_attrs(span, result) -> None:
    """Extract status, latency, and output from the dispatch result tuple."""
    if not isinstance(result, tuple) or len(result) < 2:
        return
    tool_result, dispatch_event = result[0], result[1]
    latency = getattr(dispatch_event, "latency_ms", None)
    if latency is not None:
        span.set_attribute("claw_eval.dispatch.latency_ms", float(latency))
    status = getattr(dispatch_event, "response_status", None)
    if status is not None:
        span.set_attribute("http.response.status_code", int(status))
    if getattr(tool_result, "is_error", False):
        span.set_status(Status(StatusCode.ERROR))
    output_text = _extract_tool_result_text(tool_result)
    if output_text:
        span.set_attribute(GEN_AI_TOOL_CALL_RESULT, output_text)


# ---------------------------------------------------------------------------
# Judge wrapper (LLMJudge.evaluate / evaluate_actions / evaluate_visual)
# ---------------------------------------------------------------------------


class JudgeWrapper:
    """Suppresses nested LLM SDK spans for judge evaluation calls.

    The judge step happens after the agent finishes and is conceptually an
    evaluation/grading concern rather than part of the agent's own reasoning
    trace. Emitting a dedicated LLM span here clutters the trace tail, so we
    intentionally do *not* create a span; we only attach the suppression
    context so the underlying LLM SDK (OpenAI / etc.) does not emit a chat
    span either.
    """

    __slots__ = ("_tracer", "_method_name")

    def __init__(self, tracer: Tracer, method_name: str = "evaluate"):
        self._tracer = tracer
        self._method_name = method_name

    def __call__(self, wrapped, instance, args, kwargs):
        suppress_tok = _maybe_suppress_llm_sdk()
        try:
            return wrapped(*args, **kwargs)
        finally:
            if suppress_tok is not None:
                otel_context.detach(suppress_tok)


# ---------------------------------------------------------------------------
# Per-task grader wrappers
# ---------------------------------------------------------------------------
#
# Per-task graders (``tasks/T*/grader.py``) frequently bypass
# ``LLMJudge.evaluate*`` and call ``judge.client.chat.completions.create``
# directly inside helpers like ``_llm_score_classifications``. Those calls
# would otherwise emit a stray "evaluation" LLM span at the very tail of
# the trace.
#
# Rather than statically enumerating every task module, we hook the two
# loader entry points (``registry.get_grader`` and
# ``base.load_peer_grader``) and then walk the returned class' MRO to wrap
# any matching evaluation-helper methods with ``JudgeWrapper``. This keeps
# coverage automatic for any new task that follows the same naming
# convention.


import wrapt as _wrapt  # local import to avoid widening top-level deps

_GRADER_EVAL_METHOD_NAMES: tuple[str, ...] = ("_llm_score_classifications",)

_GRADER_WRAP_MARKER = "_claw_eval_judge_wrapped"


def _wrap_grader_eval_methods(
    cls,
    tracer: Tracer,
) -> None:
    """Wrap evaluation-helper methods on ``cls`` (and its bases) with JudgeWrapper.

    Idempotent: a marker attribute is set on the wrapped descriptor so the
    same method is never wrapped twice across multiple loads of the same
    class (e.g. peer-grader inheritance chains).
    """
    if cls is None or cls is object:
        return
    for klass in getattr(cls, "__mro__", (cls,)):
        if klass is object:
            continue
        for method_name in _GRADER_EVAL_METHOD_NAMES:
            method = klass.__dict__.get(method_name)
            if method is None:
                continue
            if getattr(method, _GRADER_WRAP_MARKER, False):
                continue
            try:
                wrapper = JudgeWrapper(tracer, method_name)
                wrapped = _wrapt.FunctionWrapper(method, wrapper)
                setattr(wrapped, _GRADER_WRAP_MARKER, True)
                setattr(klass, method_name, wrapped)
            except Exception:
                # Failure here only loses suppression for one method; never
                # let it break grader loading.
                pass


class GetGraderWrapper:
    """Wraps ``claw_eval.graders.registry.get_grader``.

    After the upstream loader returns a grader instance, walk the
    instance's class MRO and wrap evaluation helpers so the inner
    ``judge.client.chat.completions.create`` calls don't emit a trailing
    LLM span.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        grader = wrapped(*args, **kwargs)
        try:
            _wrap_grader_eval_methods(type(grader), self._tracer)
        except Exception:
            pass
        return grader


class LoadPeerGraderWrapper:
    """Wraps ``claw_eval.graders.base.load_peer_grader``.

    Peer graders are loaded lazily at module-import time of a sibling
    task's ``grader.py`` (``_Base = load_peer_grader("T001zh_...")``).
    Wrapping the returned class here ensures the parent-side
    evaluation helpers are suppressed even when subclasses don't override
    them.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        cls = wrapped(*args, **kwargs)
        try:
            _wrap_grader_eval_methods(cls, self._tracer)
        except Exception:
            pass
        return cls


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _maybe_suppress_llm_sdk():
    """Suppress nested LLM SDK / generic instrumentation under the wrapped call.

    Sets two complementary context keys so the suppression covers both:

    * ``_SUPPRESS_LLM_SDK_KEY`` — Aliyun-private key honored by
      ``aliyun-instrumentation-openai``, ``opentelemetry-instrumentation-litellm``
      and ``aliyun-opentelemetry-util-genai``.
    * ``_SUPPRESS_INSTRUMENTATION_KEY`` — the OpenTelemetry standard
      suppression key honored by community/upstream instrumentors
      (httpx, requests, urllib3, etc.). This catches the HTTP-level span
      that would otherwise be emitted for raw judge HTTP calls.
    """
    ctx = otel_context.get_current()
    if _SUPPRESS_LLM_SDK_KEY is not None:
        ctx = otel_context.set_value(_SUPPRESS_LLM_SDK_KEY, True, ctx)
    ctx = otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True, ctx)
    return otel_context.attach(ctx)
