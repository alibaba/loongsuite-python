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

"""``wrapt`` hooks that emit WebArena GenAI spans.

Span hierarchy (per task)::

    ENTRY  webarena_task                       (env.reset)
    └── CHAIN  workflow webarena_task          (env.reset)
         ├── STEP  react step (round=N)        (next_action enter)
         │    ├── AGENT  invoke_agent          (next_action body)
         │    │    ├── TASK  build_prompt_context  (PromptConstructor.construct)
         │    │    └── LLM   chat / text_completion
         │    │              (OpenAI: produced by the OpenAI SDK probe;
         │    │               HuggingFace: produced by this package via
         │    │               ``generate_from_huggingface_completion``)
         │    └── TOOL  execute_tool {action_type}   (env.step)
         └── ...

ENTRY/CHAIN/STEP boundaries are *not* present as discrete functions in
WebArena, so we synthesise them by latching on to:

* ``ScriptBrowserEnv.reset`` — open ENTRY/CHAIN (one task starts)
* ``ScriptBrowserEnv.close`` — close any open spans (batch ends)
* ``PromptAgent.next_action`` — rotate STEP (one ReAct round starts)

A new STEP is closed lazily: by the next ``next_action`` call (next
round) or by ``env.reset`` / ``env.close`` (next task / batch end).
That makes us robust against early-stop / STOP-action paths in
``run.py:test()`` where ``env.step`` is *not* called for the last
round.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.webarena.config import (
    capture_message_content,
)
from opentelemetry.instrumentation.webarena.internal import _state as state
from opentelemetry.instrumentation.webarena.internal._attrs import (
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
    GEN_AI_SPAN_KIND,
    WEBARENA_ACTION_SET_TAG,
    WEBARENA_ACTION_TYPE,
    WEBARENA_BROWSER_ELEMENT_ID,
    WEBARENA_FAIL_ERROR,
    WEBARENA_MEMORY_OBS_TEXT_LENGTH,
    WEBARENA_MEMORY_TRAJECTORY_LENGTH,
    WEBARENA_OBSERVATION_MAIN_TYPE,
    WEBARENA_OBSERVATION_TYPE,
    WEBARENA_PAGE_URL_AFTER,
    WEBARENA_PAGE_URL_BEFORE,
    WEBARENA_PARSING_FAILURE_COUNT,
    WEBARENA_PREVIOUS_ACTION,
    WEBARENA_REQUIRE_LOGIN,
    WEBARENA_SITES,
    WEBARENA_STEP_COUNT,
    WEBARENA_TASK_ID,
    WEBARENA_TOOL_COUNT,
    action_arguments,
    action_type_name,
    messages_to_input_value,
    safe_json_dumps,
    truncate,
    truncate_content,
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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _read_config_file(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Best-effort: load the WebArena task config attached to ``env.reset``."""
    if not options or not isinstance(options, dict):
        return None
    cfg_file = options.get("config_file")
    if not cfg_file:
        return None
    try:
        import json as _json  # noqa: PLC0415

        with open(cfg_file, "r", encoding="utf-8") as f:
            data = _json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    except Exception:  # noqa: BLE001
        return None
    return None


def _set_common_attrs(span: trace_api.Span, kind: str) -> None:
    span.set_attribute(GEN_AI_SPAN_KIND, kind)
    span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)


def _json_dumps(value: Any) -> str:
    """JSON-encode with best-effort fallback."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(value)


# WebArena browser action types as tool definitions for gen_ai.tool.definitions
_BROWSER_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "click",
        "description": "Click on a web element by ID",
    },
    {
        "type": "function",
        "name": "type",
        "description": "Type text into a web element",
    },
    {
        "type": "function",
        "name": "hover",
        "description": "Hover over a web element",
    },
    {
        "type": "function",
        "name": "scroll",
        "description": "Scroll the page up or down",
    },
    {"type": "function", "name": "goto", "description": "Navigate to a URL"},
    {
        "type": "function",
        "name": "go_back",
        "description": "Go back to the previous page",
    },
    {
        "type": "function",
        "name": "go_forward",
        "description": "Go forward to the next page",
    },
    {
        "type": "function",
        "name": "stop",
        "description": "Stop and return the answer",
    },
]


def _set_agent_content_attrs(
    span: trace_api.Span,
    instance: Any,
    intent: str | None,
    meta_data: dict[str, Any],
) -> None:
    """Set gen_ai.input.messages, gen_ai.system_instructions, gen_ai.tool.definitions on AGENT span."""
    try:
        # gen_ai.system_instructions — from PromptConstructor.instruction["intro"]
        pc = getattr(instance, "prompt_constructor", None)
        if pc is not None:
            instruction = getattr(pc, "instruction", None)
            if isinstance(instruction, dict):
                intro = instruction.get("intro", "")
                if intro:
                    span.set_attribute(
                        "gen_ai.system_instructions",
                        _json_dumps(
                            [
                                {
                                    "type": "text",
                                    "content": truncate_content(str(intro)),
                                }
                            ]
                        ),
                    )

        # gen_ai.input.messages — intent as user message
        if intent:
            previous = "None"
            history = meta_data.get("action_history") if meta_data else None
            if isinstance(history, list) and history:
                previous = str(history[-1])
            input_content = f"Task: {intent}\nPrevious action: {previous}"
            span.set_attribute(
                "gen_ai.input.messages",
                _json_dumps(
                    [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "type": "text",
                                    "content": truncate_content(input_content),
                                }
                            ],
                        }
                    ]
                ),
            )

        # gen_ai.tool.definitions — browser action types
        span.set_attribute(
            "gen_ai.tool.definitions",
            _json_dumps(_BROWSER_TOOL_DEFINITIONS),
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# ENTRY / CHAIN lifecycle (driven by ScriptBrowserEnv.reset / .close)
# ---------------------------------------------------------------------------


def _open_task_spans(
    tracer: Tracer,
    options: dict[str, Any] | None,
) -> None:
    """Start ENTRY + CHAIN spans for a fresh WebArena task."""

    # Finalise any spans left open by the previous task (writes summary
    # attributes such as step.count before closing). When called for the
    # very first task this is a no-op.
    _close_task_spans()

    cfg = _read_config_file(options) or {}
    task_id = cfg.get("task_id")
    intent = cfg.get("intent") or ""
    sites = cfg.get("sites") or []
    require_login = bool(cfg.get("storage_state"))

    span_name = (
        f"enter webarena_task {task_id}"
        if task_id is not None
        else "enter webarena_task"
    )
    entry_span = tracer.start_span(span_name, kind=SpanKind.INTERNAL)
    _set_common_attrs(entry_span, "ENTRY")
    entry_span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "enter")
    if task_id is not None:
        entry_span.set_attribute(WEBARENA_TASK_ID, str(task_id))
        try:
            entry_span.set_attribute(
                GenAI.GEN_AI_CONVERSATION_ID, str(task_id)
            )
        except Exception:  # noqa: BLE001
            pass
    if sites:
        entry_span.set_attribute(WEBARENA_SITES, safe_json_dumps(sites))
    entry_span.set_attribute(WEBARENA_REQUIRE_LOGIN, require_login)
    if intent and capture_message_content():
        entry_span.set_attribute(
            "gen_ai.input.messages",
            _json_dumps(
                [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "content": truncate_content(intent),
                            }
                        ],
                    }
                ]
            ),
        )

    entry_token = otel_context.attach(set_span_in_context(entry_span))
    state.set_entry(entry_span, entry_token)

    chain_span = tracer.start_span(
        "workflow webarena_task", kind=SpanKind.INTERNAL
    )
    _set_common_attrs(chain_span, "CHAIN")
    chain_span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "workflow")
    if intent and capture_message_content():
        chain_span.set_attribute("input.value", truncate_content(intent))
    chain_token = otel_context.attach(set_span_in_context(chain_span))
    state.set_chain(chain_span, chain_token)

    state.mark_in_task(True)

    # Stash the resolved task_id on the entry span attributes for later use.
    if task_id is not None:
        try:
            chain_span.set_attribute(WEBARENA_TASK_ID, str(task_id))
        except Exception:  # noqa: BLE001
            pass


def _close_task_spans() -> None:
    """Finalise CHAIN/ENTRY: write summary attributes and call ``end()``."""

    chain = state.get_chain_span()
    entry = state.get_entry_span()
    steps = state.step_count()
    tools = state.tool_count()
    failures = state.parsing_failure_count()
    if chain is not None:
        try:
            chain.set_attribute(WEBARENA_STEP_COUNT, steps)
            chain.set_attribute(WEBARENA_TOOL_COUNT, tools)
            chain.set_attribute(WEBARENA_PARSING_FAILURE_COUNT, failures)
            if capture_message_content():
                chain.set_attribute(
                    "output.value",
                    truncate_content(
                        f"Completed {steps} steps, {tools} tool calls, "
                        f"{failures} parsing failures"
                    ),
                )
        except Exception:  # noqa: BLE001
            pass
    if entry is not None:
        try:
            entry.set_attribute(WEBARENA_STEP_COUNT, steps)
        except Exception:  # noqa: BLE001
            pass
    state.end_task_spans()


# ---------------------------------------------------------------------------
# ScriptBrowserEnv.reset / .close
# ---------------------------------------------------------------------------


class EnvResetWrapper:
    """Open ENTRY+CHAIN spans for a new task on every ``env.reset``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        options = kwargs.get("options")
        _open_task_spans(self._tracer, options)
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            entry = state.get_entry_span()
            if entry is not None:
                try:
                    entry.record_exception(exc)
                    entry.set_status(Status(StatusCode.ERROR))
                except Exception:  # noqa: BLE001
                    pass
            _close_task_spans()
            raise

        # Set gen_ai.output.messages on ENTRY span with initial observation
        if capture_message_content():
            entry = state.get_entry_span()
            if entry is not None:
                try:
                    obs_text = ""
                    if isinstance(result, tuple) and len(result) >= 1:
                        obs = result[0]
                        if isinstance(obs, dict):
                            obs_text = str(obs.get("text", ""))[:512]
                        elif isinstance(obs, str):
                            obs_text = obs[:512]
                    if obs_text:
                        entry.set_attribute(
                            "gen_ai.output.messages",
                            _json_dumps(
                                [
                                    {
                                        "role": "assistant",
                                        "parts": [
                                            {
                                                "type": "text",
                                                "content": obs_text,
                                            }
                                        ],
                                    }
                                ]
                            ),
                        )
                except Exception:  # noqa: BLE001
                    pass

        return result


class EnvCloseWrapper:
    """Close any still-open ENTRY/CHAIN/STEP at end of the batch."""

    __slots__ = ()

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            return wrapped(*args, **kwargs)
        finally:
            _close_task_spans()


# ---------------------------------------------------------------------------
# PromptAgent.next_action  → AGENT(invoke_agent), drives STEP rotation
# ---------------------------------------------------------------------------


def _rotate_step(tracer: Tracer) -> trace_api.Span:
    """End the previous STEP and open a new one as a child of CHAIN."""
    state.end_step()
    round_no = state.increment_step()
    step_span = tracer.start_span("react step", kind=SpanKind.INTERNAL)
    _set_common_attrs(step_span, "STEP")
    step_span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "react")
    step_span.set_attribute(GEN_AI_REACT_ROUND, round_no)
    token = otel_context.attach(set_span_in_context(step_span))
    state.set_step(step_span, token)
    return step_span


class NextActionWrapper:
    """Wrap ``PromptAgent.next_action`` as AGENT(invoke_agent)."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Each call to next_action begins a new ReAct round.
        if state.in_task():
            _rotate_step(self._tracer)

        agent_class = instance.__class__.__name__
        try:
            instr_path = getattr(
                instance.prompt_constructor, "instruction_path", None
            )
            instr_stem = (
                getattr(instr_path, "stem", None) if instr_path else None
            )
        except Exception:  # noqa: BLE001
            instr_stem = None
        agent_name = (
            f"{agent_class}:{instr_stem}" if instr_stem else agent_class
        )
        span_name = f"invoke_agent {agent_class}"

        meta_data: dict[str, Any] = {}
        if len(args) >= 3 and isinstance(args[2], dict):
            meta_data = args[2]
        elif "meta_data" in kwargs and isinstance(kwargs["meta_data"], dict):
            meta_data = kwargs["meta_data"]

        intent: str | None = None
        if len(args) >= 2 and isinstance(args[1], str):
            intent = args[1]
        elif "intent" in kwargs and isinstance(kwargs["intent"], str):
            intent = kwargs["intent"]

        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL
        ) as span:
            _set_common_attrs(span, "AGENT")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
            )
            span.set_attribute(GenAI.GEN_AI_AGENT_NAME, agent_name)
            try:
                lm_cfg = getattr(instance, "lm_config", None)
                if lm_cfg is not None:
                    provider = getattr(lm_cfg, "provider", None)
                    model = getattr(lm_cfg, "model", None)
                    if provider:
                        span.set_attribute(
                            GenAI.GEN_AI_PROVIDER_NAME, str(provider)
                        )
                    if model:
                        span.set_attribute(
                            GenAI.GEN_AI_REQUEST_MODEL, str(model)
                        )
            except Exception:  # noqa: BLE001
                pass

            previous = "None"
            if meta_data:
                history = meta_data.get("action_history")
                if isinstance(history, list) and history:
                    previous = str(history[-1])
            span.set_attribute(WEBARENA_PREVIOUS_ACTION, truncate(previous))

            if capture_message_content():
                _set_agent_content_attrs(span, instance, intent, meta_data)

            try:
                action = wrapped(*args, **kwargs)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute(
                    GEN_AI_REACT_FINISH_REASON, type(exc).__qualname__
                )
                step_span = state.get_step_span()
                if step_span is not None:
                    try:
                        step_span.set_attribute(
                            GEN_AI_REACT_FINISH_REASON,
                            type(exc).__qualname__,
                        )
                        step_span.set_status(Status(StatusCode.ERROR))
                    except Exception:  # noqa: BLE001
                        pass
                raise

            # Successful next_action — record action info and propagate to STEP.
            atype = action_type_name(action)
            span.set_attribute(WEBARENA_ACTION_TYPE, atype)
            raw_pred = (
                action.get("raw_prediction")
                if isinstance(action, dict)
                else None
            )
            if capture_message_content():
                if raw_pred:
                    span.set_attribute(
                        "gen_ai.output.messages",
                        _json_dumps(
                            [
                                {
                                    "role": "assistant",
                                    "parts": [
                                        {
                                            "type": "text",
                                            "content": truncate_content(
                                                str(raw_pred)
                                            ),
                                        }
                                    ],
                                    "finish_reason": "stop"
                                    if atype == "STOP"
                                    else "action",
                                }
                            ]
                        ),
                    )

            if atype == "NONE":
                state.increment_parsing_failure()

            step_span = state.get_step_span()
            if step_span is not None:
                try:
                    step_span.set_attribute(WEBARENA_ACTION_TYPE, atype)
                    if atype == "STOP":
                        step_span.set_attribute(
                            GEN_AI_REACT_FINISH_REASON, "stop"
                        )
                    elif atype == "NONE":
                        step_span.set_attribute(
                            GEN_AI_REACT_FINISH_REASON, "parse_failure"
                        )
                except Exception:  # noqa: BLE001
                    pass

            return action


# ---------------------------------------------------------------------------
# PromptConstructor.construct  →  TASK(build_prompt_context)
# ---------------------------------------------------------------------------


class PromptConstructWrapper:
    """Emit a TASK span for each prompt-construction call."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        trajectory = args[0] if len(args) >= 1 else kwargs.get("trajectory")
        intent = args[1] if len(args) >= 2 else kwargs.get("intent")
        meta_data = (
            args[2] if len(args) >= 3 else kwargs.get("meta_data") or {}
        )

        with self._tracer.start_as_current_span(
            "run_task build_prompt_context", kind=SpanKind.INTERNAL
        ) as span:
            _set_common_attrs(span, "TASK")
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
            span.set_attribute("webarena.task.name", "build_prompt_context")

            try:
                if trajectory is not None:
                    span.set_attribute(
                        WEBARENA_MEMORY_TRAJECTORY_LENGTH,
                        int(len(trajectory)),
                    )
            except Exception:  # noqa: BLE001
                pass

            previous = "None"
            if isinstance(meta_data, dict):
                history = meta_data.get("action_history")
                if isinstance(history, list) and history:
                    previous = str(history[-1])

            url_before = ""
            try:
                if (
                    trajectory is not None
                    and len(trajectory) > 0
                    and isinstance(trajectory[-1], dict)
                ):
                    info = trajectory[-1].get("info") or {}
                    page = info.get("page") if isinstance(info, dict) else None
                    if page is not None and getattr(page, "url", None):
                        url_before = str(page.url)
            except Exception:  # noqa: BLE001
                pass

            if capture_message_content():
                input_summary = {
                    "intent": str(intent) if intent else "",
                    "url": url_before,
                    "previous_action": previous,
                }
                span.set_attribute(
                    "input.value", safe_json_dumps(input_summary)
                )
                span.set_attribute("input.mime_type", "application/json")

            try:
                prompt = wrapped(*args, **kwargs)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            try:
                if isinstance(prompt, list):
                    span.set_attribute(
                        "webarena.prompt.messages_count", len(prompt)
                    )
                elif isinstance(prompt, str):
                    span.set_attribute("webarena.prompt.length", len(prompt))
            except Exception:  # noqa: BLE001
                pass

            try:
                obs_modality = getattr(instance, "obs_modality", None)
                if (
                    obs_modality
                    and trajectory is not None
                    and len(trajectory) > 0
                    and isinstance(trajectory[-1], dict)
                ):
                    obs = trajectory[-1].get("observation")
                    if isinstance(obs, dict) and obs_modality in obs:
                        span.set_attribute(
                            WEBARENA_MEMORY_OBS_TEXT_LENGTH,
                            int(len(obs[obs_modality])),
                        )
            except Exception:  # noqa: BLE001
                pass

            if capture_message_content():
                span.set_attribute(
                    "output.value", messages_to_input_value(prompt)
                )
                span.set_attribute("output.mime_type", "application/json")
            return prompt


# ---------------------------------------------------------------------------
# ScriptBrowserEnv.step  →  TOOL(execute_tool)
# ---------------------------------------------------------------------------


class EnvStepWrapper:
    """Wrap a single browser action execution as a TOOL span."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        action = args[0] if args else kwargs.get("action")
        atype = action_type_name(action)

        url_before = ""
        try:
            page = getattr(instance, "page", None)
            if page is not None and getattr(page, "url", None):
                url_before = str(page.url)
        except Exception:  # noqa: BLE001
            pass

        with self._tracer.start_as_current_span(
            f"execute_tool {atype}", kind=SpanKind.INTERNAL
        ) as span:
            _set_common_attrs(span, "TOOL")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, atype)
            span.set_attribute(GenAI.GEN_AI_TOOL_TYPE, "browser_action")
            if url_before:
                span.set_attribute(
                    WEBARENA_PAGE_URL_BEFORE, truncate(url_before)
                )

            try:
                main_obs = getattr(instance, "main_observation_type", None)
                if main_obs:
                    span.set_attribute(
                        WEBARENA_OBSERVATION_MAIN_TYPE, str(main_obs)
                    )
            except Exception:  # noqa: BLE001
                pass

            if isinstance(action, dict):
                eid = action.get("element_id")
                if eid:
                    span.set_attribute(WEBARENA_BROWSER_ELEMENT_ID, str(eid))

            if capture_message_content():
                span.set_attribute(
                    GenAI.GEN_AI_TOOL_CALL_ARGUMENTS,
                    safe_json_dumps(action_arguments(action)),
                )

            state.increment_tool()

            try:
                result = wrapped(*args, **kwargs)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            url_after = ""
            try:
                page = getattr(instance, "page", None)
                if page is not None and getattr(page, "url", None):
                    url_after = str(page.url)
            except Exception:  # noqa: BLE001
                pass
            if url_after:
                span.set_attribute(
                    WEBARENA_PAGE_URL_AFTER, truncate(url_after)
                )

            success = False
            fail_error = ""
            terminated = False
            if isinstance(result, tuple) and len(result) >= 5:
                try:
                    success = bool(result[1])
                    terminated = bool(result[2])
                    info = result[4] or {}
                    if isinstance(info, dict):
                        fail_error = str(info.get("fail_error") or "")
                except Exception:  # noqa: BLE001
                    pass

            span.set_attribute("webarena.tool.success", success)
            if fail_error:
                span.set_attribute(WEBARENA_FAIL_ERROR, truncate(fail_error))
                span.set_status(Status(StatusCode.ERROR, fail_error))

            if capture_message_content():
                span.set_attribute(
                    GenAI.GEN_AI_TOOL_CALL_RESULT,
                    safe_json_dumps(
                        {
                            "success": success,
                            "fail_error": fail_error,
                            "url_after": url_after,
                            "terminated": terminated,
                        }
                    ),
                )

            step_span = state.get_step_span()
            if step_span is not None and terminated:
                try:
                    step_span.set_attribute(
                        GEN_AI_REACT_FINISH_REASON, "terminated"
                    )
                except Exception:  # noqa: BLE001
                    pass

            return result


# ---------------------------------------------------------------------------
# construct_agent  →  AGENT(create_agent)
# ---------------------------------------------------------------------------


class ConstructAgentWrapper:
    """Wrap the agent factory as a one-shot AGENT(create_agent) span."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        ns_args = args[0] if args else kwargs.get("args")
        agent_type = getattr(ns_args, "agent_type", None) or "unknown"
        provider = getattr(ns_args, "provider", None) or ""
        model = getattr(ns_args, "model", None) or ""
        instr_path = getattr(ns_args, "instruction_path", None) or ""
        action_set = getattr(ns_args, "action_set_tag", None) or ""

        with self._tracer.start_as_current_span(
            f"create_agent {FRAMEWORK_NAME}", kind=SpanKind.INTERNAL
        ) as span:
            _set_common_attrs(span, "AGENT")
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "create_agent")
            span.set_attribute(
                GenAI.GEN_AI_AGENT_NAME,
                truncate(f"{agent_type}:{instr_path}"),
            )
            span.set_attribute(
                GenAI.GEN_AI_AGENT_DESCRIPTION,
                truncate(
                    f"provider={provider}, model={model}, action_set={action_set}"
                ),
            )
            try:
                aid = hashlib.md5(
                    f"{provider}:{model}:{instr_path}:{action_set}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:16]
                span.set_attribute(GenAI.GEN_AI_AGENT_ID, aid)
            except Exception:  # noqa: BLE001
                pass
            if provider:
                span.set_attribute(GenAI.GEN_AI_PROVIDER_NAME, str(provider))
            if model:
                span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, str(model))
            if action_set:
                span.set_attribute(WEBARENA_ACTION_SET_TAG, str(action_set))
            obs_type = getattr(ns_args, "observation_type", None)
            if obs_type:
                span.set_attribute(WEBARENA_OBSERVATION_TYPE, str(obs_type))

            if capture_message_content():
                # gen_ai.system_instructions from instruction file
                try:
                    if instr_path:
                        import pathlib  # noqa: PLC0415

                        p = pathlib.Path(instr_path)
                        if p.exists():
                            with open(p, "r", encoding="utf-8") as f:
                                instr_data = json.load(f)
                            intro = instr_data.get("intro", "")
                            if intro:
                                span.set_attribute(
                                    "gen_ai.system_instructions",
                                    _json_dumps(
                                        [
                                            {
                                                "type": "text",
                                                "content": truncate_content(
                                                    str(intro)
                                                ),
                                            }
                                        ]
                                    ),
                                )
                except Exception:  # noqa: BLE001
                    pass
                # gen_ai.tool.definitions
                span.set_attribute(
                    "gen_ai.tool.definitions",
                    _json_dumps(_BROWSER_TOOL_DEFINITIONS),
                )

            try:
                result = wrapped(*args, **kwargs)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            return result


# ---------------------------------------------------------------------------
# generate_from_huggingface_completion  →  LLM(text_completion)
# ---------------------------------------------------------------------------


class HuggingFaceCompletionWrapper:
    """LLM span for the only WebArena LLM call **not** going through OpenAI SDK."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Signature:
        # generate_from_huggingface_completion(
        #     prompt, model_endpoint, temperature, top_p, max_new_tokens,
        #     stop_sequences=None,
        # )
        def _arg(idx: int, name: str, default: Any = None) -> Any:
            if len(args) > idx:
                return args[idx]
            return kwargs.get(name, default)

        prompt = _arg(0, "prompt", "")
        model_endpoint = _arg(1, "model_endpoint", "")
        temperature = _arg(2, "temperature")
        top_p = _arg(3, "top_p")
        max_new_tokens = _arg(4, "max_new_tokens")
        stop_sequences = _arg(5, "stop_sequences")

        span_name = f"text_completion {model_endpoint or 'huggingface'}"
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT
        ) as span:
            _set_common_attrs(span, "LLM")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.TEXT_COMPLETION.value,
            )
            span.set_attribute(GenAI.GEN_AI_PROVIDER_NAME, "huggingface")
            if model_endpoint:
                span.set_attribute(
                    GenAI.GEN_AI_REQUEST_MODEL, str(model_endpoint)
                )
                span.set_attribute(
                    GenAI.GEN_AI_RESPONSE_MODEL, str(model_endpoint)
                )
            try:
                if temperature is not None:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_TEMPERATURE, float(temperature)
                    )
                if top_p is not None:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_TOP_P, float(top_p)
                    )
                if max_new_tokens is not None:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_MAX_TOKENS, int(max_new_tokens)
                    )
            except (TypeError, ValueError):
                pass
            if stop_sequences:
                try:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_STOP_SEQUENCES,
                        list(stop_sequences),
                    )
                except Exception:  # noqa: BLE001
                    pass
            if (
                capture_message_content()
                and isinstance(prompt, str)
                and prompt
            ):
                span.set_attribute("input.value", truncate_content(prompt))

            try:
                generation = wrapped(*args, **kwargs)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            if capture_message_content() and isinstance(generation, str):
                span.set_attribute(
                    "output.value", truncate_content(generation)
                )
            span.set_attribute("gen_ai.output.type", "text")

            return generation


__all__ = [
    "ConstructAgentWrapper",
    "EnvCloseWrapper",
    "EnvResetWrapper",
    "EnvStepWrapper",
    "HuggingFaceCompletionWrapper",
    "NextActionWrapper",
    "PromptConstructWrapper",
]
