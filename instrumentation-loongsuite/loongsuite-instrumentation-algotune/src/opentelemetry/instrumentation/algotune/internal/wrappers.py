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

"""Wrapt wrappers for AlgoTune OpenTelemetry instrumentation.

Span hierarchy (final selection)::

    ENTRY: enter_ai_application_system          ← AlgoTuner.main:main()
    └── AGENT: invoke_agent AlgoTuner           ← LLMInterface.run_task()
        ├── STEP: react step  [round=N]         ← get_response + handle_function_call
        │   ├── LLM:  chat <model>              ← LiteLLM instrumentor (auto)
        │   │                                     OR TogetherModel.query (this pkg)
        │   └── TOOL: execute_tool <command>    ← CommandHandlers.handle_command
        │       └── TASK: run_task benchmark.dataset_eval ← _runner_eval_dataset
        │           ├── TASK: run_task benchmark.baseline_generation ← get_baseline_times
        │           └── TASK: run_task benchmark.problem_eval [×N] ← evaluate_single
        └── ...

This module never creates LLM spans for the LiteLLM path. The LiteLLM
instrumentor (loaded separately at runtime) is responsible for that and
naturally becomes a child of the active STEP span via OpenTelemetry
context propagation. The only LLM-layer hook here is a lightweight
attempt counter (``algo.llm.retry_count``) written onto the STEP span.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Callable, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.algotune.config import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.instrumentation.algotune.internal.utils import (
    ALGOTUNE_FRAMEWORK_VALUE,
    GEN_AI_FRAMEWORK,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    INST_LITELLM_ATTEMPTS_ATTR,
    INST_ROUND_ATTR,
    INST_STEP_SPAN_ATTR,
    INST_STEP_TOKEN_ATTR,
    provider_from_model,
    safe_close_step,
    truncate,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
)

logger = logging.getLogger(__name__)


def _algotune_capture_span_content_enabled() -> bool:
    raw = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "")
    return raw.strip().upper() in {
        "TRUE",
        "1",
        "YES",
        "ON",
        "SPAN_ONLY",
        "SPAN_AND_EVENT",
    }


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(value)


def _span_message(role: str, content: Any) -> dict[str, Any]:
    return {
        "role": role or "user",
        "parts": [{"type": "text", "content": truncate(_text_value(content))}],
    }


def _algotune_tool_definitions() -> list[dict[str, Any]]:
    try:
        from AlgoTuner.interfaces.commands.types import (  # noqa: PLC0415
            COMMAND_FORMATS,
        )
    except Exception:  # noqa: BLE001
        return []

    definitions: list[dict[str, Any]] = []
    for name, fmt in COMMAND_FORMATS.items():
        description = (
            getattr(fmt, "description", "") or f"AlgoTune command {name}"
        )
        example = getattr(fmt, "example", "") or ""
        definitions.append(
            {
                "type": "function",
                "name": str(name),
                "description": truncate(str(description)),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": truncate(
                                str(example).strip() or str(description)
                            ),
                        }
                    },
                    "required": ["command"],
                },
            }
        )
    return definitions


def _agent_content_attributes(instance: Any) -> dict[str, Any]:
    if not _algotune_capture_span_content_enabled():
        return {}

    state = getattr(instance, "state", None)
    messages = list(getattr(state, "messages", None) or [])
    input_messages: list[dict[str, Any]] = []
    output_messages: list[dict[str, Any]] = []
    system_instructions: list[dict[str, Any]] = []

    for msg in messages[-20:]:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "assistant":
            output_messages.append(_span_message("assistant", content))
        elif role == "system":
            system_instructions.append(
                {"type": "text", "content": truncate(_text_value(content))}
            )
        else:
            input_messages.append(_span_message(role, content))

    # AlgoTune puts its application instructions in the first user message.
    # Surface that separately for UIs that render system instructions.
    if not system_instructions and messages:
        first = messages[0]
        if isinstance(first, dict) and first.get("content"):
            system_instructions.append(
                {
                    "type": "text",
                    "content": truncate(_text_value(first.get("content"))),
                }
            )

    tool_definitions = _algotune_tool_definitions()
    attrs: dict[str, Any] = {
        "algo.debug.input_messages.count": len(input_messages),
        "algo.debug.output_messages.count": len(output_messages),
        "algo.debug.system_instructions.count": len(system_instructions),
        "algo.debug.tool_definitions.count": len(tool_definitions),
    }

    # Keep parent span output compact; large parent attributes are commonly
    # harder to render than LLM child attributes in trace UIs.
    output_payload = output_messages[-1:] if output_messages else []
    attrs["gen_ai.output.messages"] = json.dumps(
        output_payload, ensure_ascii=False, default=str
    )
    if output_payload:
        try:
            attrs["output.value"] = truncate(
                _text_value(output_payload[-1]["parts"][0].get("content", ""))
            )
        except Exception:  # noqa: BLE001
            pass

    if input_messages:
        attrs["gen_ai.input.messages"] = json.dumps(
            input_messages[-6:], ensure_ascii=False, default=str
        )
    if system_instructions:
        attrs["gen_ai.system_instructions"] = json.dumps(
            system_instructions[:1], ensure_ascii=False, default=str
        )
    if tool_definitions:
        attrs["gen_ai.tool.definitions"] = json.dumps(
            tool_definitions, ensure_ascii=False, default=str
        )
    return attrs


def _publish_agent_content_attributes(instance: Any, *spans: Span) -> None:
    attrs = _agent_content_attributes(instance)
    if not attrs:
        return
    for span in spans:
        try:
            if span is not None and span.is_recording():
                span.set_attributes(attrs)
        except Exception:  # noqa: BLE001
            pass


def _task_json_value(value: Any) -> str:
    try:
        return truncate(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        return truncate(str(value))


def _set_task_input(span: Span, value: Any) -> None:
    span.set_attribute("input.mime_type", "application/json")
    span.set_attribute("input.value", _task_json_value(value))


def _set_task_output(span: Span, value: Any) -> None:
    span.set_attribute("output.mime_type", "application/json")
    span.set_attribute("output.value", _task_json_value(value))


# ---------------------------------------------------------------------------
# ENTRY: AlgoTuner.main.main()
# ---------------------------------------------------------------------------


class MainWrapper:
    """ENTRY span around ``AlgoTuner.main.main()``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        session_id = uuid.uuid4().hex
        argv_repr = ""
        try:
            argv_repr = " ".join(map(str, sys.argv[1:8]))
        except Exception:  # noqa: BLE001
            pass

        with self._tracer.start_as_current_span(
            "enter_ai_application_system", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "ENTRY")
            span.set_attribute("gen_ai.operation.name", "enter")
            span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
            span.set_attribute("gen_ai.session.id", session_id)
            if argv_repr:
                span.set_attribute(
                    "algotune.invocation.argv", truncate(argv_repr)
                )

            # Best-effort: pull --model and --task out of sys.argv so the
            # ENTRY span carries the user's intent before main() finishes.
            try:
                argv = list(sys.argv[1:])
                for i, tok in enumerate(argv):
                    if tok == "--model" and i + 1 < len(argv):
                        span.set_attribute(
                            GenAI.GEN_AI_REQUEST_MODEL, argv[i + 1]
                        )
                    elif tok == "--task" and i + 1 < len(argv):
                        span.set_attribute("algo.task.name", argv[i + 1])
            except Exception:  # noqa: BLE001
                pass

            try:
                return wrapped(*args, **kwargs)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 0
                if code:
                    span.set_attribute("algotune.exit_code", int(code))
                    span.set_status(
                        Status(StatusCode.ERROR, f"sys.exit({code})")
                    )
                raise
            except MemoryError as exc:
                span.set_attribute("error.type", "MemoryError")
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, "MemoryError"))
                raise
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise


# ---------------------------------------------------------------------------
# AGENT: LLMInterface.run_task()
# ---------------------------------------------------------------------------


class RunTaskWrapper:
    """AGENT span around ``LLMInterface.run_task()``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Reset round counter at the beginning of each AGENT invocation.
        try:
            setattr(instance, INST_ROUND_ATTR, 0)
            setattr(instance, INST_STEP_SPAN_ATTR, None)
            setattr(instance, INST_STEP_TOKEN_ATTR, None)
        except Exception:  # noqa: BLE001
            pass

        model_name = str(getattr(instance, "model_name", "") or "")
        parent_span = trace_api.get_current_span()

        with self._tracer.start_as_current_span(
            "invoke_agent AlgoTuner", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "AGENT")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
            )
            span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
            span.set_attribute(GenAI.GEN_AI_AGENT_NAME, "AlgoTuner")
            span.set_attribute(
                GenAI.GEN_AI_AGENT_DESCRIPTION,
                "Iterative code optimization agent for benchmark tasks",
            )
            if model_name:
                span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, model_name)
                span.set_attribute(
                    GenAI.GEN_AI_PROVIDER_NAME,
                    provider_from_model(model_name),
                )

            terminated_reason: str = "unknown"
            try:
                result = wrapped(*args, **kwargs)
                terminated_reason = self._infer_termination_reason(instance)
                return result
            except (KeyboardInterrupt, SystemExit) as exc:
                terminated_reason = type(exc).__qualname__
                if isinstance(exc, SystemExit):
                    code = exc.code if isinstance(exc.code, int) else 0
                    if code:
                        span.set_status(
                            Status(StatusCode.ERROR, f"sys.exit({code})")
                        )
                raise
            except Exception as exc:
                terminated_reason = "exception"
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            finally:
                # Always close any dangling STEP span first so the trace tree
                # never has STEP outliving AGENT.
                safe_close_step(instance)

                rounds = int(getattr(instance, INST_ROUND_ATTR, 0) or 0)
                span.set_attribute("algo.agent.total_rounds", rounds)
                span.set_attribute(
                    "algo.agent.final_status", terminated_reason
                )
                _publish_agent_content_attributes(instance, span, parent_span)

                # Spend / final eval bookkeeping (best-effort; AlgoTune may
                # have torn the interface down by now).
                try:
                    state = getattr(instance, "state", None)
                    if state is not None:
                        spend = getattr(state, "spend", None)
                        if spend is not None:
                            span.set_attribute(
                                "algo.agent.spend_usd", float(spend)
                            )
                except Exception:  # noqa: BLE001
                    pass

                try:
                    final_success = getattr(
                        instance, "_final_eval_success", None
                    )
                    if final_success is not None:
                        span.set_attribute(
                            "algo.agent.final_eval_success",
                            bool(final_success),
                        )
                    final_eval_result = getattr(
                        instance, "_final_eval_metrics", None
                    )
                    if isinstance(final_eval_result, dict):
                        ms = final_eval_result.get("mean_speedup")
                        if ms is not None:
                            try:
                                span.set_attribute(
                                    "algo.agent.final_mean_speedup", float(ms)
                                )
                            except (TypeError, ValueError):
                                pass
                except Exception:  # noqa: BLE001
                    pass

                span.add_event(
                    "agent.loop.terminated",
                    {"reason": terminated_reason},
                )

    @staticmethod
    def _infer_termination_reason(instance: Any) -> str:
        # Heuristics that align with the loop logic in
        # LLMInterface.run_task() (line 996+).
        try:
            check = getattr(instance, "check_limits", None)
            if callable(check) and check():
                return "terminated_by_limit"
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(instance, "_final_eval_success", False):
                return "completed"
        except Exception:  # noqa: BLE001
            pass
        return "completed"


# ---------------------------------------------------------------------------
# STEP: LLMInterface.get_response() + handle_function_call()
# ---------------------------------------------------------------------------


class GetResponseWrapper:
    """Open a STEP span when ``get_response`` starts a new react round."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Close any previously opened STEP span before starting a new one
        # (covers the empty-response retry path where the loop ``continue``s
        # without invoking handle_function_call).
        safe_close_step(instance)

        round_n = int(getattr(instance, INST_ROUND_ATTR, 0) or 0) + 1
        try:
            setattr(instance, INST_ROUND_ATTR, round_n)
            setattr(instance, INST_LITELLM_ATTEMPTS_ATTR, 0)
        except Exception:  # noqa: BLE001
            pass

        span = self._tracer.start_span("react step", kind=SpanKind.INTERNAL)
        span.set_attribute(GEN_AI_SPAN_KIND, "STEP")
        span.set_attribute("gen_ai.operation.name", "react")
        span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
        span.set_attribute("gen_ai.react.round", round_n)

        ctx = set_span_in_context(span)
        token = otel_context.attach(ctx)
        try:
            setattr(instance, INST_STEP_SPAN_ATTR, span)
            setattr(instance, INST_STEP_TOKEN_ATTR, token)
        except Exception:  # noqa: BLE001
            pass

        try:
            response = wrapped(*args, **kwargs)
        except BaseException as exc:
            span.set_attribute(
                "gen_ai.react.finish_reason", type(exc).__qualname__
            )
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            self._publish_attempt_count(instance, span)
            try:
                span.end()
            finally:
                otel_context.detach(token)
                _clear_step_state(instance)
            raise

        if response is None:
            span.set_attribute("algo.step.response_empty", True)
            span.set_attribute(
                "gen_ai.react.finish_reason", "empty_response_retry"
            )
            self._publish_attempt_count(instance, span)
            try:
                span.end()
            finally:
                otel_context.detach(token)
                _clear_step_state(instance)
            return response

        # Non-empty response: STEP stays open, handle_function_call wrapper
        # is responsible for closing it.
        return response

    @staticmethod
    def _publish_attempt_count(instance: Any, span: Span) -> None:
        try:
            attempts = int(
                getattr(instance, INST_LITELLM_ATTEMPTS_ATTR, 0) or 0
            )
            if attempts:
                span.set_attribute("algo.llm.retry_count", attempts)
        except Exception:  # noqa: BLE001
            pass


class HandleFunctionCallWrapper:
    """Close the STEP span opened by ``GetResponseWrapper`` after the tool
    call (or its error path) completes."""

    __slots__ = ()

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        span: Optional[Span] = getattr(instance, INST_STEP_SPAN_ATTR, None)
        token = getattr(instance, INST_STEP_TOKEN_ATTR, None)

        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            if span is not None and span.is_recording():
                span.set_attribute(
                    "gen_ai.react.finish_reason", type(exc).__qualname__
                )
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
            self._close_step(instance, span, token)
            raise

        if span is not None and span.is_recording():
            # finish_reason recorded based on result shape
            cmd_name = _extract_command_name(result)
            if cmd_name:
                span.set_attribute("algo.step.command_name", cmd_name)
            span.set_attribute("gen_ai.react.finish_reason", "tool_executed")
            try:
                attempts = int(
                    getattr(instance, INST_LITELLM_ATTEMPTS_ATTR, 0) or 0
                )
                if attempts:
                    span.set_attribute("algo.llm.retry_count", attempts)
            except Exception:  # noqa: BLE001
                pass

        self._close_step(instance, span, token)
        return result

    @staticmethod
    def _close_step(
        instance: Any, span: Optional[Span], token: Optional[Any]
    ) -> None:
        try:
            if span is not None and span.is_recording():
                span.end()
        except Exception:  # noqa: BLE001
            pass
        try:
            if token is not None:
                otel_context.detach(token)
        except Exception:  # noqa: BLE001
            pass
        _clear_step_state(instance)


def _clear_step_state(instance: Any) -> None:
    try:
        setattr(instance, INST_STEP_SPAN_ATTR, None)
        setattr(instance, INST_STEP_TOKEN_ATTR, None)
    except Exception:  # noqa: BLE001
        pass


def _extract_command_name(result: Any) -> str:
    """Try to recover the executed command name from ``handle_function_call``
    output."""
    if not isinstance(result, dict):
        return ""
    # CommandResult-style payloads may carry the command name inside
    # ``data`` or via ``status_field``-keyed entries; we keep this loose
    # because the AlgoTune handlers vary per command.
    for key in ("command", "name", "cmd"):
        val = result.get(key)
        if isinstance(val, str) and val:
            return val
    data = result.get("data")
    if isinstance(data, dict):
        for key in ("command", "name", "cmd"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
    return ""


# ---------------------------------------------------------------------------
# TOOL: CommandHandlers.handle_command()
# ---------------------------------------------------------------------------


class HandleCommandWrapper:
    """TOOL span around ``CommandHandlers.handle_command``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        command_obj = args[0] if args else kwargs.get("command_str")
        cmd_name, cmd_args, is_error_response = _parse_command(command_obj)

        span_name = f"execute_tool {cmd_name or 'unknown'}"
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "TOOL")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )
            span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, cmd_name or "unknown")
            span.set_attribute(GenAI.GEN_AI_TOOL_TYPE, "function")
            span.set_attribute(
                GenAI.GEN_AI_TOOL_DESCRIPTION,
                "AlgoTune internal command",
            )
            span.set_attribute(GenAI.GEN_AI_TOOL_CALL_ID, uuid.uuid4().hex)

            if is_error_response:
                span.set_attribute("algotune.command.error_response", True)

            if (
                OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
                and cmd_args is not None
            ):
                try:
                    span.set_attribute(
                        GenAI.GEN_AI_TOOL_CALL_ARGUMENTS,
                        truncate(json.dumps(cmd_args, default=str)),
                    )
                except Exception:  # noqa: BLE001
                    pass

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            if isinstance(result, dict):
                success = bool(result.get("success", False))
                span.set_attribute("algo.command.success", success)
                if not success and not is_error_response:
                    span.set_status(Status(StatusCode.ERROR, "command failed"))

                if OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT:
                    msg = result.get("message")
                    if msg:
                        span.set_attribute(
                            GenAI.GEN_AI_TOOL_CALL_RESULT,
                            truncate(msg),
                        )

                # Best-effort snapshot detection (only present for ``edit``).
                data = result.get("data")
                if isinstance(data, dict):
                    snap = data.get("snapshot_saved")
                    if snap is not None:
                        try:
                            span.set_attribute(
                                "algo.snapshot.saved", bool(snap)
                            )
                        except Exception:  # noqa: BLE001
                            pass

            return result


def _parse_command(command_obj: Any) -> tuple[str, Optional[dict], bool]:
    """Extract ``(command_name, args_dict, is_error_response)`` from the
    command object passed to ``handle_command``.

    AlgoTune passes either a ``ParsedCommand`` dataclass or a structured
    error dict (see handlers.py line 226).
    """
    if isinstance(command_obj, dict):
        # Validation/parsing error dict path.
        cmd = command_obj.get("command") or "error_response"
        return str(cmd), None, True
    name = getattr(command_obj, "command", None)
    args = getattr(command_obj, "args", None)
    if isinstance(args, dict):
        return str(name or "unknown"), args, False
    return str(name or "unknown"), None, False


# ---------------------------------------------------------------------------
# TASK(dataset_eval): CommandHandlers._runner_eval_dataset()
# ---------------------------------------------------------------------------


class RunnerEvalDatasetWrapper:
    """TASK span around ``CommandHandlers._runner_eval_dataset``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        data_subset = (
            args[0] if len(args) >= 1 else kwargs.get("data_subset", "")
        )
        command_source = (
            args[1] if len(args) >= 2 else kwargs.get("command_source", "")
        )

        with self._tracer.start_as_current_span(
            "run_task benchmark.dataset_eval", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "TASK")
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
            span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
            span.set_attribute("gen_ai.task.name", "benchmark.dataset_eval")
            if data_subset:
                span.set_attribute("algo.eval.subset", str(data_subset))
            if command_source:
                span.set_attribute(
                    "algo.eval.command_source", str(command_source)
                )
            _set_task_input(
                span,
                {
                    "task": "benchmark.dataset_eval",
                    "data_subset": str(data_subset) if data_subset else "",
                    "command_source": str(command_source)
                    if command_source
                    else "",
                },
            )

            interface = getattr(instance, "interface", None)
            try:
                max_samples = getattr(interface, "max_samples", None)
                span.set_attribute(
                    "algo.eval.test_mode", max_samples is not None
                )
            except Exception:  # noqa: BLE001
                pass

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                self._record_eval_attributes(span, result)
                try:
                    result_data = (
                        result.data if hasattr(result, "data") else result
                    )
                    _set_task_output(
                        span,
                        {
                            "success": getattr(result, "success", None),
                            "status": getattr(result, "status", None),
                            "message": getattr(result, "message", None),
                            "data": result_data,
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
                return result
            finally:
                pass

    @staticmethod
    def _record_eval_attributes(span: Span, result: Any) -> None:
        # ``result`` is typically a ``CommandResult`` dataclass with .data
        # carrying aggregate evaluation values, but downstream code also accepts
        # raw dicts. We use getattr/dict-access defensively.
        try:
            data = result.data if hasattr(result, "data") else result
        except Exception:  # noqa: BLE001
            data = None

        if not isinstance(data, dict):
            return

        # The aggregate payload may live at the top level or inside
        # ``data``/``raw``/``metrics``.
        candidates = [data]
        for key in ("aggregate_metrics", "metrics", "raw"):
            sub = data.get(key) if isinstance(data, dict) else None
            if isinstance(sub, dict):
                candidates.append(sub)

        for src in candidates:
            for src_key, dst_attr, caster in (
                ("num_evaluated", "algo.eval.total_problems", int),
                ("mean_speedup", "algo.eval.mean_speedup", float),
                ("num_valid", "algo.eval.num_valid", int),
                ("num_invalid", "algo.eval.num_invalid", int),
                ("num_timeout", "algo.eval.num_timeout", int),
            ):
                if src_key in src and src[src_key] is not None:
                    try:
                        span.set_attribute(dst_attr, caster(src[src_key]))
                    except (TypeError, ValueError):
                        pass


# ---------------------------------------------------------------------------
# TASK(problem_eval): EvaluationOrchestrator.evaluate_single()
# ---------------------------------------------------------------------------


class EvaluateSingleWrapper:
    """TASK span around ``EvaluationOrchestrator.evaluate_single``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        problem_id = kwargs.get("problem_id", "problem")
        problem_index = kwargs.get("problem_index", 0)
        baseline_time_ms = kwargs.get("baseline_time_ms")

        with self._tracer.start_as_current_span(
            "run_task benchmark.problem_eval", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "TASK")
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
            span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
            span.set_attribute("gen_ai.task.name", "benchmark.problem_eval")
            span.set_attribute("algo.problem.id", str(problem_id))
            try:
                span.set_attribute("algo.problem.index", int(problem_index))
            except (TypeError, ValueError):
                pass
            if baseline_time_ms is not None:
                try:
                    span.set_attribute(
                        "algo.problem.baseline_time_ms",
                        float(baseline_time_ms),
                    )
                except (TypeError, ValueError):
                    pass
            _set_task_input(
                span,
                {
                    "task": "benchmark.problem_eval",
                    "problem_id": str(problem_id),
                    "problem_index": problem_index,
                    "baseline_time_ms": baseline_time_ms,
                    "kwargs": kwargs,
                },
            )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                self._record_problem_attributes(span, result)
                try:
                    _set_task_output(
                        span,
                        {
                            "speedup": _safe_get(result, "speedup"),
                            "solver_time_ms": _safe_get(
                                result, "solver_time_ms"
                            ),
                            "is_valid": _safe_get(result, "is_valid"),
                            "error_type": _safe_get(
                                _safe_get(result, "execution"),
                                "error_type",
                            ),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
                return result
            finally:
                pass

    @staticmethod
    def _record_problem_attributes(span: Span, result: Any) -> None:
        # ``ProblemResult`` is a dataclass; defensive getattr handles
        # alternate shapes (dict / namedtuple).
        speedup = _safe_get(result, "speedup")
        if speedup is not None:
            try:
                span.set_attribute("algo.problem.speedup", float(speedup))
            except (TypeError, ValueError):
                pass

        solver_time = _safe_get(result, "solver_time_ms")
        if solver_time is not None:
            try:
                span.set_attribute(
                    "algo.problem.solver_time_ms", float(solver_time)
                )
            except (TypeError, ValueError):
                pass

        is_valid = _safe_get(result, "is_valid")
        if is_valid is not None:
            try:
                span.set_attribute("algo.problem.is_valid", bool(is_valid))
            except (TypeError, ValueError):
                pass

        execution = _safe_get(result, "execution")
        if execution is not None:
            timed_out = _safe_get(execution, "timeout_occurred")
            if timed_out is not None:
                try:
                    span.set_attribute(
                        "algo.problem.timeout_occurred", bool(timed_out)
                    )
                except (TypeError, ValueError):
                    pass
            err_type = _safe_get(execution, "error_type")
            if err_type is not None:
                value = getattr(err_type, "value", err_type)
                span.set_attribute("algo.problem.error_type", str(value))


def _safe_get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# ---------------------------------------------------------------------------
# TASK(baseline): BaselineManager.get_baseline_times()
# ---------------------------------------------------------------------------


class GetBaselineTimesWrapper:
    """TASK span around ``BaselineManager.get_baseline_times``.

    Special-cased to keep the span healthy across ``SystemExit(1)``
    raised from inside the retry loop on fatal failure.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        subset = args[0] if args else kwargs.get("subset", "")
        cache_hit = False
        try:
            cache = getattr(instance, "_cache", None)
            if isinstance(cache, dict) and cache.get(subset) is not None:
                cache_hit = True
        except Exception:  # noqa: BLE001
            pass

        with self._tracer.start_as_current_span(
            "run_task benchmark.baseline_generation",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "TASK")
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
            span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
            span.set_attribute(
                "gen_ai.task.name", "benchmark.baseline_generation"
            )
            if subset:
                span.set_attribute("algo.baseline.subset", str(subset))
            span.set_attribute("algo.baseline.cache_hit", cache_hit)
            _set_task_input(
                span,
                {
                    "task": "benchmark.baseline_generation",
                    "subset": str(subset) if subset else "",
                    "cache_hit": cache_hit,
                },
            )

            try:
                result = wrapped(*args, **kwargs)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                span.add_event(
                    "baseline.fatal_failure", {"exit_code": int(code)}
                )
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        "Baseline generation fatal failure",
                    )
                )
                raise
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                if isinstance(result, dict):
                    span.set_attribute(
                        "algo.baseline.actual_count", len(result)
                    )
                _set_task_output(
                    span,
                    {
                        "count": len(result)
                        if isinstance(result, dict)
                        else None,
                        "result": result,
                    },
                )
                return result
            finally:
                pass


# ---------------------------------------------------------------------------
# LLM retry counters (no spans). Cooperates with the LiteLLM instrumentor
# which is responsible for actual LLM spans.
# ---------------------------------------------------------------------------


class LiteLLMQueryWrapper:
    """Wrap ``LiteLLMModel.query`` to publish ``algo.llm.retry_count`` onto
    the active STEP span. **Never creates a span.**"""

    __slots__ = ()

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Use the LLMInterface instance (carrying the STEP span) which is
        # accessible from the model only indirectly. We instead read the
        # current span and treat it as the STEP if its kind matches.
        step_span = trace_api.get_current_span()
        # Reset attempt count on this LiteLLMModel instance for this call.
        try:
            setattr(instance, "_otel_algo_litellm_call_attempts", 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            return wrapped(*args, **kwargs)
        finally:
            try:
                attempts = int(
                    getattr(instance, "_otel_algo_litellm_call_attempts", 0)
                    or 0
                )
                if (
                    attempts
                    and step_span is not None
                    and step_span.is_recording()
                ):
                    # Surface raw per-call attempts as a separate attribute
                    # (the wrapping STEP also aggregates across multiple
                    # query() invocations via INST_LITELLM_ATTEMPTS_ATTR).
                    step_span.set_attribute(
                        "algo.llm.last_call_attempts", attempts
                    )
            except Exception:  # noqa: BLE001
                pass


class LiteLLMExecuteQueryWrapper:
    """Wrap ``LiteLLMModel._execute_query`` to count attempts.

    Each call corresponds to one ``litellm.completion()`` invocation. We
    increment a counter on both the LiteLLMModel instance (for the per-call
    metric above) and on the LLMInterface instance hosting the STEP
    span (for the total per-step retry count)."""

    __slots__ = ()

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Per-call attempts (on LiteLLMModel instance).
        try:
            cur = int(
                getattr(instance, "_otel_algo_litellm_call_attempts", 0) or 0
            )
            setattr(instance, "_otel_algo_litellm_call_attempts", cur + 1)
        except Exception:  # noqa: BLE001
            pass

        # Per-step attempts (on LLMInterface instance, located via the
        # current STEP span's holder). Walk up the wrapt context: the
        # LLMInterface owns the LiteLLMModel via ``self.model``, so we
        # use a global registry-free approach by looking at the active
        # span's instance binding through the OTel context stack.
        active = trace_api.get_current_span()
        if active is not None and active.is_recording():
            # We can't directly resolve the LLMInterface from the active span,
            # so we increment a counter we keep on the active span itself.
            try:
                # Read existing total via OTel attribute is not supported;
                # we keep our own counter on the span object via a private
                # attribute. ``Span`` doesn't expose attribute reads, so
                # we maintain a side-band store via setattr on ``active``
                # only when it's a typed mutable Span (SDK ``ReadableSpan``
                # is hashable and supports attribute assignment in CPython).
                cur_total = getattr(active, "_otel_algo_step_attempts", 0) + 1
                try:
                    setattr(active, "_otel_algo_step_attempts", cur_total)
                except Exception:  # noqa: BLE001
                    cur_total = 0
                if cur_total:
                    active.set_attribute("algo.llm.retry_count", cur_total)
            except Exception:  # noqa: BLE001
                pass

        return wrapped(*args, **kwargs)


# ---------------------------------------------------------------------------
# LLM (optional bypass): TogetherModel.query()
# ---------------------------------------------------------------------------


class TogetherModelQueryWrapper:
    """LLM span around ``TogetherModel.query``.

    Together's HTTP client is invoked directly via ``requests.post`` and
    therefore not covered by the LiteLLM instrumentor. This wrapper is
    **opt-in** via ``ALGOTUNE_OTEL_INSTRUMENT_TOGETHER=true``.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        model_name = str(getattr(instance, "model_name", "") or "unknown")
        span_name = f"chat {model_name}"
        defaults = getattr(instance, "default_params", None) or {}

        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "LLM")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.CHAT.value,
            )
            span.set_attribute(GEN_AI_FRAMEWORK, ALGOTUNE_FRAMEWORK_VALUE)
            span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(GenAI.GEN_AI_PROVIDER_NAME, "together_ai")

            try:
                if isinstance(defaults, dict):
                    if (
                        "temperature" in defaults
                        and defaults["temperature"] is not None
                    ):
                        span.set_attribute(
                            GenAI.GEN_AI_REQUEST_TEMPERATURE,
                            float(defaults["temperature"]),
                        )
                    if "top_p" in defaults and defaults["top_p"] is not None:
                        span.set_attribute(
                            GenAI.GEN_AI_REQUEST_TOP_P,
                            float(defaults["top_p"]),
                        )
                    if (
                        "max_tokens" in defaults
                        and defaults["max_tokens"] is not None
                    ):
                        span.set_attribute(
                            GenAI.GEN_AI_REQUEST_MAX_TOKENS,
                            int(defaults["max_tokens"]),
                        )
            except Exception:  # noqa: BLE001
                pass

            input_tokens = 0
            output_tokens = 0
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                if isinstance(result, dict):
                    cost = result.get("cost")
                    if cost is not None:
                        try:
                            span.set_attribute(
                                "algo.llm.response_cost_usd", float(cost)
                            )
                        except (TypeError, ValueError):
                            pass
                    usage = result.get("usage")
                    if isinstance(usage, dict):
                        input_tokens, output_tokens = _extract_together_usage(
                            usage
                        )
                        if input_tokens:
                            span.set_attribute(
                                GenAI.GEN_AI_USAGE_INPUT_TOKENS, input_tokens
                            )
                        if output_tokens:
                            span.set_attribute(
                                GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens
                            )
                        total = (
                            usage.get("total_tokens")
                            if usage.get("total_tokens") is not None
                            else (input_tokens + output_tokens or None)
                        )
                        if total:
                            try:
                                span.set_attribute(
                                    GEN_AI_USAGE_TOTAL_TOKENS, int(total)
                                )
                            except (TypeError, ValueError):
                                pass
                    if OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT:
                        msg = result.get("message")
                        if msg:
                            span.set_attribute(
                                GenAI.GEN_AI_OUTPUT_MESSAGES, truncate(msg)
                            )
                return result
            finally:
                pass


def _extract_together_usage(usage: dict) -> tuple[int, int]:
    """Pick (input_tokens, output_tokens) from Together's usage payload.

    Together returns OpenAI-compatible ``prompt_tokens`` /
    ``completion_tokens`` but we tolerate ``input_tokens`` / ``output_tokens``
    as well in case the upstream schema drifts.
    """
    inp = usage.get("prompt_tokens")
    if inp is None:
        inp = usage.get("input_tokens")
    out = usage.get("completion_tokens")
    if out is None:
        out = usage.get("output_tokens")
    try:
        inp_i = int(inp) if inp is not None else 0
    except (TypeError, ValueError):
        inp_i = 0
    try:
        out_i = int(out) if out is not None else 0
    except (TypeError, ValueError):
        out_i = 0
    return inp_i, out_i
