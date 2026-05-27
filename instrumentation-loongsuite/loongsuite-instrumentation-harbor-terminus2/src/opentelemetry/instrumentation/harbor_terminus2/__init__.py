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

"""
OpenTelemetry Harbor Terminus2 Instrumentation.

Provides automatic instrumentation for Harbor's Terminus2 agent via external
monkey patching (no Harbor source changes required).

Span hierarchy:

  enter_ai_application_system        (ENTRY  / enter)
    -> invoke_agent harbor-terminus2 (AGENT  / invoke_agent)
         -> react step              (STEP   / react)
              -> (LLM span produced by litellm instrumentation)
              -> run_task parse_response (TASK / run_task)
              -> chain summarize    (CHAIN  / task)
              -> execute_tool terminal (TOOL / execute_tool)

LLM spans are intentionally not produced here. Harbor Terminus2 ultimately
calls its configured LLM backend; litellm calls are traced by
opentelemetry-instrumentation-litellm when that instrumentor is enabled.
"""

import contextvars
import json
import logging
from typing import Any, Collection

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_FRAMEWORK = "harbor"
_AGENT_NAME = "harbor-terminus2"
_TERMINAL_TOOL_NAME = "terminal"
_TERMINAL_TOOL_DESCRIPTION = "Send keystrokes to a tmux terminal session"

_GEN_AI_SPAN_KIND = "gen_ai.span.kind"
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_FRAMEWORK = "gen_ai.framework"
_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
_GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
_GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
_GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
_GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
_GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
_GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"

_SPAN_KIND_ENTRY = "ENTRY"
_SPAN_KIND_AGENT = "AGENT"
_SPAN_KIND_TOOL = "TOOL"
_SPAN_KIND_STEP = "STEP"
_SPAN_KIND_TASK = "TASK"
_SPAN_KIND_CHAIN = "CHAIN"
_OP_ENTER = "enter"
_OP_INVOKE_AGENT = "invoke_agent"
_OP_EXECUTE_TOOL = "execute_tool"
_OP_REACT = "react"
_OP_RUN_TASK = "run_task"
_OP_TASK = "task"
_TOOL_TYPE_EXTENSION = "extension"

_GEN_AI_REACT_ROUND = "gen_ai.react.round"
_GEN_AI_REACT_FINISH_REASON = "gen_ai.react.finish_reason"

_TERMINAL_TOOL_DEFINITION = json.dumps([{
    "type": "function",
    "name": _TERMINAL_TOOL_NAME,
    "description": _TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "keystrokes": {
                "type": "string",
                "description": "Exact keystrokes to send to the terminal",
            },
            "duration_sec": {
                "type": "number",
                "description": "Seconds to wait for the command to complete",
            },
        },
        "required": ["keystrokes"],
    },
}], ensure_ascii=False)

_current_step_span = contextvars.ContextVar(
    "harbor_terminus2_current_step_span", default=None
)
_current_step_token = contextvars.ContextVar(
    "harbor_terminus2_current_step_token", default=None
)
_react_round_counter = contextvars.ContextVar(
    "harbor_terminus2_react_round_counter", default=0
)

_HARBOR_TERMINUS2_MARKER = "_otel_harbor_terminus2_wrapped"


def _commands_to_arguments_json(commands) -> str:
    serialized = []
    for cmd in commands or []:
        serialized.append({
            "keystrokes": getattr(cmd, "keystrokes", ""),
            "duration_sec": getattr(cmd, "duration_sec", None),
        })
    try:
        return json.dumps(serialized, ensure_ascii=False)
    except Exception:
        return str(serialized)


def _text_messages_json(role: str, content: Any) -> str:
    message = {
        "role": role,
        "parts": [{"type": "text", "content": str(content)}],
    }
    try:
        return json.dumps([message], ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str([message])


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _end_current_step(finish_reason: str | None = None) -> None:
    span = _current_step_span.get()
    token = _current_step_token.get()
    if span is not None:
        if finish_reason:
            span.set_attribute(_GEN_AI_REACT_FINISH_REASON, finish_reason)
        span.end()
        _current_step_span.set(None)
    if token is not None:
        context_api.detach(token)
        _current_step_token.set(None)


def _infer_provider_name(model_name: str) -> str:
    if not model_name:
        return "unknown"
    lower = model_name.lower()
    if any(k in lower for k in ("gpt", "o1-", "o3-", "o4-")):
        return "openai"
    if "claude" in lower or "anthropic" in lower:
        return "anthropic"
    if "gemini" in lower:
        return "google"
    if "llama" in lower or "meta" in lower:
        return "meta"
    if "mistral" in lower:
        return "mistral"
    if "qwen" in lower:
        return "alibaba"
    if "deepseek" in lower:
        return "deepseek"
    if "/" in model_name:
        return model_name.split("/", 1)[0]
    return "unknown"


def _get_model_name(instance: Any) -> str:
    return (
        getattr(instance, "_model_name", None)
        or getattr(instance, "model_name", None)
        or "unknown"
    )


def _get_chat_messages(chat: Any) -> list:
    messages = getattr(chat, "_messages", None)
    if messages is None:
        messages = getattr(chat, "messages", None)
    return list(messages or [])


def _resolve_target(module: str, name: str):
    from importlib import import_module

    mod = import_module(module)
    parts = name.split(".")
    parent = mod
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    return parent, attr, getattr(parent, attr, None)


def _try_wrap(module: str, name: str, wrapper) -> None:
    try:
        parent, attr, current = _resolve_target(module, name)
    except Exception as exc:
        logger.warning(f"Could not resolve {module}.{name}: {exc}")
        return

    if current is None:
        logger.warning(f"{module}.{name} not found")
        return

    if getattr(current, _HARBOR_TERMINUS2_MARKER, False):
        logger.debug(
            f"{module}.{name} already wrapped by harbor terminus2 "
            "instrumentation, skipping"
        )
        return

    try:
        wrap_function_wrapper(module=module, name=name, wrapper=wrapper)
    except Exception as exc:
        logger.warning(f"Could not wrap {module}.{name}: {exc}")
        return

    new_value = getattr(parent, attr, None)
    if new_value is not None:
        try:
            setattr(new_value, _HARBOR_TERMINUS2_MARKER, True)
        except Exception as exc:
            logger.debug(f"Could not mark {module}.{name}: {exc}")


def _try_unwrap(module: str, name: str) -> None:
    try:
        parent, attr, current = _resolve_target(module, name)
    except Exception:
        return

    if current is None or not getattr(current, _HARBOR_TERMINUS2_MARKER, False):
        return

    try:
        delattr(current, _HARBOR_TERMINUS2_MARKER)
    except (AttributeError, TypeError):
        pass

    try:
        unwrap(parent, attr)
    except Exception as exc:
        logger.debug(f"Could not unwrap {module}.{name}: {exc}")


class HarborTerminus2Instrumentor(BaseInstrumentor):
    """Instrumentor for Harbor's Terminus2 agent."""

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(__name__, "", tracer_provider=tracer_provider)

        _try_wrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2.run",
            _RunWrapper(tracer),
        )
        _try_wrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2._run_agent_loop",
            _RunAgentLoopWrapper(tracer),
        )
        _try_wrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2._execute_commands",
            _ExecuteCommandsWrapper(tracer),
        )
        _try_wrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2._handle_llm_interaction",
            _HandleLLMInteractionWrapper(tracer),
        )
        _try_wrap(
            "harbor.agents.terminus_2.terminus_json_plain_parser",
            "TerminusJSONPlainParser.parse_response",
            _ParseResponseWrapper(tracer, "json"),
        )
        _try_wrap(
            "harbor.agents.terminus_2.terminus_xml_plain_parser",
            "TerminusXMLPlainParser.parse_response",
            _ParseResponseWrapper(tracer, "xml"),
        )
        _try_wrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2._summarize",
            _SummarizeWrapper(tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        _try_unwrap("harbor.agents.terminus_2.terminus_2", "Terminus2.run")
        _try_unwrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2._run_agent_loop",
        )
        _try_unwrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2._execute_commands",
        )
        _try_unwrap(
            "harbor.agents.terminus_2.terminus_2",
            "Terminus2._handle_llm_interaction",
        )
        _try_unwrap(
            "harbor.agents.terminus_2.terminus_json_plain_parser",
            "TerminusJSONPlainParser.parse_response",
        )
        _try_unwrap(
            "harbor.agents.terminus_2.terminus_xml_plain_parser",
            "TerminusXMLPlainParser.parse_response",
        )
        _try_unwrap("harbor.agents.terminus_2.terminus_2", "Terminus2._summarize")
        _end_current_step()


class _RunWrapper:
    """Wrap ``Terminus2.run`` to produce the ENTRY span."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        model_name = _get_model_name(instance)
        instruction = args[0] if args else kwargs.get("instruction", "")
        context = args[2] if len(args) > 2 else kwargs.get("context")

        with self._tracer.start_as_current_span(
            "enter_ai_application_system",
            kind=SpanKind.SERVER,
        ) as span:
            span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_ENTRY)
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_ENTER)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute(_GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(
                _GEN_AI_PROVIDER_NAME,
                _infer_provider_name(model_name),
            )
            span.set_attribute("gen_ai.agent.name", _AGENT_NAME)

            if instruction:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_messages_json("user", instruction),
                )

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            metadata = getattr(context, "metadata", None) if context else None
            output_summary = {
                "metadata": metadata or {},
                "completed": True,
            }
            span.set_attribute(
                _GEN_AI_OUTPUT_MESSAGES,
                _text_messages_json("assistant", _json_dumps(output_summary)),
            )
            if metadata and isinstance(metadata, dict):
                internal_error_type = metadata.get("internal_error_type")
                if internal_error_type:
                    span.set_attribute(
                        "harbor_terminus2.internal_error_type",
                        str(internal_error_type),
                    )
                n_episodes = metadata.get("n_episodes")
                if n_episodes is not None:
                    span.set_attribute("harbor_terminus2.episodes", n_episodes)
                summarization_count = metadata.get("summarization_count")
                if summarization_count is not None:
                    span.set_attribute(
                        "harbor_terminus2.summarization_count",
                        summarization_count,
                    )

            span.set_status(Status(StatusCode.OK))
            return result


class _RunAgentLoopWrapper:
    """Wrap ``Terminus2._run_agent_loop`` to produce the AGENT span."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        _react_round_counter.set(0)
        _end_current_step()

        model_name = _get_model_name(instance)
        parser_name = getattr(instance, "_parser_name", "unknown")
        chat = args[1] if len(args) > 1 else kwargs.get("chat")
        original_instruction = (
            args[3] if len(args) > 3 else kwargs.get("original_instruction", "")
        )

        with self._tracer.start_as_current_span(
            f"invoke_agent {_AGENT_NAME}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_AGENT)
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_INVOKE_AGENT)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute("gen_ai.agent.name", _AGENT_NAME)
            span.set_attribute(
                "gen_ai.agent.description",
                "Harbor Terminus2 agent (ReAct loop over a tmux session)",
            )
            span.set_attribute(_GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(
                _GEN_AI_PROVIDER_NAME,
                _infer_provider_name(model_name),
            )
            span.set_attribute("harbor_terminus2.parser", parser_name)

            system_instructions = getattr(instance, "_prompt_template", "")
            if system_instructions:
                span.set_attribute(
                    _GEN_AI_SYSTEM_INSTRUCTIONS, system_instructions
                )

            span.set_attribute(_GEN_AI_TOOL_DEFINITIONS, _TERMINAL_TOOL_DEFINITION)

            if original_instruction:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_messages_json("user", original_instruction),
                )

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                _end_current_step(finish_reason="loop_end")
                raise

            _end_current_step(finish_reason="loop_end")

            rounds = _react_round_counter.get()
            span.set_attribute("harbor_terminus2.react.rounds", rounds)

            pending_completion = bool(getattr(instance, "_pending_completion", False))
            final_assistant_text = ""
            for msg in reversed(_get_chat_messages(chat)):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content")
                    if content is not None:
                        final_assistant_text = str(content)
                    break

            output_summary = {
                "react_rounds": rounds,
                "pending_completion": pending_completion,
                "final_assistant_message": final_assistant_text,
            }
            span.set_attribute(
                _GEN_AI_OUTPUT_MESSAGES,
                _text_messages_json("assistant", _json_dumps(output_summary)),
            )
            span.set_attribute(
                "harbor_terminus2.pending_completion", pending_completion
            )

            span.set_status(Status(StatusCode.OK))
            return result


class _ExecuteCommandsWrapper:
    """Wrap ``Terminus2._execute_commands`` to produce a TOOL span."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        commands = args[0] if args else kwargs.get("commands", [])

        with self._tracer.start_as_current_span(
            f"execute_tool {_TERMINAL_TOOL_NAME}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_TOOL)
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_EXECUTE_TOOL)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute(_GEN_AI_TOOL_NAME, _TERMINAL_TOOL_NAME)
            span.set_attribute(
                _GEN_AI_TOOL_DESCRIPTION, _TERMINAL_TOOL_DESCRIPTION
            )
            span.set_attribute(_GEN_AI_TOOL_TYPE, _TOOL_TYPE_EXTENSION)
            span.set_attribute("harbor_terminus2.commands.count", len(commands or []))
            span.set_attribute(
                _GEN_AI_TOOL_CALL_ARGUMENTS,
                _commands_to_arguments_json(commands),
            )

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            timeout_occurred, terminal_output = result
            span.set_attribute(
                "harbor_terminus2.terminal.timeout", timeout_occurred
            )
            if terminal_output is not None:
                span.set_attribute(_GEN_AI_TOOL_CALL_RESULT, str(terminal_output))

            span.set_status(Status(StatusCode.OK))
            return result


class _HandleLLMInteractionWrapper:
    """Wrap ``Terminus2._handle_llm_interaction`` to produce a STEP span."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        _end_current_step(finish_reason="next_round")

        round_num = _react_round_counter.get() + 1
        _react_round_counter.set(round_num)

        step_span = self._tracer.start_span("react step", kind=SpanKind.INTERNAL)
        step_span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_STEP)
        step_span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_REACT)
        step_span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
        step_span.set_attribute(_GEN_AI_REACT_ROUND, round_num)

        ctx = trace_api.set_span_in_context(step_span)
        token = context_api.attach(ctx)
        _current_step_span.set(step_span)
        _current_step_token.set(token)

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            step_span.set_attribute(_GEN_AI_REACT_FINISH_REASON, "error")
            step_span.record_exception(exc)
            step_span.set_status(Status(StatusCode.ERROR))
            raise

        commands, is_task_complete, feedback, *_ = result

        step_span.set_attribute(
            "harbor_terminus2.commands.count", len(commands or [])
        )
        if is_task_complete:
            step_span.set_attribute(_GEN_AI_REACT_FINISH_REASON, "complete")
        elif feedback and "ERROR:" in feedback:
            step_span.set_attribute(_GEN_AI_REACT_FINISH_REASON, "parse_error")

        return result


class _ParseResponseWrapper:
    """Wrap parser ``parse_response`` to produce a TASK span."""

    def __init__(self, tracer, parser_type):
        self._tracer = tracer
        self._parser_type = parser_type

    def __call__(self, wrapped, instance, args, kwargs):
        response_text = args[0] if args else kwargs.get("response", "")

        with self._tracer.start_as_current_span(
            "run_task parse_response",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_TASK)
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_RUN_TASK)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute("harbor_terminus2.parser", self._parser_type)

            if response_text is not None:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_messages_json("assistant", response_text),
                )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            commands = getattr(result, "commands", []) or []
            span.set_attribute(
                "harbor_terminus2.task_complete",
                bool(getattr(result, "is_task_complete", False)),
            )
            span.set_attribute("harbor_terminus2.commands.count", len(commands))

            output_summary = {
                "is_task_complete": getattr(result, "is_task_complete", False),
                "commands": [
                    {
                        "keystrokes": getattr(cmd, "keystrokes", ""),
                        "duration": getattr(cmd, "duration", None),
                    }
                    for cmd in commands
                ],
                "error": getattr(result, "error", "") or "",
                "warning": getattr(result, "warning", "") or "",
                "analysis": getattr(result, "analysis", "") or "",
                "plan": getattr(result, "plan", "") or "",
            }
            span.set_attribute(
                _GEN_AI_OUTPUT_MESSAGES,
                _text_messages_json("assistant", _json_dumps(output_summary)),
            )

            error = getattr(result, "error", None)
            warning = getattr(result, "warning", None)
            if error:
                span.set_attribute("harbor_terminus2.parse.error", str(error))
            if warning:
                span.set_attribute("harbor_terminus2.parse.warning", str(warning))

            span.set_status(Status(StatusCode.OK))
            return result


class _SummarizeWrapper:
    """Wrap ``Terminus2._summarize`` to produce a CHAIN span."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        with self._tracer.start_as_current_span(
            "chain summarize",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_CHAIN)
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_TASK)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

            span.set_status(Status(StatusCode.OK))
            return result
