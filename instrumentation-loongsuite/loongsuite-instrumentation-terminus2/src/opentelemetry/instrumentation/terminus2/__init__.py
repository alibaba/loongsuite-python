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
OpenTelemetry Terminus2 Instrumentation

Provides automatic instrumentation for the terminus-2 agent from terminal-bench
via external monkey patching (no upstream changes required).

Span hierarchy & semantic mapping (strictly follows ARMS gen-ai semantic
conventions, see ``arms_docs/trace/gen-ai.md``):

  enter_ai_application_system        (ENTRY  / enter)
    └── invoke_agent terminus-2      (AGENT  / invoke_agent)
          └── react step             (STEP   / react)              ── episode N
                ├── (LLM span produced by ``opentelemetry-instrumentation-litellm``)
                ├── run_task parse_response (TASK   / run_task)
                ├── chain summarize  (CHAIN  / task)               ── on overflow
                └── execute_tool terminal  (TOOL   / execute_tool)

LLM spans are intentionally **not** produced by this package. The underlying
``LiteLLM.call`` invokes ``litellm.completion`` which is already traced by
``opentelemetry-instrumentation-litellm``; emitting another span here would
duplicate that record.

Token totals are NOT aggregated on the AGENT or ENTRY span. Upstream
``Chat`` uses a local-tokenizer estimate that misses reasoning tokens,
tool-call arg serialization, prompt-template wrapping, and anthropic-
caching injections — and ``Terminus2._summarize`` issues bare
``chat._model.call`` invocations that bypass ``Chat.chat`` entirely. Any
agent-level total computed from those counters would be systematically
wrong, so we omit it. Authoritative per-call token usage lives on each
LLM child span produced by the litellm instrumentor.

Patch targets (all monkey-patched via ``wrapt.wrap_function_wrapper``):

  P0  Terminus2.perform_task          → ENTRY span (application entry)
  P0  Terminus2._run_agent_loop       → AGENT span + episode lifecycle
  P0  Terminus2._execute_commands     → TOOL span
  P1  Terminus2._handle_llm_interaction → STEP span (per ReAct iteration)
  P1  TerminusJSONPlainParser.parse_response /
      TerminusXMLPlainParser.parse_response → TASK span
  P2  Terminus2._summarize            → CHAIN span (handoff)
"""

import contextvars
import json
import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.terminus2.package import _instruments
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

# ── Framework / agent identifiers ────────────────────────────────────────────
_FRAMEWORK = "terminal-bench"
_AGENT_NAME = "terminus-2"
_TERMINAL_TOOL_NAME = "terminal"
_TERMINAL_TOOL_DESCRIPTION = "Send keystrokes to a tmux terminal session"

# ── GenAI semantic-convention attribute keys ────────────────────────────────
# Strings inlined to avoid taking a hard dependency on private aliyun packages
# that aren't published to PyPI (aliyun.semconv.trace_v2,
# aliyun.sdk.extension.arms.*). Values track the ARMS gen-ai semconv.
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

# ── Span kind / operation / tool-type values ────────────────────────────────
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

_TERMINAL_TOOL_DEFINITION = json.dumps(
    [
        {
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
        }
    ],
    ensure_ascii=False,
)

# ── ReAct extension attributes (阿里云扩展规范) ──────────────────────────────
_GEN_AI_REACT_ROUND = "gen_ai.react.round"
_GEN_AI_REACT_FINISH_REASON = "gen_ai.react.finish_reason"

# ── Content capture ─────────────────────────────────────────────────────────
# Inputs / outputs (instruction text, terminal keystrokes, terminal output,
# AgentResult summary) are captured **unconditionally and untruncated** —
# they are the primary observability signal for terminus-2. If full content
# is undesirable in a given deployment, configure exporter-side filtering or
# attribute-length limits in the SDK instead.


def _commands_to_arguments_json(commands) -> str:
    """Serialize a list of ``Command`` objects into a JSON string for
    ``gen_ai.tool.call.arguments``."""
    serialized = []
    for cmd in commands:
        serialized.append(
            {
                "keystrokes": getattr(cmd, "keystrokes", ""),
                "duration_sec": getattr(cmd, "duration_sec", None),
            }
        )
    try:
        return json.dumps(serialized, ensure_ascii=False)
    except Exception:
        return str(serialized)


def _text_messages_json(role: str, content: Any) -> str:
    """Serialize a single text message using the GenAI message schema."""
    message = {
        "role": role,
        "parts": [{"type": "text", "content": str(content)}],
    }
    try:
        return json.dumps([message], ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str([message])


# ── ReAct step lifecycle tracked via contextvars ────────────────────────────
# A STEP span stays open across `_handle_llm_interaction` ⇒ `_execute_commands`
# so both become its children. It is closed when the next iteration starts or
# when `_run_agent_loop` returns.
_current_step_span = contextvars.ContextVar(
    "terminus2_current_step_span", default=None
)
_current_step_token = contextvars.ContextVar(
    "terminus2_current_step_token", default=None
)
_react_round_counter = contextvars.ContextVar(
    "terminus2_react_round_counter", default=0
)


def _end_current_step(finish_reason: str | None = None) -> None:
    """End the active ReAct STEP span (if any) and detach its context."""
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
    """Infer ``gen_ai.provider.name`` from a model identifier string."""
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


# Sentinel attribute attached to every target we successfully wrap. Stored
# on the target callable itself (not in module-level state) so that
# duplicate wraps are detected even if this package is loaded as multiple
# module instances (e.g. wheel install + ``pip install -e`` source, or
# under different sys.path roots), or if ``_instrument()`` is invoked
# twice via auto-loader + manual call.
_TERMINUS2_MARKER = "_otel_terminus2_wrapped"


def _resolve_target(module: str, name: str):
    """Resolve ``module.name`` (where ``name`` may be ``Class.method``).

    Returns ``(parent, attr_name, current_value)``. Raises on missing
    module / attribute.
    """
    from importlib import import_module

    mod = import_module(module)
    parts = name.split(".")
    parent = mod
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    return parent, attr, getattr(parent, attr, None)


def _try_wrap(module: str, name: str, wrapper) -> None:
    """Wrap ``module.name`` with ``wrapper`` exactly once.

    Idempotency is enforced via a sentinel attribute attached to the
    target — robust against multiple module instances of this package and
    repeated ``_instrument()`` invocations.
    """
    try:
        parent, attr, current = _resolve_target(module, name)
    except Exception as e:
        logger.warning(f"Could not resolve {module}.{name}: {e}")
        return

    if current is None:
        logger.warning(f"{module}.{name} not found")
        return

    if getattr(current, _TERMINUS2_MARKER, False):
        logger.debug(
            f"{module}.{name} already wrapped by terminus2 instrumentation, "
            "skipping"
        )
        return

    try:
        wrap_function_wrapper(module=module, name=name, wrapper=wrapper)
    except Exception as e:
        logger.warning(f"Could not wrap {module}.{name}: {e}")
        return

    # Mark the freshly installed wrapper. wrapt's FunctionWrapper proxies
    # attribute writes to the underlying wrapped object, but reading the
    # attribute back through the proxy returns the same value, so a
    # subsequent ``getattr`` check on either layer detects the marker.
    new_value = getattr(parent, attr, None)
    if new_value is not None:
        try:
            setattr(new_value, _TERMINUS2_MARKER, True)
        except Exception as e:
            logger.debug(f"Could not mark {module}.{name}: {e}")


def _try_unwrap(module: str, name: str) -> None:
    """Reverse of :func:`_try_wrap`."""
    try:
        parent, attr, current = _resolve_target(module, name)
    except Exception:
        return

    if current is None or not getattr(current, _TERMINUS2_MARKER, False):
        return

    # Clear the marker on the underlying object first (FunctionWrapper
    # forwards delattr to the wrapped object, so the marker — which was
    # written through to the original — is removed cleanly).
    try:
        delattr(current, _TERMINUS2_MARKER)
    except (AttributeError, TypeError):
        pass

    try:
        unwrap(parent, attr)
    except Exception as e:
        logger.debug(f"Could not unwrap {module}.{name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Instrumentor
# ═══════════════════════════════════════════════════════════════════════════


class Terminus2Instrumentor(BaseInstrumentor):
    """Instrumentor for the terminus-2 agent from terminal-bench."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__, "", tracer_provider=tracer_provider
        )

        # P0 – ENTRY span (application entry point)
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2.perform_task",
            _PerformTaskWrapper(tracer),
        )

        # P0 – AGENT span (agent invocation) + ReAct loop lifecycle
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._run_agent_loop",
            _RunAgentLoopWrapper(tracer),
        )

        # NOTE: LLM spans for ``LiteLLM.call`` are NOT produced here —
        # ``opentelemetry-instrumentation-litellm`` already traces the
        # underlying ``litellm.completion`` invocation. Wrapping again would
        # produce duplicate LLM spans for every model call.

        # P0 – TOOL span for terminal command batch
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._execute_commands",
            _ExecuteCommandsWrapper(tracer),
        )

        # P1 – STEP span per ReAct iteration
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._handle_llm_interaction",
            _HandleLLMInteractionWrapper(tracer),
        )

        # P1 – TASK span for parser (json + xml)
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_json_plain_parser",
            "TerminusJSONPlainParser.parse_response",
            _ParseResponseWrapper(tracer, "json"),
        )
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_xml_plain_parser",
            "TerminusXMLPlainParser.parse_response",
            _ParseResponseWrapper(tracer, "xml"),
        )

        # P2 – CHAIN span for context-overflow handoff
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._summarize",
            _SummarizeWrapper(tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2.perform_task",
        )
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._run_agent_loop",
        )
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._execute_commands",
        )
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._handle_llm_interaction",
        )
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_json_plain_parser",
            "TerminusJSONPlainParser.parse_response",
        )
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_xml_plain_parser",
            "TerminusXMLPlainParser.parse_response",
        )
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._summarize",
        )
        _end_current_step()


# ═══════════════════════════════════════════════════════════════════════════
# P0 — ENTRY span: Terminus2.perform_task
# ═══════════════════════════════════════════════════════════════════════════


class _PerformTaskWrapper:
    """Wrap ``Terminus2.perform_task`` to produce the **ENTRY** span.

    Per spec: span name ``enter_ai_application_system``,
    ``gen_ai.span.kind=ENTRY``, ``gen_ai.operation.name=enter``.

    Records the user instruction as ``gen_ai.input.messages`` and a
    serialized summary of ``AgentResult`` (failure_mode, token totals,
    marker count) as ``gen_ai.output.messages`` once the task completes.
    """

    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        model_name = getattr(instance, "_model_name", "unknown")
        instruction = args[0] if args else kwargs.get("instruction", "")

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

            if instruction:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_messages_json("user", instruction),
                )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

            # AgentResult.total_*_tokens is a local-tokenizer estimate —
            # see docstring at top of module for why we don't surface it.
            failure_mode = getattr(result, "failure_mode", None)
            failure_mode_str = (
                str(getattr(failure_mode, "value", failure_mode))
                if failure_mode is not None
                else "none"
            )
            markers = getattr(result, "timestamped_markers", None) or []

            output_summary = {
                "failure_mode": failure_mode_str,
                "marker_count": len(markers),
            }
            try:
                output_value = json.dumps(output_summary, ensure_ascii=False)
            except Exception:
                output_value = str(output_summary)

            span.set_attribute(
                _GEN_AI_OUTPUT_MESSAGES,
                _text_messages_json("assistant", output_value),
            )
            span.set_attribute("terminus2.failure_mode", failure_mode_str)

            span.set_status(Status(StatusCode.OK))
            return result


# ═══════════════════════════════════════════════════════════════════════════
# P0 — AGENT span: Terminus2._run_agent_loop
# ═══════════════════════════════════════════════════════════════════════════


class _RunAgentLoopWrapper:
    """Wrap ``Terminus2._run_agent_loop`` to produce the **AGENT** span.

    Per spec: span name ``invoke_agent {agent.name}``,
    ``gen_ai.span.kind=AGENT``, ``gen_ai.operation.name=invoke_agent``.

    The AGENT span precisely brackets the ReAct loop body — STEP / TOOL /
    TASK / CHAIN children all hang off it. Token totals are aggregated
    from the ``Chat`` cumulative counters once the loop returns. Also
    cleans up any trailing STEP span on loop exit.
    """

    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        # Reset per-loop ReAct state
        _react_round_counter.set(0)
        _end_current_step()

        model_name = getattr(instance, "_model_name", "unknown")
        parser_name = getattr(instance, "_parser_name", "unknown")

        # _run_agent_loop signature:
        #   (initial_prompt, session, chat, logging_dir=None,
        #    original_instruction="")
        original_instruction = (
            args[4]
            if len(args) > 4
            else kwargs.get("original_instruction", "")
        )
        chat = args[2] if len(args) > 2 else kwargs.get("chat")

        with self._tracer.start_as_current_span(
            f"invoke_agent {_AGENT_NAME}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(
                _GEN_AI_SPAN_KIND,
                _SPAN_KIND_AGENT,
            )
            span.set_attribute(
                _GEN_AI_OPERATION_NAME,
                _OP_INVOKE_AGENT,
            )
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute("gen_ai.agent.name", _AGENT_NAME)
            span.set_attribute(
                "gen_ai.agent.description",
                "Terminus-2 terminal-bench agent (ReAct loop over a tmux session)",
            )
            span.set_attribute(_GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(
                _GEN_AI_PROVIDER_NAME,
                _infer_provider_name(model_name),
            )
            span.set_attribute("terminus2.parser", parser_name)

            system_instructions = getattr(instance, "_prompt_template", "")
            if system_instructions:
                span.set_attribute(
                    _GEN_AI_SYSTEM_INSTRUCTIONS, system_instructions
                )

            span.set_attribute(
                _GEN_AI_TOOL_DEFINITIONS, _TERMINAL_TOOL_DEFINITION
            )

            if original_instruction:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_messages_json("user", original_instruction),
                )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                _end_current_step(finish_reason="loop_end")
                raise

            _end_current_step(finish_reason="loop_end")

            # Per-call token usage is recorded on each litellm child span by
            # ``opentelemetry-instrumentation-litellm`` (provider-reported
            # ``response.usage``). We deliberately do NOT aggregate a total
            # on the AGENT span: ``Chat.total_*_tokens`` uses a local
            # tokenizer estimate (misses reasoning tokens, tool args,
            # prompt-template wrapping, anthropic-caching) and the bare
            # ``chat._model.call`` in ``_summarize`` bypasses ``Chat.chat``
            # entirely — any aggregate computed from those counters would
            # be systematically wrong.

            rounds = _react_round_counter.get()
            span.set_attribute("terminus2.react.rounds", rounds)

            # AGENT output: ``_run_agent_loop`` returns ``None``, so synthesize
            # an output message from the final state of the chat history (the
            # last ``assistant`` entry — the agent's terminal action/response)
            # plus loop-exit context. Without this the AGENT span has only
            # input, which is what the user sees as "no output".
            pending_completion = bool(
                getattr(instance, "_pending_completion", False)
            )
            final_assistant_text = ""
            messages = list(getattr(chat, "_messages", []) or [])
            for msg in reversed(messages):
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
            try:
                output_value = json.dumps(output_summary, ensure_ascii=False)
            except Exception:
                output_value = str(output_summary)
            span.set_attribute(
                _GEN_AI_OUTPUT_MESSAGES,
                _text_messages_json("assistant", output_value),
            )
            span.set_attribute(
                "terminus2.pending_completion", pending_completion
            )

            span.set_status(Status(StatusCode.OK))
            return result


# ═══════════════════════════════════════════════════════════════════════════
# P0 — TOOL span: Terminus2._execute_commands
# ═══════════════════════════════════════════════════════════════════════════


class _ExecuteCommandsWrapper:
    """Wrap ``Terminus2._execute_commands`` to produce a **TOOL** span.

    Per spec: span name ``execute_tool {tool_name}``,
    ``gen_ai.span.kind=TOOL``, ``gen_ai.operation.name=execute_tool``.
    """

    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        commands = args[0] if args else kwargs.get("commands", [])

        with self._tracer.start_as_current_span(
            f"execute_tool {_TERMINAL_TOOL_NAME}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(
                _GEN_AI_SPAN_KIND,
                _SPAN_KIND_TOOL,
            )
            span.set_attribute(
                _GEN_AI_OPERATION_NAME,
                _OP_EXECUTE_TOOL,
            )
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute(_GEN_AI_TOOL_NAME, _TERMINAL_TOOL_NAME)
            span.set_attribute(
                _GEN_AI_TOOL_DESCRIPTION, _TERMINAL_TOOL_DESCRIPTION
            )
            span.set_attribute(
                _GEN_AI_TOOL_TYPE,
                _TOOL_TYPE_EXTENSION,
            )
            span.set_attribute("terminus2.commands.count", len(commands))

            arguments_json = _commands_to_arguments_json(commands)
            # Spec attribute (gen-ai.md §Tool)
            span.set_attribute(_GEN_AI_TOOL_CALL_ARGUMENTS, arguments_json)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

            timeout_occurred, terminal_output = result
            span.set_attribute("terminus2.terminal.timeout", timeout_occurred)

            if terminal_output is not None:
                output_text = str(terminal_output)
                # Spec attribute (gen-ai.md §Tool)
                span.set_attribute(_GEN_AI_TOOL_CALL_RESULT, output_text)

            span.set_status(Status(StatusCode.OK))
            return result


# ═══════════════════════════════════════════════════════════════════════════
# P1 — STEP span: Terminus2._handle_llm_interaction
# ═══════════════════════════════════════════════════════════════════════════


class _HandleLLMInteractionWrapper:
    """Wrap ``Terminus2._handle_llm_interaction`` to produce a **STEP** span.

    The STEP span represents one ReAct iteration. It opens here, stays open
    after this method returns (so the subsequent ``_execute_commands`` call
    in ``_run_agent_loop`` becomes its child), and is closed on the next
    iteration entry or by ``_RunAgentLoopWrapper`` cleanup.
    """

    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        # Close previous STEP first (if any)
        _end_current_step(finish_reason="next_round")

        round_num = _react_round_counter.get() + 1
        _react_round_counter.set(round_num)

        step_span = self._tracer.start_span(
            "react step",
            kind=SpanKind.INTERNAL,
        )
        step_span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_STEP)
        step_span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_REACT)
        step_span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
        step_span.set_attribute(_GEN_AI_REACT_ROUND, round_num)

        ctx = trace_api.set_span_in_context(step_span)
        token = context_api.attach(ctx)
        _current_step_span.set(step_span)
        _current_step_token.set(token)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as e:
            step_span.set_attribute(_GEN_AI_REACT_FINISH_REASON, "error")
            step_span.record_exception(e)
            step_span.set_status(Status(StatusCode.ERROR))
            raise

        commands, is_task_complete, feedback = result

        if is_task_complete:
            step_span.set_attribute(_GEN_AI_REACT_FINISH_REASON, "complete")
        elif feedback and "ERROR:" in feedback:
            step_span.set_attribute(_GEN_AI_REACT_FINISH_REASON, "parse_error")

        # Span stays open: closed by next iteration or _RunAgentLoopWrapper
        return result


# ═══════════════════════════════════════════════════════════════════════════
# P1 — TASK span: parser.parse_response
# ═══════════════════════════════════════════════════════════════════════════


class _ParseResponseWrapper:
    """Wrap ``parser.parse_response`` to produce a **TASK** span.

    Per spec: span name ``run_task {task_name}``,
    ``gen_ai.span.kind=TASK``, ``gen_ai.operation.name=run_task``.
    """

    def __init__(self, tracer, parser_type):
        self._tracer = tracer
        self._parser_type = parser_type

    def __call__(self, wrapped, instance, args, kwargs):
        # parse_response signature: (self, response: str)
        response_text = args[0] if args else kwargs.get("response", "")

        with self._tracer.start_as_current_span(
            "run_task parse_response",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(
                _GEN_AI_SPAN_KIND,
                _SPAN_KIND_TASK,
            )
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_RUN_TASK)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute("terminus2.parser", self._parser_type)

            if response_text is not None:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_messages_json("assistant", response_text),
                )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

            span.set_attribute(
                "terminus2.task_complete", result.is_task_complete
            )
            span.set_attribute(
                "terminus2.commands.count", len(result.commands)
            )

            output_summary = {
                "is_task_complete": result.is_task_complete,
                "commands": [
                    {
                        "keystrokes": getattr(c, "keystrokes", ""),
                        "duration": getattr(c, "duration", None),
                    }
                    for c in result.commands
                ],
                "error": result.error or "",
                "warning": result.warning or "",
            }
            try:
                output_value = json.dumps(output_summary, ensure_ascii=False)
            except Exception:
                output_value = str(output_summary)
            span.set_attribute(
                _GEN_AI_OUTPUT_MESSAGES,
                _text_messages_json("assistant", output_value),
            )

            if result.error:
                span.set_attribute("terminus2.parse.error", str(result.error))

            if result.warning:
                span.set_attribute(
                    "terminus2.parse.warning", str(result.warning)
                )

            span.set_status(Status(StatusCode.OK))
            return result


# ═══════════════════════════════════════════════════════════════════════════
# P2 — CHAIN span: Terminus2._summarize
# ═══════════════════════════════════════════════════════════════════════════


class _SummarizeWrapper:
    """Wrap ``Terminus2._summarize`` to produce a **CHAIN** span.

    Per spec: span name ``chain {chain_name}``,
    ``gen_ai.span.kind=CHAIN``. The summarize handoff itself triggers
    multiple inner LLM calls so it semantically maps to a Chain.
    """

    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        with self._tracer.start_as_current_span(
            "chain summarize",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(
                _GEN_AI_SPAN_KIND,
                _SPAN_KIND_CHAIN,
            )
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_TASK)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

            span.set_status(Status(StatusCode.OK))
            return result
