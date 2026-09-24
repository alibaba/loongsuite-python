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

"""Wrapper classes for WildToolBench instrumentation.

Each wrapper corresponds to one patch point and manages the lifecycle
of one or more span types.

Round 2 fix highlights (see ``llm-dev/execute.md`` § "修订记录 (Round 2 fix)"):

H1
    TOOL span parent is now STEP rather than CHAIN. Each STEP invocation is
    appended to a per-chain list in :data:`_chain_step_invocations`; when the
    chain wrapper post-processes ``inference_log`` it looks up the matching
    STEP span by ``round`` and uses
    :func:`opentelemetry.trace.set_span_in_context` so ``start_execute_tool``
    parents the TOOL span on the STEP context (even if STEP is already
    closed — its :class:`SpanContext` remains a valid parent reference).

H2
    The OpenAI v2 provider instrumentation (0.62b1) writes only the legacy
    ``gen_ai.system`` attribute to its LLM span. The wildtool plugin now
    writes both ``gen_ai.system`` and ``gen_ai.provider.name`` on the STEP
    span as a fallback so the new semantic-conventions attribute is present
    in the trace tree even before the upstream OpenAI v2 instrumentation
    catches up. We do **not** patch the OpenAI v2 instrumentation itself.

M1
    ``input.value`` (last user message in the chain's ``messages``, truncated
    to 4096 chars) and ``output.value`` (a JSON of action label, task index
    and is_optimal) are written on the CHAIN span.

M2
    ``gen_ai.react.finish_reason`` is derived from ``inference_log`` on the
    *last* (currently active) STEP. Mappings:

    ``"parse_tool_calls_failed"``
        ``error_reason`` contains "parse tool_calls failed".
    ``"action_name_mismatch"``
        ``error_reason`` contains "action name not in candidate".
    ``"empty_response"``
        ``error_reason`` contains "tool_calls and content are None".
    ``"error"``
        request raised an exception (handled in
        :class:`WildToolRequestWrapper`).

M3
    ``gen_ai.tool.call.arguments``, ``gen_ai.tool.call.result`` and
    ``gen_ai.tool.description`` are written explicitly on TOOL spans
    *before* close as a fallback. ``opentelemetry-util-genai`` gates these
    sensitive attributes behind ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_*`` env
    vars; the wildtool plugin always writes them since wtb data is
    benchmark-synthetic and never PII.
"""

import json
import logging
from contextvars import ContextVar
from dataclasses import asdict
from typing import List, Optional

from opentelemetry.trace import StatusCode, set_span_in_context
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

# ─────────────────────────── ContextVars ───────────────────────────────
# The CHAIN wrapper opens a new logical "chain" by flipping ``_in_chain``
# and resetting the counter. The REQUEST wrapper reads these to decide
# whether to create a STEP span and what round number to assign.
_in_chain: ContextVar[bool] = ContextVar("_wt_in_chain", default=False)

# Currently open STEP invocation. Used by the parse wrapper to attach
# token attributes to the right span.
_step_invocation: ContextVar[Optional[ReactStepInvocation]] = ContextVar(
    "_wt_step_inv", default=None
)
_step_counter: ContextVar[int] = ContextVar("_wt_step_ctr", default=0)

# Per-chain list of every STEP invocation created in the current chain
# (in `round` order). The chain wrapper allocates this list on entry and
# uses it after ``wrapped`` returns to re-parent TOOL spans onto the
# matching STEP. Even if a STEP span is already ``end()``-ed, its
# :class:`SpanContext` stays valid as a parent reference for new spans.
_chain_step_invocations: ContextVar[Optional[List[ReactStepInvocation]]] = (
    ContextVar("_wt_chain_step_invs", default=None)
)

_PROVIDER_FALLBACK_NAME = "openai"
_INPUT_VALUE_MAX_CHARS = 4096
_MESSAGE_CONTENT_MAX_CHARS = 4096


def _close_active_step(handler: ExtendedTelemetryHandler) -> None:
    """Close the currently active STEP span, if any."""
    prev = _step_invocation.get()
    if prev is not None:
        try:
            handler.stop_react_step(prev)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to close step: %s", e)
        _step_invocation.set(None)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _tasks_to_input_messages(test_entry) -> List[InputMessage]:
    if not isinstance(test_entry, dict):
        return []
    tasks = test_entry.get("english_tasks")
    if not isinstance(tasks, list):
        return []

    messages = []
    for task in tasks:
        if task in (None, "", [], {}):
            continue
        messages.append(
            InputMessage(
                role="user",
                parts=[
                    Text(
                        content=_truncate(
                            _stringify(task), _MESSAGE_CONTENT_MAX_CHARS
                        )
                    )
                ],
            )
        )
    return messages


def _task_results_to_output_messages(result) -> List[OutputMessage]:
    task_results = _extract_task_results(result)
    messages = []
    for task_result in task_results:
        content = _extract_task_result_output(task_result)
        if content in (None, "", [], {}):
            continue
        messages.append(
            OutputMessage(
                role="assistant",
                parts=[
                    Text(
                        content=_truncate(
                            _stringify(content), _MESSAGE_CONTENT_MAX_CHARS
                        )
                    )
                ],
                finish_reason=_extract_finish_reason(task_result),
            )
        )
    return messages


def _get_message_attributes(input_messages, output_messages) -> dict:
    attributes = {}
    try:
        if input_messages:
            attributes["gen_ai.input.messages"] = json.dumps(
                [asdict(message) for message in input_messages],
                ensure_ascii=False,
            )
        if output_messages:
            attributes["gen_ai.output.messages"] = json.dumps(
                [asdict(message) for message in output_messages],
                ensure_ascii=False,
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to serialize message attrs: %s", e)
    return attributes


def _set_message_attributes(invocation) -> None:
    attributes = _get_message_attributes(
        invocation.input_messages, invocation.output_messages
    )
    if not attributes:
        return
    invocation.attributes.update(attributes)
    span = invocation.span
    if span is None or not span.is_recording():
        return
    try:
        span.set_attributes(attributes)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to set message attrs: %s", e)


def _extract_task_results(result) -> List:
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []

    for key in (
        "result",
        "results",
        "inference_result",
        "inference_results",
        "result_list",
        "task_results",
        "answer",
        "answers",
    ):
        value = result.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        if value not in (None, "", [], {}):
            return [value]

    if any(
        key in result
        for key in (
            "action_name_label",
            "is_optimal",
            "inference_log",
            "inference_output",
            "final_answer",
        )
    ):
        return [result]
    return []


def _extract_task_result_output(task_result):
    if not isinstance(task_result, dict):
        return task_result

    for key in ("final_answer", "answer", "output", "result"):
        value = task_result.get(key)
        if value not in (None, "", [], {}):
            return value

    inference_log = task_result.get("inference_log")
    output_from_log = _extract_output_from_inference_log(inference_log)
    if output_from_log not in (None, "", [], {}):
        return output_from_log

    label = task_result.get("action_name_label")
    if label is not None or "is_optimal" in task_result:
        return {
            "action_name_label": label,
            "is_optimal": task_result.get("is_optimal"),
        }
    return None


def _extract_output_from_inference_log(inference_log):
    if not isinstance(inference_log, dict):
        return None

    for key in sorted(
        (k for k in inference_log if k.startswith("step_")),
        key=_step_log_sort_key,
        reverse=True,
    ):
        step_data = inference_log.get(key)
        if not isinstance(step_data, dict):
            continue

        output = step_data.get("inference_output")
        if isinstance(output, dict):
            for output_key in (
                "content",
                "reasoning_content",
                "current_action_name_label",
                "error_reason",
            ):
                value = output.get(output_key)
                if value not in (None, "", [], {}):
                    return value

        answer = step_data.get("inference_answer")
        if isinstance(answer, dict):
            candidate = answer.get("candidate_0_answer_function_list")
            if isinstance(candidate, dict):
                observation = candidate.get("observation")
                if observation not in (None, "", [], {}):
                    return observation
            if answer not in (None, "", [], {}):
                return answer
    return None


def _step_log_sort_key(key: str) -> int:
    try:
        return int(key[len("step_") :])
    except (TypeError, ValueError):
        return -1


def _extract_finish_reason(task_result) -> str:
    if isinstance(task_result, dict):
        label = task_result.get("action_name_label")
        if label == "error":
            return "error"
    return "stop"


class WildToolEntryWrapper:
    """P1: Wraps multi_threaded_inference → ENTRY span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        # Signature: multi_threaded_inference(handler, model_name, test_case).
        # We only need model_name and test_case for ENTRY attributes; the
        # handler instance flows through as args[0] untouched.
        model_name = args[1] if len(args) > 1 else kwargs.get("model_name", "")
        test_case = args[2] if len(args) > 2 else kwargs.get("test_case", {})

        invocation = EntryInvocation(
            session_id=test_case.get("id"),
            input_messages=_tasks_to_input_messages(test_case),
            attributes={
                "gen_ai.framework": "wildtool",
                "gen_ai.request.model": model_name,
                "wildtool.turn_count": len(test_case.get("english_tasks", [])),
            },
        )
        self._handler.start_entry(invocation)
        _set_message_attributes(invocation)
        try:
            result = wrapped(*args, **kwargs)
            invocation.output_messages = _task_results_to_output_messages(
                result
            )
            _set_message_attributes(invocation)
            self._handler.stop_entry(invocation)
            return result
        except Exception as e:
            _set_message_attributes(invocation)
            self._handler.fail_entry(
                invocation, Error(message=str(e), type=type(e))
            )
            raise


class WildToolAgentWrapper:
    """P2: Wraps BaseHandler.inference_multi_turn → AGENT span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        test_entry = args[0] if args else kwargs.get("test_entry", {})

        attributes = {
            "gen_ai.framework": "wildtool",
            "wildtool.turn_count": len(
                test_entry.get("english_answer_list", [])
            ),
        }

        env_info = test_entry.get("english_env_info", "")
        if env_info:
            attributes["gen_ai.system_instructions"] = json.dumps(
                [{"type": "text", "content": f"Current Date: {env_info}"}],
                ensure_ascii=False,
            )

        tools = test_entry.get("english_tools")
        if isinstance(tools, list) and tools:
            attributes["gen_ai.tool.definitions"] = json.dumps(
                tools,
                ensure_ascii=False,
            )

        invocation = InvokeAgentInvocation(
            provider=None,
            agent_name=type(instance).__name__,
            input_messages=_tasks_to_input_messages(test_entry),
            conversation_id=test_entry.get("id"),
            request_model=getattr(instance, "model_name", None),
            attributes=attributes,
        )
        self._handler.start_invoke_agent(invocation)
        _set_message_attributes(invocation)
        try:
            result = wrapped(*args, **kwargs)
            invocation.output_messages = _task_results_to_output_messages(
                result
            )
            _set_message_attributes(invocation)
            total_input = 0
            total_output = 0
            for task_result in result or []:
                if isinstance(task_result, dict):
                    total_input += sum(
                        task_result.get("input_token_count", [])
                    )
                    total_output += sum(
                        task_result.get("output_token_count", [])
                    )
            if total_input:
                invocation.input_tokens = total_input
            if total_output:
                invocation.output_tokens = total_output
            self._handler.stop_invoke_agent(invocation)
            return result
        except Exception as e:
            _set_message_attributes(invocation)
            self._handler.fail_invoke_agent(
                invocation, Error(message=str(e), type=type(e))
            )
            raise


class WildToolChainWrapper:
    """P3: Wraps BaseHandler.inference_and_eval_multi_step → CHAIN span.

    Also manages the lifecycle of the final STEP span and creates TOOL spans
    from the returned ``inference_log`` after the original function completes.
    Round 2 fixes (H1/M1/M2/M3) are implemented here.
    """

    def __init__(self, handler: ExtendedTelemetryHandler, instrumentor=None):
        self._handler = handler
        self._instrumentor = instrumentor

    def __call__(self, wrapped, instance, args, kwargs):
        if self._instrumentor is not None and instance is not None:
            try:
                self._instrumentor.ensure_handler_class_patched(type(instance))
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to ensure subclass patched: %s", e)

        inference_data = args[0] if args else kwargs.get("inference_data", {})
        if not isinstance(inference_data, dict):
            inference_data = {}
        task_idx = inference_data.get("task_idx", 0)
        test_entry_id = inference_data.get("test_entry_id", "")

        span_name = f"workflow task_{task_idx}"
        tracer = self._handler._tracer

        chain_token = _in_chain.set(True)
        counter_token = _step_counter.set(0)
        step_token = _step_invocation.set(None)
        chain_steps: List[ReactStepInvocation] = []
        chain_steps_token = _chain_step_invocations.set(chain_steps)

        chain_attributes = {
            "gen_ai.span.kind": "CHAIN",
            "gen_ai.operation.name": "workflow",
            "gen_ai.framework": "wildtool",
            "wildtool.task_idx": task_idx,
            "wildtool.test_entry_id": test_entry_id,
        }

        # M1: Capture last user message as ``input.value`` BEFORE running the
        # wrapped function (the wtb function mutates ``messages`` in place).
        input_value = self._extract_input_value(inference_data)
        if input_value is not None:
            chain_attributes["input.value"] = input_value

        with tracer.start_as_current_span(
            name=span_name, attributes=chain_attributes
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                # M2: Set finish_reason on the currently active (last) STEP
                # BEFORE we close it. Only the terminal step ever carries an
                # error finish_reason (every wtb error path triggers `break`).
                if isinstance(result, dict):
                    self._apply_last_step_finish_reason(
                        result.get("inference_log", {})
                    )

                _close_active_step(self._handler)

                if isinstance(result, dict):
                    label = result.get("action_name_label", "")
                    is_optimal = bool(result.get("is_optimal", False))
                    span.set_attribute("wildtool.action_name_label", label)
                    span.set_attribute("wildtool.is_optimal", is_optimal)

                    # M1: ``output.value`` summarising chain outcome.
                    try:
                        span.set_attribute(
                            "output.value",
                            json.dumps(
                                {
                                    "action_name_label": label,
                                    "task_idx": task_idx,
                                    "is_optimal": is_optimal,
                                },
                                ensure_ascii=False,
                            ),
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Failed to set output.value: %s", e)

                    # H1 + M3: re-parent TOOL spans on STEP and force-write
                    # tool call sensitive attributes.
                    self._create_tool_spans_from_log(
                        result.get("inference_log", {}),
                        inference_data,
                        chain_steps,
                    )

                span.set_status(StatusCode.OK)
                return result
            except Exception as e:
                _close_active_step(self._handler)
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise
            finally:
                _chain_step_invocations.reset(chain_steps_token)
                _step_counter.reset(counter_token)
                _step_invocation.reset(step_token)
                _in_chain.reset(chain_token)

    # -- M1 ---------------------------------------------------------------

    @staticmethod
    def _extract_input_value(inference_data) -> Optional[str]:
        msgs = (
            inference_data.get("messages")
            if isinstance(inference_data, dict)
            else None
        )
        if not isinstance(msgs, list):
            return None
        for m in reversed(msgs):
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if content is None:
                continue
            text = _stringify(content)
            return _truncate(text, _INPUT_VALUE_MAX_CHARS)
        return None

    # -- M2 ---------------------------------------------------------------

    def _apply_last_step_finish_reason(self, inference_log) -> None:
        if not isinstance(inference_log, dict):
            return
        current_step = _step_invocation.get()
        if current_step is None or current_step.round is None:
            return
        step_key = f"step_{current_step.round - 1}"
        step_data = inference_log.get(step_key)
        if not isinstance(step_data, dict):
            return
        output = step_data.get("inference_output") or {}
        if not isinstance(output, dict):
            return
        label = output.get("current_action_name_label")
        error_reason = output.get("error_reason") or ""
        reason = self._derive_step_finish_reason(label, error_reason)
        if reason is None:
            return
        # Setting `invocation.finish_reason` is enough — the util-genai
        # `_apply_react_step_finish_attributes` writes
        # ``gen_ai.react.finish_reason`` from this field on stop.
        current_step.finish_reason = reason

    @staticmethod
    def _derive_step_finish_reason(label, error_reason: str) -> Optional[str]:
        """Map wtb inference_log error_reason → gen_ai.react.finish_reason."""
        if label != "error":
            return None
        if "parse tool_calls failed" in error_reason:
            return "parse_tool_calls_failed"
        if "action name not in candidate" in error_reason:
            return "action_name_mismatch"
        if "tool_calls and content are None" in error_reason:
            return "empty_response"
        return "error"

    # -- H1 + M3 ----------------------------------------------------------

    def _create_tool_spans_from_log(
        self,
        inference_log,
        inference_data,
        chain_steps: List[ReactStepInvocation],
    ) -> None:
        """Post-hoc TOOL span creation from inference_log.

        Uses the per-chain STEP invocation list to parent each TOOL span on
        the matching STEP span (H1).  Sensitive tool-call attributes are
        written explicitly on the span (M3) so they appear regardless of
        ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_*`` settings.
        """
        if not isinstance(inference_log, dict):
            return

        # round → SpanContext-bearing OTel context for parenting
        step_ctx_by_round = {}
        for step_inv in chain_steps:
            if step_inv.round is None or step_inv.span is None:
                continue
            try:
                step_ctx_by_round[step_inv.round] = set_span_in_context(
                    step_inv.span
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to compute step parent context: %s", e)

        # tool name → description (for gen_ai.tool.description)
        tool_desc_map = {}
        tools = (
            inference_data.get("tools")
            if isinstance(inference_data, dict)
            else None
        )
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                func = tool.get("function") or tool
                if not isinstance(func, dict):
                    continue
                name = func.get("name")
                desc = func.get("description")
                if name:
                    tool_desc_map[name] = desc

        # Extract tool observations from final messages keyed by tool_call_id;
        # wtb only embeds them in messages (not in inference_answer) for the
        # tool_call branch.
        observation_by_call_id = {}
        messages = (
            inference_data.get("messages")
            if isinstance(inference_data, dict)
            else None
        )
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict) or msg.get("role") != "tool":
                    continue
                tid = msg.get("tool_call_id")
                if tid is None:
                    continue
                content = msg.get("content")
                if content is None:
                    continue
                observation_by_call_id[tid] = (
                    content
                    if isinstance(content, str)
                    else _stringify(content)
                )

        for key in sorted(k for k in inference_log if k.startswith("step_")):
            try:
                step_idx = int(key[len("step_") :])
            except ValueError:
                continue
            round_num = step_idx + 1

            step_data = inference_log[key]
            if not isinstance(step_data, dict):
                continue
            output = step_data.get("inference_output") or {}
            if not isinstance(output, dict):
                continue
            tool_calls = output.get("tool_calls")
            label = output.get("current_action_name_label")
            if not tool_calls or label != "correct":
                continue

            answer_data = step_data.get("inference_answer") or {}
            candidate = (
                answer_data.get("candidate_0_answer_function_list")
                if isinstance(answer_data, dict)
                else None
            ) or {}
            candidate_observation = (
                candidate.get("observation")
                if isinstance(candidate, dict)
                else None
            )

            parent_ctx = step_ctx_by_round.get(round_num)

            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function") or {}
                if not isinstance(func, dict):
                    func = {}
                tool_name = func.get("name", "unknown")
                tool_id = tc.get("id")
                tool_args_raw = func.get("arguments", "")
                tool_args_str = (
                    tool_args_raw
                    if isinstance(tool_args_raw, str)
                    else _stringify(tool_args_raw)
                )

                observation_str: Optional[str] = None
                if tool_id is not None and tool_id in observation_by_call_id:
                    observation_str = observation_by_call_id[tool_id]
                elif candidate_observation is not None:
                    observation_str = (
                        candidate_observation
                        if isinstance(candidate_observation, str)
                        else _stringify(candidate_observation)
                    )

                description = tool_desc_map.get(tool_name)

                invocation = ExecuteToolInvocation(
                    tool_name=tool_name,
                    tool_call_id=tool_id,
                    tool_call_arguments=tool_args_str,
                    tool_call_result=observation_str,
                    tool_type="function",
                    tool_description=description,
                    attributes={
                        "wildtool.tool.execution_mode": "ground_truth_replay",
                    },
                )

                try:
                    self._handler.start_execute_tool(
                        invocation, context=parent_ctx
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to start_execute_tool: %s", e)
                    continue

                # M3: explicitly write tool_call sensitive attrs. The
                # util-genai `_get_tool_call_data_attributes` helper guards
                # these behind experimental-mode + content-capture-mode env
                # vars which are not always set in real deployments.
                tool_span = invocation.span
                if tool_span is not None and tool_span.is_recording():
                    try:
                        tool_span.set_attribute(
                            "gen_ai.tool.call.arguments", tool_args_str
                        )
                        if observation_str is not None:
                            tool_span.set_attribute(
                                "gen_ai.tool.call.result", observation_str
                            )
                        if description:
                            tool_span.set_attribute(
                                "gen_ai.tool.description", description
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Failed to set tool span attrs: %s", e)

                try:
                    self._handler.stop_execute_tool(invocation)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to stop_execute_tool: %s", e)


class WildToolRequestWrapper:
    """P4: Wraps BaseHandler._request_tool_call.

    Creates STEP span (ReactStepInvocation) before each LLM call.
    Extracts latency from return value. Also writes the H2 provider-name
    fallback attributes (``gen_ai.system`` + ``gen_ai.provider.name``) on
    the STEP span so the new semconv attribute is present in the trace
    even when the upstream OpenAI v2 instrumentation only emits the legacy
    ``gen_ai.system``.
    """

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        if not _in_chain.get():
            return wrapped(*args, **kwargs)

        # Close the previous step (the natural end-of-step is when the next
        # request fires). The STEP span's SpanContext stays valid as a
        # parent for TOOL spans created later.
        _close_active_step(self._handler)

        step_num = _step_counter.get() + 1
        _step_counter.set(step_num)

        step_inv = ReactStepInvocation(round=step_num)
        try:
            self._handler.start_react_step(step_inv)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to start react step: %s", e)
            return wrapped(*args, **kwargs)

        # H2: provider-name fallback attributes. Written on the STEP, not
        # on the LLM span, because the LLM span is owned by the OpenAI v2
        # provider instrumentation and is created lazily inside the wtb
        # request implementation.
        if step_inv.span is not None and step_inv.span.is_recording():
            try:
                step_inv.span.set_attribute(
                    "gen_ai.system", _PROVIDER_FALLBACK_NAME
                )
                step_inv.span.set_attribute(
                    "gen_ai.provider.name", _PROVIDER_FALLBACK_NAME
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to set provider fallback attrs: %s", e)

        # Track this step for H1 TOOL re-parenting.
        chain_steps = _chain_step_invocations.get()
        if chain_steps is not None:
            chain_steps.append(step_inv)
        _step_invocation.set(step_inv)

        try:
            result = wrapped(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                _, latency = result
                if step_inv.span and step_inv.span.is_recording():
                    try:
                        step_inv.span.set_attribute(
                            "wildtool.latency", float(latency)
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Failed to set wildtool.latency: %s", e)
            return result
        except Exception as e:
            step_inv.finish_reason = "error"
            self._handler.fail_react_step(
                step_inv, Error(message=str(e), type=type(e))
            )
            _step_invocation.set(None)
            raise


class WildToolParseWrapper:
    """P5: Wraps BaseHandler._parse_api_response.

    Extracts token counts from parsed response and sets them on the
    current STEP span as attributes.
    """

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)

        step_inv = _step_invocation.get()
        if step_inv and step_inv.span and step_inv.span.is_recording():
            if isinstance(result, dict):
                input_t = result.get("input_token")
                output_t = result.get("output_token")
                if input_t is not None:
                    step_inv.span.set_attribute(
                        "gen_ai.usage.input_tokens", input_t
                    )
                if output_t is not None:
                    step_inv.span.set_attribute(
                        "gen_ai.usage.output_tokens", output_t
                    )

        return result
