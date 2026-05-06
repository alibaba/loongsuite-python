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

"""Wrapper classes for the BFCL v4 instrumentation.

Each wrapper follows the standard ``wrapt`` callable contract::

    def __call__(self, wrapped, instance, args, kwargs):
        ...

All wrappers rely on :func:`get_extended_telemetry_handler` (LoongSuite
``util-genai``) to create the actual spans, so that ENTRY / AGENT / STEP /
TOOL spans get the canonical ``gen_ai.span.kind`` and operation-name values
that the LoongSuite semantic-validator expects.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Iterable, List, Optional

from opentelemetry.instrumentation.bfclv4.internal.attributes import (
    BFCL_NUM_THREADS,
    BFCL_OSS_BACKEND,
    BFCL_QUERY_MODE,
    BFCL_RUN_IDS,
    BFCL_TEST_CASE_COUNT,
    BFCL_TEST_CATEGORY,
    BFCL_TEST_ENTRY_ID,
    BFCL_TOOL_DURATION_IS_ESTIMATED,
    BFCL_TOOL_INDEX,
    BFCL_TURN_IDX,
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_PROVIDER_NAME,
)
from opentelemetry.instrumentation.bfclv4.internal.provider import (
    OSS_BACKEND_ENV,
    infer_provider,
)
from opentelemetry.instrumentation.bfclv4.internal.state import (
    bump_round,
    bump_turn,
    init_state,
    next_tool_index,
    reset_state,
)
from opentelemetry.instrumentation.bfclv4.internal.threading_propagation import (
    ContextPropagatingExecutor,
)
from opentelemetry.instrumentation.bfclv4.utils import (
    GenAIHookHelper,
    to_text_input,
    to_text_output,
    truncate_text,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _flatten_tokens(value: Any) -> Optional[int]:
    """Sum a possibly nested ``int|float|list|list[list]`` BFCL token field."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, Iterable):
        total = 0
        any_seen = False
        for item in value:
            sub = _flatten_tokens(item)
            if sub is not None:
                total += sub
                any_seen = True
        if any_seen:
            return total
    return None


def _test_category_from_id(test_entry_id: Optional[str]) -> Optional[str]:
    if not test_entry_id or "_" not in test_entry_id:
        return None
    return test_entry_id.rsplit("_", 1)[0]


def _join_test_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        joined = ",".join(str(v) for v in value if v is not None)
        return joined or None
    return str(value)


# ---------------------------------------------------------------------------
# ENTRY wrapper


class GenerateResultsWrapper:
    """Wraps ``bfcl_eval._llm_response_generation.generate_results``.

    Responsibilities:

    * Open the ENTRY span (``enter_ai_application_system``).
    * Temporarily swap the ``ThreadPoolExecutor`` reference inside the BFCL
      generation module to a context-propagating subclass so that AGENT spans
      created in worker threads inherit the ENTRY span as parent.
    * Publish ``args.backend`` to ``BFCL_BACKEND`` so that
      :func:`infer_provider` can attribute OSS spans to vllm / sglang.
    """

    def __init__(self, helper: GenAIHookHelper) -> None:
        self._helper = helper

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        # ``generate_results(args, model_name, test_cases_total)``
        cli_args = args[0] if len(args) >= 1 else kwargs.get("args")
        model_name = args[1] if len(args) >= 2 else kwargs.get("model_name")
        test_cases_total = (
            args[2] if len(args) >= 3 else kwargs.get("test_cases_total")
        )

        try:
            from bfcl_eval import (  # noqa: PLC0415
                _llm_response_generation as _bfcl_gen,
            )
        except ImportError:
            return wrapped(*args, **kwargs)

        original_executor = getattr(_bfcl_gen, "ThreadPoolExecutor", None)
        if original_executor is not None:
            _bfcl_gen.ThreadPoolExecutor = ContextPropagatingExecutor

        backend_value = (
            _safe_get(cli_args, "backend", None) if cli_args is not None else None
        )
        previous_backend_env = os.environ.get(OSS_BACKEND_ENV)
        if backend_value:
            os.environ[OSS_BACKEND_ENV] = str(backend_value)

        session_id_default = None
        if model_name is not None:
            try:
                session_id_default = f"{model_name}@{int(time.time())}"
            except Exception:  # noqa: BLE001
                session_id_default = None
        session_id = (
            os.environ.get("BFCL_SESSION_ID") or session_id_default
        )

        entry_inv = EntryInvocation(session_id=session_id)
        handler = get_extended_telemetry_handler()

        attributes = {GEN_AI_FRAMEWORK: FRAMEWORK_NAME}
        category_value = _join_test_category(
            _safe_get(cli_args, "test_category", None)
        )
        if category_value:
            attributes[BFCL_TEST_CATEGORY] = category_value
        num_threads = _safe_get(cli_args, "num_threads", None)
        if num_threads is not None:
            try:
                attributes[BFCL_NUM_THREADS] = int(num_threads)
            except (TypeError, ValueError):
                pass
        if isinstance(test_cases_total, (list, tuple)):
            attributes[BFCL_TEST_CASE_COUNT] = len(test_cases_total)
        attributes[BFCL_RUN_IDS] = bool(
            _safe_get(cli_args, "run_ids", False)
        )

        try:
            with handler.entry(entry_inv) as inv:
                if inv.span is not None and inv.span.is_recording():
                    for key, value in attributes.items():
                        try:
                            inv.span.set_attribute(key, value)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "bfclv4 ENTRY set_attribute(%s) failed",
                                key,
                                exc_info=True,
                            )
                return wrapped(*args, **kwargs)
        finally:
            if original_executor is not None:
                try:
                    _bfcl_gen.ThreadPoolExecutor = original_executor
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "bfclv4 ENTRY: failed to restore ThreadPoolExecutor",
                        exc_info=True,
                    )
            if backend_value:
                if previous_backend_env is None:
                    os.environ.pop(OSS_BACKEND_ENV, None)
                else:
                    os.environ[OSS_BACKEND_ENV] = previous_backend_env


# ---------------------------------------------------------------------------
# AGENT wrapper


_BFCL_INFERENCE_ERROR_PREFIX = "Error during inference:"


class BaseHandlerInferenceWrapper:
    """Wraps ``BaseHandler.inference``.

    Creates the AGENT span (kind=AGENT, op=invoke_agent) and initialises the
    per-thread ReAct state used by the STEP wrapper.

    BFCL's outer ``multi_threaded_inference`` catches every exception and
    converts it into a ``"Error during inference: ..."`` string; we mirror
    that behaviour by setting the AGENT span status to ERROR when the
    returned ``result`` looks like an error string, instead of relying on
    a re-raised exception.
    """

    def __init__(self, helper: GenAIHookHelper) -> None:
        self._helper = helper

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        # ``inference(self, test_entry, include_input_log, exclude_state_log)``
        test_entry = args[0] if args else kwargs.get("test_entry")
        if not isinstance(test_entry, dict):
            return wrapped(*args, **kwargs)

        provider, extra_attrs = infer_provider(instance)
        request_model = getattr(instance, "model_name", None)
        test_entry_id = test_entry.get("id")
        category = _test_category_from_id(test_entry_id)
        involved_classes = test_entry.get("involved_classes") or []
        agent_description = (
            ", ".join(str(c) for c in involved_classes)
            if isinstance(involved_classes, (list, tuple))
            else None
        )

        invocation = InvokeAgentInvocation(
            provider=provider or "unknown",
            request_model=request_model,
            agent_id=test_entry_id,
            agent_name=category or "bfcl_agent",
            agent_description=agent_description or None,
            conversation_id=test_entry_id,
        )

        token = init_state()
        handler = get_extended_telemetry_handler()
        try:
            with handler.invoke_agent(invocation) as inv:
                if inv.span is not None and inv.span.is_recording():
                    inv.span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
                    if provider:
                        inv.span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
                    if test_entry_id is not None:
                        inv.span.set_attribute(
                            BFCL_TEST_ENTRY_ID, test_entry_id
                        )
                    if category is not None:
                        inv.span.set_attribute(BFCL_TEST_CATEGORY, category)
                    for key, value in extra_attrs.items():
                        if value is not None:
                            inv.span.set_attribute(key, value)

                # Capture inputs for the AGENT (gated by content-capture mode).
                question = test_entry.get("question")
                if question is not None:
                    inv.input_messages = to_text_input(
                        "user", truncate_text(_safe_str(question))
                    )

                # Run the original inference call.
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    # The CM will mark the span as failed; we leave it to
                    # the handler/CM to call ``fail_invoke_agent``.
                    raise exc

                # Detect BFCL's own captured error path (no exception raised
                # but the returned result is the error string).
                result_payload = (
                    result[0] if isinstance(result, tuple) and result else None
                )
                metadata_payload = (
                    result[1]
                    if isinstance(result, tuple) and len(result) >= 2
                    else None
                )

                if (
                    isinstance(result_payload, str)
                    and result_payload.startswith(_BFCL_INFERENCE_ERROR_PREFIX)
                    and inv.span is not None
                    and inv.span.is_recording()
                ):
                    try:
                        from opentelemetry.trace import Status, StatusCode

                        inv.span.set_status(
                            Status(StatusCode.ERROR, result_payload[:200])
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "bfclv4 AGENT: failed to set ERROR status",
                            exc_info=True,
                        )

                if isinstance(metadata_payload, dict):
                    input_tokens = _flatten_tokens(
                        metadata_payload.get("input_token_count")
                    )
                    output_tokens = _flatten_tokens(
                        metadata_payload.get("output_token_count")
                    )
                    if input_tokens is not None:
                        inv.input_tokens = input_tokens
                    if output_tokens is not None:
                        inv.output_tokens = output_tokens

                if result_payload is not None:
                    inv.output_messages = to_text_output(
                        "assistant",
                        truncate_text(_safe_str(result_payload)),
                    )

                return result
        finally:
            reset_state(token)


def _safe_str(value: Any) -> str:
    try:
        if isinstance(value, str):
            return value
        import json

        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        try:
            return str(value)
        except Exception:  # noqa: BLE001
            return "<unserialisable>"


# ---------------------------------------------------------------------------
# STEP wrapper


class QueryWrapper:
    """Wraps ``<Handler>._query_FC`` / ``_query_prompting``.

    Creates a ReAct STEP span, attaches token usage by re-calling the
    handler's matching ``_parse_query_response_*`` (which is documented as
    side-effect-free).
    """

    def __init__(self, helper: GenAIHookHelper, mode: str) -> None:
        self._helper = helper
        self._mode = mode  # "FC" or "prompting"

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        round_idx = bump_round()
        provider, extra_attrs = infer_provider(instance)

        invocation = ReactStepInvocation(round=round_idx)
        handler_obj = get_extended_telemetry_handler()
        with handler_obj.react_step(invocation) as step_inv:
            span = step_inv.span
            if span is not None and span.is_recording():
                span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
                span.set_attribute(BFCL_QUERY_MODE, self._mode)
                if provider:
                    span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
                model_name = getattr(instance, "model_name", None)
                if model_name:
                    span.set_attribute(
                        "gen_ai.request.model", str(model_name)
                    )
                from opentelemetry.instrumentation.bfclv4.internal.state import (
                    get_state,
                )

                state = get_state()
                if state is not None:
                    span.set_attribute(BFCL_TURN_IDX, state.get("turn_idx", 0))
                for key, value in extra_attrs.items():
                    if value is not None:
                        span.set_attribute(key, value)

            try:
                api_response, query_latency = wrapped(*args, **kwargs)
            except Exception:
                # Let the context-manager mark the span as failed; the BFCL
                # outer try/except will turn this into an "Error during
                # inference: ..." result string at the AGENT layer.
                raise

            # Post-call attribute enrichment - use try/except so that any
            # vendor-side parsing surprise never breaks BFCL itself.
            #
            # IMPORTANT: We must NOT re-call ``_parse_query_response_*`` here,
            # because for streaming providers (e.g. Qwen DashScope) the
            # ``api_response`` is a single-pass generator that the parser
            # consumes; calling it twice leaves BFCL's own subsequent call to
            # the parser with an exhausted iterator, which crashes inference
            # with ``UnboundLocalError: chunk``. Token usage will instead be
            # recovered later from the AGENT-level metadata payload.
            try:
                if span is not None and span.is_recording():
                    if isinstance(query_latency, (int, float)):
                        try:
                            span.set_attribute(
                                "gen_ai.response.time_to_first_token",
                                int(float(query_latency) * 1e9),
                            )
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                logger.debug(
                    "bfclv4 STEP: post-call enrichment failed", exc_info=True
                )

            return api_response, query_latency


def _infer_finish_reason(model_responses: Any) -> str:
    """Best-effort heuristic for ``gen_ai.react.finish_reason``."""
    if model_responses is None:
        return "unknown"
    if isinstance(model_responses, list):
        if len(model_responses) == 0:
            return "empty_response"
        if len(model_responses) == 1 and not model_responses[0]:
            return "empty_response"
        return "tool_calls"
    if isinstance(model_responses, str):
        # Prompting models often return decoded strings even when there are
        # no tool calls - treat as "stop" so downstream callers know there is
        # no further work to do.
        return "stop"
    return "continue"


# ---------------------------------------------------------------------------
# turn_idx maintenance wrappers (no spans)


class TurnBumpWrapper:
    """Wraps ``<Handler>.add_first_turn_message_*`` and
    ``<Handler>._add_next_turn_user_message_*`` to keep ``bfcl.turn_idx`` in
    sync.  No spans are created here.
    """

    def __init__(self, *, reset: bool) -> None:
        self._reset = reset

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        try:
            if self._reset:
                # ``add_first_turn_message_*`` runs once at the very start of
                # multi-turn / single-turn inference.  We only want to reset
                # to ``turn_idx=0`` here.
                from opentelemetry.instrumentation.bfclv4.internal.state import (
                    get_state,
                )

                state = get_state()
                if state is not None:
                    state["turn_idx"] = 0
                    state["fc_round"] = 0
            else:
                bump_turn()
        except Exception:  # noqa: BLE001
            logger.debug(
                "bfclv4: turn_idx maintenance failed", exc_info=True
            )
        return wrapped(*args, **kwargs)


# ---------------------------------------------------------------------------
# TOOL wrapper


class ExecuteFuncCallWrapper:
    """Wraps
    ``bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.execute_multi_turn_func_call``.

    BFCL evaluates a list of function-call strings in a single Python call;
    we surface each one as its own TOOL span by post-processing the wrapped
    result.  Per-call latency is approximated by averaging the total elapsed
    time across the batch (``bfcl.tool.duration_is_estimated=true``).
    """

    def __init__(self, helper: GenAIHookHelper) -> None:
        self._helper = helper

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        # ``execute_multi_turn_func_call(func_call_list, initial_config,
        #                                involved_classes, model_name,
        #                                test_entry_id, long_context=False,
        #                                is_evaL_run=False)``
        func_call_list = (
            args[0] if args else kwargs.get("func_call_list", [])
        )
        model_name = (
            args[3]
            if len(args) >= 4
            else kwargs.get("model_name")
        )
        test_entry_id = (
            args[4]
            if len(args) >= 5
            else kwargs.get("test_entry_id")
        )

        if not isinstance(func_call_list, list) or not func_call_list:
            return wrapped(*args, **kwargs)

        t0 = time.perf_counter()
        try:
            result = wrapped(*args, **kwargs)
        finally:
            elapsed = max(time.perf_counter() - t0, 0.0)

        execution_results: List[str] = []
        if isinstance(result, tuple) and result:
            payload = result[0]
            if isinstance(payload, list):
                execution_results = list(payload)

        per_call_seconds = (
            elapsed / len(func_call_list) if func_call_list else 0.0
        )

        handler_obj = get_extended_telemetry_handler()
        for index, func_call in enumerate(func_call_list):
            tool_name = _extract_tool_name(func_call)
            arguments = _extract_tool_arguments(func_call)
            execution_result = (
                execution_results[index]
                if index < len(execution_results)
                else None
            )

            tool_inv = ExecuteToolInvocation(
                tool_name=tool_name,
                tool_call_id=_synth_tool_call_id(
                    test_entry_id, model_name, index
                ),
                tool_type="function",
                tool_call_arguments=arguments,
                tool_call_result=execution_result,
            )

            try:
                with handler_obj.execute_tool(tool_inv) as inv:
                    span = inv.span
                    if span is not None and span.is_recording():
                        span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
                        span.set_attribute(BFCL_TOOL_INDEX, index)
                        span.set_attribute(
                            BFCL_TOOL_DURATION_IS_ESTIMATED, True
                        )
                        if test_entry_id is not None:
                            span.set_attribute(
                                BFCL_TEST_ENTRY_ID, str(test_entry_id)
                            )
                        if isinstance(execution_result, str) and execution_result.startswith(
                            "Error during execution:"
                        ):
                            try:
                                from opentelemetry.trace import (
                                    Status,
                                    StatusCode,
                                )

                                span.set_status(
                                    Status(
                                        StatusCode.ERROR,
                                        execution_result[:200],
                                    )
                                )
                            except Exception:  # noqa: BLE001
                                pass
                        # Approximate latency by sleeping the budgeted slice
                        # would distort BFCL execution; we instead rely on
                        # span start/end (currently both wall-clock-now).
                        # The ``bfcl.tool.duration_is_estimated`` attribute
                        # signals the limitation to consumers.
                        _ = per_call_seconds  # unused but documented
                # Bump a per-AGENT counter for downstream debugging.
                next_tool_index()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "bfclv4 TOOL: span emission failed for %s",
                    tool_name,
                    exc_info=True,
                )

        return result


def _extract_tool_name(func_call: Any) -> str:
    if not isinstance(func_call, str) or "(" not in func_call:
        return "unknown"
    head = func_call.split("(", 1)[0]
    # ``head`` may be ``module.method`` or ``instance.method`` - keep the
    # last segment which is the actual callable.
    return head.split(".")[-1] or "unknown"


def _extract_tool_arguments(func_call: Any) -> Optional[str]:
    if not isinstance(func_call, str):
        return None
    if "(" not in func_call or not func_call.endswith(")"):
        return func_call
    args_part = func_call[func_call.index("(") + 1 : -1]
    return args_part if args_part else None


def _synth_tool_call_id(
    test_entry_id: Optional[Any], model_name: Optional[Any], index: int
) -> str:
    parts = [
        str(test_entry_id) if test_entry_id is not None else "no_id",
        str(model_name) if model_name is not None else "no_model",
        str(index),
    ]
    return "-".join(parts)
