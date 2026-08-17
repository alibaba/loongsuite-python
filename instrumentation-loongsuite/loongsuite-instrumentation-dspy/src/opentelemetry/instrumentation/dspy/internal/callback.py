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
LoongSuite DSPy callback — framework span emission.

Implements ``dspy.utils.callback.BaseCallback`` and maps DSPy's native
``on_module_*`` / ``on_tool_*`` / ``on_evaluate_*`` hooks onto ``util-genai``
invocations:

* outermost module call → ENTRY
* ``ReAct`` (and subclasses) → AGENT
* ``Retrieve`` / retriever modules → RETRIEVER
* every other module / ``Evaluate`` → CHAIN
* ``dspy.Tool`` → TOOL

``on_lm_*`` is deliberately **not** implemented: LLM and EMBEDDING spans (and
all token metrics) come from ``loongsuite-instrumentation-litellm``, which
wraps the LiteLLM entry points DSPy funnels every model call through. Every
span created here is therefore attached to the OpenTelemetry context so those
LLM spans nest underneath instead of starting a detached trace.
"""

from __future__ import annotations

import contextvars
import logging
import random
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import dspy
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.dspy.internal._utils import (
    aggregate_lm_usage,
    build_input_messages,
    build_output_messages,
    build_retrieval_documents,
    extract_query,
    extract_tool_definitions,
    extract_top_k,
    normalize_callback_inputs,
    resolve_request_model,
    safe_json,
    should_capture_content,
)
from opentelemetry.instrumentation.dspy.internal.config import (
    entry_span_enabled,
    root_sample_ratio,
)
from opentelemetry.instrumentation.dspy.internal.semconv import (
    CHAIN_OPERATION_TASK,
    CHAIN_OPERATION_WORKFLOW,
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    INPUT_VALUE,
    OUTPUT_VALUE,
    SPAN_KIND_CHAIN,
)
from opentelemetry.instrumentation.dspy.version import __version__
from opentelemetry.trace import (
    Span,
    SpanKind,
    StatusCode,
    get_tracer,
    set_span_in_context,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
    RetrievalInvocation,
)
from opentelemetry.util.genai.handler import _safe_detach
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)

# Set on the outermost DSPy call that sampling dropped; every nested callback
# checks it so a dropped program never emits a partial span tree.
_SAMPLING_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "loongsuite_dspy_sampling_suppressed", default=False
)


@dataclass
class _CallData:
    """Per ``call_id`` bookkeeping.

    ``on_*_end`` only receives ``(call_id, outputs, exception)``, so the
    instance, the span and the context token have to be cached at start time
    and looked up again on end.
    """

    kind: str
    invocation: Any = None
    span: Span | None = None
    context_token: object | None = None
    entry: EntryInvocation | None = None
    suppress_token: object | None = None
    # AGENT only: ReAct step state
    react_round: int = 0
    active_step: ReactStepInvocation | None = None
    last_tool_name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


def _is_agent(instance: Any) -> bool:
    react_cls = getattr(dspy, "ReAct", None)
    return react_cls is not None and isinstance(instance, react_cls)


def _is_retriever(instance: Any) -> bool:
    retrieve_cls = getattr(dspy, "Retrieve", None)
    return retrieve_cls is not None and isinstance(instance, retrieve_cls)


def _chain_operation_name(instance: Any) -> str:
    predict_cls = getattr(dspy, "Predict", None)
    if predict_cls is not None and isinstance(instance, predict_cls):
        return CHAIN_OPERATION_TASK
    return CHAIN_OPERATION_WORKFLOW


class OTelDSPyCallback(BaseCallback):
    """Emits DSPy framework spans through ``ExtendedTelemetryHandler``."""

    def __init__(
        self,
        handler: ExtendedTelemetryHandler,
        tracer_provider: Any = None,
    ) -> None:
        self._handler = handler
        self._tracer = get_tracer(
            __name__, __version__, tracer_provider=tracer_provider
        )
        self._calls: dict[str, _CallData] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _put(self, call_id: str, data: _CallData) -> None:
        with self._lock:
            self._calls[call_id] = data

    def _pop(self, call_id: str) -> _CallData | None:
        with self._lock:
            return self._calls.pop(call_id, None)

    def _get(self, call_id: str) -> _CallData | None:
        with self._lock:
            return self._calls.get(call_id)

    @staticmethod
    def _framework_attributes(with_model: bool = False) -> dict[str, Any]:
        """Attributes every DSPy span carries.

        ``with_model`` is for span types whose invocation has no typed
        ``request_model`` field (ENTRY, CHAIN).
        """
        attributes: dict[str, Any] = {GEN_AI_FRAMEWORK: FRAMEWORK_NAME}
        if with_model:
            model = resolve_request_model()
            if model:
                attributes[GEN_AI_REQUEST_MODEL] = model
        return attributes

    # ------------------------------------------------------------------
    # Module hooks — ENTRY / AGENT / CHAIN / RETRIEVER
    # ------------------------------------------------------------------

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        try:
            self._start_module(call_id, instance, inputs)
        except Exception:
            logger.debug("Failed to start DSPy module span", exc_info=True)

    def _start_module(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        if _SAMPLING_SUPPRESSED.get():
            return

        is_root = ACTIVE_CALL_ID.get() is None
        if is_root and random.random() >= root_sample_ratio():
            self._put(
                call_id,
                _CallData(
                    kind="suppressed",
                    suppress_token=_SAMPLING_SUPPRESSED.set(True),
                ),
            )
            return

        normalized_inputs = normalize_callback_inputs(inputs)
        entry = None
        if is_root and entry_span_enabled():
            entry = EntryInvocation(
                input_messages=build_input_messages(normalized_inputs),
                attributes=self._framework_attributes(with_model=True),
            )
            self._handler.start_entry(entry)

        if _is_agent(instance):
            data = self._start_agent(instance, normalized_inputs)
        elif _is_retriever(instance):
            data = self._start_retrieval(instance, normalized_inputs)
        else:
            data = self._start_chain(instance, normalized_inputs)

        data.entry = entry
        self._put(call_id, data)

    def _start_agent(self, instance: Any, inputs: Any) -> _CallData:
        invocation = InvokeAgentInvocation(
            provider=FRAMEWORK_NAME,
            agent_name=type(instance).__name__,
            input_messages=build_input_messages(inputs),
            tool_definitions=extract_tool_definitions(instance),
            request_model=resolve_request_model(),
            attributes=self._framework_attributes(),
        )
        self._handler.start_invoke_agent(invocation)
        return _CallData(
            kind="agent", invocation=invocation, span=invocation.span
        )

    def _start_retrieval(self, instance: Any, inputs: Any) -> _CallData:
        invocation = RetrievalInvocation(
            query=extract_query(inputs),
            top_k=extract_top_k(inputs, instance),
            data_source_id=type(instance).__name__,
            request_model=resolve_request_model(),
            attributes=self._framework_attributes(),
        )
        self._handler.start_retrieval(invocation)
        return _CallData(
            kind="retriever", invocation=invocation, span=invocation.span
        )

    def _start_chain(self, instance: Any, inputs: Any) -> _CallData:
        return self._start_chain_span(
            name=type(instance).__name__,
            operation_name=_chain_operation_name(instance),
            inputs=inputs,
        )

    def _start_chain_span(
        self, name: str, operation_name: str, inputs: Any
    ) -> _CallData:
        """Create a CHAIN span directly: ``util-genai`` has no chain operation."""
        span = self._tracer.start_span(
            name=f"chain {name}",
            kind=SpanKind.INTERNAL,
        )
        attributes = self._framework_attributes(with_model=True)
        attributes[GEN_AI_SPAN_KIND] = SPAN_KIND_CHAIN
        attributes[GEN_AI_OPERATION_NAME] = operation_name
        if inputs is not None and should_capture_content():
            attributes[INPUT_VALUE] = safe_json(inputs)
        span.set_attributes(attributes)

        # Attach so that the LLM spans created by the litellm instrumentation
        # deeper in the DSPy call stack become children of this chain.
        token = otel_context.attach(
            set_span_in_context(span, otel_context.get_current())
        )
        return _CallData(kind="chain", span=span, context_token=token)

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        data = self._pop(call_id)
        if data is None:
            return
        try:
            self._end_call(data, outputs, exception)
        except Exception:
            logger.debug("Failed to stop DSPy module span", exc_info=True)

    def _end_call(
        self, data: _CallData, outputs: Any, exception: Exception | None
    ) -> None:
        if data.kind == "suppressed":
            if data.suppress_token is not None:
                _SAMPLING_SUPPRESSED.reset(data.suppress_token)
            return
        try:
            if data.kind == "agent":
                self._stop_agent(data, outputs, exception)
            elif data.kind == "retriever":
                self._stop_retrieval(data, outputs, exception)
            else:
                self._stop_chain(data, outputs, exception)
        finally:
            self._stop_entry(data, outputs, exception)

    def _stop_agent(
        self, data: _CallData, outputs: Any, exception: Exception | None
    ) -> None:
        if data.active_step is not None:
            self._close_active_step(data, exception)

        invocation: InvokeAgentInvocation = data.invocation
        usage = aggregate_lm_usage(outputs)
        if usage:
            # Written as raw attributes on purpose: the aggregate double-counts
            # the per-call usage the litellm LLM spans already report, so it
            # must never reach the token histogram (only the AGENT span).
            if "prompt_tokens" in usage:
                invocation.attributes[GEN_AI_USAGE_INPUT_TOKENS] = usage[
                    "prompt_tokens"
                ]
            if "completion_tokens" in usage:
                invocation.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = usage[
                    "completion_tokens"
                ]
            if "total_tokens" in usage:
                invocation.attributes[GEN_AI_USAGE_TOTAL_TOKENS] = usage[
                    "total_tokens"
                ]

        if exception is not None:
            self._handler.fail_invoke_agent(invocation, _to_error(exception))
            return
        invocation.output_messages = build_output_messages(outputs)
        self._handler.stop_invoke_agent(invocation)

    def _stop_retrieval(
        self, data: _CallData, outputs: Any, exception: Exception | None
    ) -> None:
        invocation: RetrievalInvocation = data.invocation
        if exception is not None:
            self._handler.fail_retrieval(invocation, _to_error(exception))
            return
        invocation.documents = build_retrieval_documents(outputs)
        self._handler.stop_retrieval(invocation)

    def _stop_chain(
        self, data: _CallData, outputs: Any, exception: Exception | None
    ) -> None:
        span = data.span
        if span is None:
            _safe_detach(data.context_token)
            return
        try:
            if exception is not None:
                span.set_status(StatusCode.ERROR, str(exception))
                span.record_exception(exception)
            elif outputs is not None and should_capture_content():
                span.set_attribute(OUTPUT_VALUE, safe_json(outputs))
        finally:
            _safe_detach(data.context_token)
            span.end()

    def _stop_entry(
        self, data: _CallData, outputs: Any, exception: Exception | None
    ) -> None:
        entry = data.entry
        if entry is None:
            return
        data.entry = None
        if exception is not None:
            self._handler.fail_entry(entry, _to_error(exception))
            return
        entry.output_messages = build_output_messages(outputs)
        self._handler.stop_entry(entry)

    # ------------------------------------------------------------------
    # Tool hooks — TOOL
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        if _SAMPLING_SUPPRESSED.get():
            return
        try:
            invocation = ExecuteToolInvocation(
                tool_name=getattr(instance, "name", None)
                or type(instance).__name__,
                tool_call_id=call_id,
                tool_description=getattr(instance, "desc", None),
                tool_type="function",
                tool_call_arguments=normalize_callback_inputs(inputs),
                attributes={GEN_AI_FRAMEWORK: FRAMEWORK_NAME},
            )
            self._handler.start_execute_tool(invocation)
            self._put(
                call_id,
                _CallData(
                    kind="tool", invocation=invocation, span=invocation.span
                ),
            )
        except Exception:
            logger.debug("Failed to start DSPy tool span", exc_info=True)

    def on_tool_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        data = self._pop(call_id)
        if data is None or data.kind != "tool":
            return
        try:
            invocation: ExecuteToolInvocation = data.invocation
            if exception is not None:
                self._handler.fail_execute_tool(
                    invocation, _to_error(exception)
                )
                return
            invocation.tool_call_result = outputs
            self._handler.stop_execute_tool(invocation)
        except Exception:
            logger.debug("Failed to stop DSPy tool span", exc_info=True)

    # ------------------------------------------------------------------
    # Evaluate hooks — CHAIN
    # ------------------------------------------------------------------

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        if _SAMPLING_SUPPRESSED.get():
            return
        try:
            self._put(
                call_id,
                self._start_chain_span(
                    name=type(instance).__name__,
                    operation_name=CHAIN_OPERATION_WORKFLOW,
                    inputs=None,
                ),
            )
        except Exception:
            logger.debug("Failed to start DSPy evaluate span", exc_info=True)

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        data = self._pop(call_id)
        if data is None or data.kind != "chain":
            return
        try:
            self._stop_chain(data, outputs, exception)
        except Exception:
            logger.debug("Failed to stop DSPy evaluate span", exc_info=True)

    # ------------------------------------------------------------------
    # ReAct STEP — driven by the ReAct patch
    # ------------------------------------------------------------------

    def current_agent_data(self) -> _CallData | None:
        """Return the AGENT state of the enclosing ``ReAct`` call, if any.

        Inside ``ReAct.forward`` DSPy has already set ``ACTIVE_CALL_ID`` to the
        ``call_id`` of the ``ReAct.__call__`` that opened the AGENT span.
        """
        call_id = ACTIVE_CALL_ID.get()
        if call_id is None:
            return None
        data = self._get(call_id)
        if data is None or data.kind != "agent":
            return None
        return data

    def enter_react_step(self, data: _CallData) -> None:
        """Open a STEP span for the next ReAct round."""
        if data.active_step is not None:
            self.exit_react_step(data)

        data.react_round += 1
        data.last_tool_name = None
        invocation = ReactStepInvocation(
            round=data.react_round,
            attributes={GEN_AI_FRAMEWORK: FRAMEWORK_NAME},
        )
        self._handler.start_react_step(invocation)
        data.active_step = invocation

    def exit_react_step(self, data: _CallData) -> None:
        """Close the active STEP span successfully."""
        invocation = data.active_step
        if invocation is None:
            return
        data.active_step = None
        invocation.finish_reason = (
            "finish" if data.last_tool_name == "finish" else "tool_calls"
        )
        self._handler.stop_react_step(invocation)

    def fail_react_step(
        self, data: _CallData, exception: BaseException
    ) -> None:
        """Close the active STEP span with an error status."""
        invocation = data.active_step
        if invocation is None:
            return
        data.active_step = None
        invocation.finish_reason = "error"
        self._handler.fail_react_step(invocation, _to_error(exception))

    def _close_active_step(
        self, data: _CallData, exception: Exception | None
    ) -> None:
        if exception is not None:
            self.fail_react_step(data, exception)
        else:
            self.exit_react_step(data)


def _to_error(exception: BaseException) -> Error:
    return Error(message=str(exception), type=type(exception))
