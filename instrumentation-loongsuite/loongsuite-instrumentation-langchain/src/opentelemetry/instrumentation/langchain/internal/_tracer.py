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
LoongSuite LangChain Tracer — data extraction phase.

Extends ``langchain_core.tracers.base.BaseTracer`` and overrides the
fine-grained ``_on_*`` hooks to extract telemetry data from LangChain
``Run`` objects and emit OpenTelemetry spans via ``util-genai``.

Context propagation follows the Robin/OpenLLMetry pattern: parent-child
span relationships are established by passing ``context`` explicitly to
``start_span`` / ``handler.start_*``, rather than using hazardous
``context_api.attach`` / ``detach`` in a callback system.

The only exception is Chain spans: they use ``attach``/``detach`` so that
non-LangChain child operations (e.g. HTTP calls) nest correctly.

Run type → handler mapping
--------------------------
* **LLM / chat_model** → ``handler.start_llm`` / ``stop_llm`` / ``fail_llm``
* **Chain (Agent)**     → ``handler.start_invoke_agent`` / …
* **Chain (generic)**   → direct span creation (no ``util-genai``)
* **Tool**              → ``handler.start_execute_tool`` / …
* **Retriever**         → ``handler.start_retrieval`` / …
"""

from __future__ import annotations

import logging
import timeit
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal, Optional
from uuid import UUID
from weakref import WeakSet

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.instrumentation.langchain.internal._utils import (
    LANGGRAPH_REACT_STEP_NODE,
    _documents_to_retrieval_documents,
    _extract_finish_reasons,
    _extract_invocation_params,
    _extract_llm_input_messages,
    _extract_llm_output_messages,
    _extract_model_name,
    _extract_provider,
    _extract_response_model,
    _extract_token_usage,
    _extract_tool_definitions,
    _has_langgraph_react_metadata,
    _is_agent_run,
    _safe_json,
)
from opentelemetry.instrumentation.langchain.internal.semconv import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_SPAN_KIND,
    INPUT_VALUE,
    OUTPUT_VALUE,
)
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.trace import (
    Span,
    SpanKind,
    StatusCode,
    get_current_span,
    get_tracer,
    set_span_in_context,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
    RetrievalInvocation,
)
from opentelemetry.util.genai.handler import _safe_detach
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
    is_experimental_mode,
)

logger = logging.getLogger(__name__)
_TRACER_INSTANCES: WeakSet["LoongsuiteTracer"] = WeakSet()
_DEEPAGENTS_FRAMEWORK_NAME = "deepagents"
_DEEPAGENTS_METADATA_INTEGRATION = "ls_integration"
_DEEPAGENTS_METADATA_VERSIONS = "versions"
_DEEPAGENTS_METADATA_AGENT_NAME = "lc_agent_name"
_DEEPAGENTS_METADATA_AGENT_TYPE = "ls_agent_type"
_DEEPAGENTS_METADATA_SUBAGENT_DESCRIPTION = (
    "loongsuite_deepagents_subagent_description"
)
_DEEPAGENTS_SUBAGENT_TYPE = "subagent"
_GEN_AI_FRAMEWORK = "gen_ai.framework"
_GEN_AI_FRAMEWORK_VERSION = "gen_ai.framework.version"
_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
_GEN_AI_AGENT_TYPE = "gen_ai.agent.type"
_GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"

# ---------------------------------------------------------------------------
# _RunData — per-run bookkeeping
# ---------------------------------------------------------------------------
RunKind = Literal["llm", "agent", "chain", "tool", "retriever", "react_step"]


@dataclass
class _RunData:
    run_kind: RunKind
    span: Span | None = None
    context: Context | None = None
    invocation: Any = None
    parent_run_id: UUID | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    context_token: object | None = None  # only used for Chain attach/detach
    # Agent run only: ReAct Step state
    react_round: int = 0
    active_step: "_RunData | None" = None
    original_context: Context | None = None
    is_langgraph_react: bool = False
    inside_langgraph_react: bool = False


def _should_capture_chain_content() -> bool:
    """Check if chain input/output content should be recorded."""
    try:
        if not is_experimental_mode():
            return False
        return get_content_capturing_mode() in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    except ValueError:
        logger.debug(
            "Content capturing mode check failed (experimental mode or mode value)",
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# LoongsuiteTracer
# ---------------------------------------------------------------------------


class LoongsuiteTracer(BaseTracer):
    """LangChain tracer that emits OpenTelemetry spans via util-genai.

    Context propagation is done explicitly — parent-child relationships
    are established by passing the stored ``Context`` of the parent run
    to ``handler.start_*(…, context=parent_ctx)`` or to
    ``tracer.start_span(…, context=parent_ctx)``.

    Chain spans are the sole exception: they ``attach``/``detach`` the
    context so that non-LangChain child operations nest correctly.

    All access to ``self._runs`` is protected by an ``RLock`` because
    LangChain callbacks may be fired from different threads.
    """

    def __init__(
        self,
        handler: ExtendedTelemetryHandler,
        tracer_provider: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(_schema_format="original+chat", **kwargs)
        # We need run callback inline so that propagate the context correctly.
        self.run_inline = True
        self._handler = handler
        self._tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider=tracer_provider,
        )
        self._runs: dict[UUID, _RunData] = {}
        self._lock = RLock()
        # Don't use super().run_map because it will lead to unexpected behavior when multiple tracers are used.
        self.run_map = dict(self.run_map)
        self.run_map_lock = RLock()
        _TRACER_INSTANCES.add(self)

    def _persist_run(self, run: Run) -> None:
        pass

    # ------------------------------------------------------------------
    # Context helper
    # ------------------------------------------------------------------

    def _get_parent_context(self, run: Run) -> Context | None:
        """Return the stored context of the parent run, or *None*."""
        task_tool_ctx = self._get_deepagents_subagent_task_context(run)
        if task_tool_ctx is not None:
            return task_tool_ctx

        parent_id = getattr(run, "parent_run_id", None)
        if parent_id:
            with self._lock:
                rd = self._runs.get(parent_id)
            if rd is not None:
                return rd.context
        return None

    def get_otel_span_for_run(self, run_id: UUID | str) -> Span | None:
        """Return the active OTel span created for a LangChain run."""
        normalized_run_id = _normalize_run_id(run_id)
        if normalized_run_id is None:
            return None
        with self._lock:
            rd = self._runs.get(normalized_run_id)
        return rd.span if rd is not None else None

    def _get_deepagents_subagent_task_context(
        self, run: Run
    ) -> Context | None:
        if not _is_deepagents_subagent_candidate(run):
            return None

        current_span = get_current_span()
        if _is_task_tool_span(current_span):
            return set_span_in_context(current_span)

        parent_id = getattr(run, "parent_run_id", None)
        with self._lock:
            run_data = list(self._runs.values())
        latest_task_context: Context | None = None
        for rd in reversed(run_data):
            if rd.run_kind != "tool" or rd.tool_name != "task":
                continue
            if latest_task_context is None:
                latest_task_context = rd.context
            if parent_id is not None and rd.parent_run_id != parent_id:
                continue
            if rd.context is not None:
                return rd.context
        return latest_task_context

    # ------------------------------------------------------------------
    # _start_trace / _end_trace
    # ------------------------------------------------------------------
    # We maintain only run_map (required for _complete_* / _errored_* to find
    # the run). We do NOT call super() to avoid parent's order_map accumulation
    # and unexpected behavior when multiple tracers are used.

    def _start_trace(self, run: Run) -> None:
        with self.run_map_lock:
            self.run_map[str(run.id)] = run

    def _end_trace(self, run: Run) -> None:
        with self.run_map_lock:
            self.run_map.pop(str(run.id), None)

    # ------------------------------------------------------------------
    # TTFT (Time To First Token) — streaming support
    # ------------------------------------------------------------------

    def on_llm_new_token(  # type: ignore[override]
        self,
        token: str,
        *,
        chunk: Optional[Any] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Run | None:
        """Record the first-token timestamp for TTFT calculation."""
        with self._lock:
            rd = self._runs.get(run_id)
        if (
            rd is not None
            and rd.run_kind == "llm"
            and rd.invocation is not None
        ):
            inv: LLMInvocation = rd.invocation
            if inv.monotonic_first_token_s is None:
                inv.monotonic_first_token_s = timeit.default_timer()
        return None

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    def _on_llm_start(self, run: Run) -> None:
        self._handle_llm_start(run)

    def _on_chat_model_start(self, run: Run) -> None:
        self._handle_llm_start(run)

    def _handle_llm_start(self, run: Run) -> None:
        try:
            parent_ctx = self._get_parent_context(run)
            params = _extract_invocation_params(run)
            invocation = LLMInvocation(
                request_model=_extract_model_name(run) or run.name or "",
                provider=_extract_provider(run),
                input_messages=_extract_llm_input_messages(run),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_tokens")
                or params.get("max_output_tokens"),
            )
            tool_defs = _extract_tool_definitions(run)
            if tool_defs:
                invocation.tool_definitions = tool_defs
            self._handler.start_llm(invocation, context=parent_ctx)
            rd = _RunData(
                run_kind="llm",
                span=invocation.span,
                context=set_span_in_context(invocation.span)
                if invocation.span
                else None,
                invocation=invocation,
                parent_run_id=getattr(run, "parent_run_id", None),
            )
            with self._lock:
                self._runs[run.id] = rd
        except Exception:
            logger.debug("Failed to start LLM span", exc_info=True)

    def _on_llm_end(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "llm":
            return
        try:
            inv: LLMInvocation = rd.invocation
            inv.output_messages = _extract_llm_output_messages(run)
            inv.input_tokens, inv.output_tokens = _extract_token_usage(run)
            inv.finish_reasons = _extract_finish_reasons(run)
            inv.response_model_name = _extract_response_model(run)
            self._handler.stop_llm(inv)
        except Exception:
            logger.debug("Failed to stop LLM span", exc_info=True)

    def _on_llm_error(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "llm":
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            self._handler.fail_llm(
                rd.invocation,
                Error(message=str(err_str), type=Exception),
            )
        except Exception:
            logger.debug("Failed to fail LLM span", exc_info=True)

    # ------------------------------------------------------------------
    # Chain / Agent hooks
    # ------------------------------------------------------------------

    def _on_chain_start(self, run: Run) -> None:
        try:
            if _is_agent_run(run):
                self._start_agent(run)
            elif _has_langgraph_react_metadata(run):
                self._handle_langgraph_chain_start(run)
            else:
                self._start_chain(run)
        except Exception:
            logger.debug("Failed to start Chain/Agent span", exc_info=True)

    def _handle_langgraph_chain_start(self, run: Run) -> None:
        """Route a chain start that carries LangGraph ReAct metadata.

        Because ``config["metadata"]`` propagates to child callbacks,
        both the graph-level run and its child nodes carry the flag.
        We disambiguate by checking whether any ancestor is a LangGraph
        ReAct agent (``is_langgraph_react``) or inside one
        (``inside_langgraph_react``):

        * **Inside LangGraph agent** → child node (chain span, with
          possible ReAct step transition).
        * **Otherwise** → top-level graph → create Agent span.
        """
        parent_id = getattr(run, "parent_run_id", None)
        if _is_deepagents_subagent_root(run):
            self._start_agent(run)
            return

        with self._lock:
            parent_rd = self._runs.get(parent_id) if parent_id else None

        inside = parent_rd is not None and (
            parent_rd.is_langgraph_react or parent_rd.inside_langgraph_react
        )

        if inside:
            self._maybe_enter_langgraph_react_step(run)
            self._start_chain(run)
        else:
            self._start_agent(run)

    def _resolve_langgraph_agent_name(self, run: Run) -> str:
        """Pick a meaningful agent name for a LangGraph ReAct agent.

        When the inner graph uses the default name ``"LangGraph"`` and is
        invoked as a node inside an outer graph, the parent node's name
        (e.g. ``"product_agent"``) is far more descriptive.  We prefer it
        over the generic default.
        """
        name = run.name or ""
        if not _has_langgraph_react_metadata(run) or name != "LangGraph":
            return name

        parent_id = getattr(run, "parent_run_id", None)
        if not parent_id:
            return name

        with self._lock:
            parent_rd = self._runs.get(parent_id)
        if parent_rd is None or parent_rd.run_kind != "chain":
            return name

        span = parent_rd.span
        if span is None:
            return name

        parent_span_name: str = span.name
        if parent_span_name.startswith("chain "):
            return parent_span_name[len("chain ") :]
        return name

    def _start_agent(self, run: Run) -> None:
        parent_ctx = self._get_parent_context(run)
        inputs = getattr(run, "inputs", None) or {}
        input_messages: list[InputMessage] = []

        # AgentExecutor format: {"input": "...", "query": "..."}
        input_val = inputs.get("input") or inputs.get("query") or ""
        if isinstance(input_val, str) and input_val:
            input_messages.append(
                InputMessage(role="user", parts=[Text(content=input_val)])
            )

        # LangGraph format: {"messages": [...]}
        if not input_messages:
            raw_messages = inputs.get("messages")
            if raw_messages and isinstance(raw_messages, list):
                for msg in raw_messages:
                    converted = _extract_langgraph_input_message(msg)
                    if converted:
                        input_messages.append(converted)

        agent_name = self._resolve_langgraph_agent_name(run)

        invocation = InvokeAgentInvocation(
            provider="langchain",
            agent_name=agent_name,
            input_messages=input_messages,
        )
        self._handler.start_invoke_agent(invocation, context=parent_ctx)
        _enrich_deepagents_agent_span(invocation.span, run)
        rd = _RunData(
            run_kind="agent",
            span=invocation.span,
            context=set_span_in_context(invocation.span)
            if invocation.span
            else None,
            invocation=invocation,
            parent_run_id=getattr(run, "parent_run_id", None),
            is_langgraph_react=_has_langgraph_react_metadata(run),
        )
        with self._lock:
            self._runs[run.id] = rd

    def _start_chain(self, run: Run) -> None:
        parent_ctx = self._get_parent_context(run)
        span = self._tracer.start_span(
            name=f"chain {run.name}",
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )

        span.set_attribute(GEN_AI_OPERATION_NAME, "chain")
        span.set_attribute(GEN_AI_SPAN_KIND, "CHAIN")
        if _should_capture_chain_content():
            inputs = getattr(run, "inputs", None) or {}
            span.set_attribute(INPUT_VALUE, _safe_json(inputs))

        # Attach chain span context so non-LangChain children nest correctly.
        ctx = set_span_in_context(span)
        token = otel_context.attach(ctx)

        # Propagate inside_langgraph_react from parent so that
        # grandchildren of the graph are also recognised as internal.
        inside_lg = False
        parent_id = getattr(run, "parent_run_id", None)
        if parent_id:
            with self._lock:
                p = self._runs.get(parent_id)
            if p is not None:
                inside_lg = p.is_langgraph_react or p.inside_langgraph_react

        rd = _RunData(
            run_kind="chain",
            span=span,
            context=ctx,
            context_token=token,
            parent_run_id=parent_id,
            inside_langgraph_react=inside_lg,
        )
        with self._lock:
            self._runs[run.id] = rd

    def _on_chain_end(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None:
            return
        try:
            if rd.run_kind == "agent":
                self._stop_agent(run, rd)
            elif rd.run_kind == "chain":
                self._stop_chain(run, rd)
        except Exception:
            logger.debug("Failed to stop Chain/Agent span", exc_info=True)

    def _stop_agent(self, run: Run, rd: _RunData) -> None:
        # End last ReAct step if still active.
        # Cannot use _exit_react_step here because rd has already been
        # popped from self._runs by _on_chain_end.
        if rd.active_step is not None:
            step_inv: ReactStepInvocation = rd.active_step.invocation
            step_inv.finish_reason = "stop"
            self._handler.stop_react_step(step_inv)
            rd.active_step = None

        inv: InvokeAgentInvocation = rd.invocation
        outputs = getattr(run, "outputs", None) or {}

        # AgentExecutor format
        output_val = outputs.get("output") or outputs.get("result") or ""
        if isinstance(output_val, str) and output_val:
            inv.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=output_val)],
                    finish_reason="stop",
                )
            ]

        # LangGraph format: {"messages": [...]}
        if not inv.output_messages:
            raw_messages = outputs.get("messages")
            if raw_messages and isinstance(raw_messages, list):
                last_msg = raw_messages[-1]
                content = _extract_message_content(last_msg)
                if content:
                    inv.output_messages = [
                        OutputMessage(
                            role="assistant",
                            parts=[Text(content=content)],
                            finish_reason="stop",
                        )
                    ]

        self._handler.stop_invoke_agent(inv)

    def _stop_chain(self, run: Run, rd: _RunData) -> None:
        span = rd.span
        if span is None:
            return
        if _should_capture_chain_content():
            outputs = getattr(run, "outputs", None) or {}
            span.set_attribute(OUTPUT_VALUE, _safe_json(outputs))
        span.end()
        _safe_detach(rd.context_token)

    def _on_chain_error(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None:
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            if rd.run_kind == "agent":
                # Fail active step directly (rd already popped from _runs).
                if rd.active_step is not None:
                    step_inv = rd.active_step.invocation
                    self._handler.fail_react_step(
                        step_inv,
                        Error(message=str(err_str), type=Exception),
                    )
                    rd.active_step = None
                self._handler.fail_invoke_agent(
                    rd.invocation,
                    Error(message=str(err_str), type=Exception),
                )
            elif rd.run_kind == "chain":
                span = rd.span
                if span is not None:
                    span.set_status(StatusCode.ERROR, str(err_str))
                    span.record_exception(Exception(str(err_str)))
                    span.end()
                _safe_detach(rd.context_token)
        except Exception:
            logger.debug("Failed to fail Chain/Agent span", exc_info=True)

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    def _on_tool_start(self, run: Run) -> None:
        try:
            parent_ctx = self._get_parent_context(run)
            inputs = getattr(run, "inputs", None) or {}
            input_str = inputs.get("input") or inputs.get("query") or ""
            if not isinstance(input_str, str):
                input_str = _safe_json(input_str)
            extra = getattr(run, "extra", None) or {}
            tool_call_id = extra.get("tool_call_id")
            invocation = ExecuteToolInvocation(
                tool_name=run.name or "unknown_tool",
                tool_call_arguments=input_str,
                tool_call_id=tool_call_id,
            )
            self._handler.start_execute_tool(invocation, context=parent_ctx)
            rd = _RunData(
                run_kind="tool",
                span=invocation.span,
                context=set_span_in_context(invocation.span)
                if invocation.span
                else None,
                invocation=invocation,
                parent_run_id=getattr(run, "parent_run_id", None),
                tool_name=run.name,
                tool_call_id=tool_call_id,
            )
            with self._lock:
                self._runs[run.id] = rd
        except Exception:
            logger.debug("Failed to start Tool span", exc_info=True)

    def _on_tool_end(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "tool":
            return
        try:
            inv: ExecuteToolInvocation = rd.invocation
            outputs = getattr(run, "outputs", None) or {}
            output = outputs.get("output") or ""
            if hasattr(output, "content"):
                # Extract content from ToolMessage instance
                output = output.content
            if not isinstance(output, str):
                output = _safe_json(output)
            inv.tool_call_result = output
            self._handler.stop_execute_tool(inv)
        except Exception:
            logger.debug("Failed to stop Tool span", exc_info=True)

    def _on_tool_error(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "tool":
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            self._handler.fail_execute_tool(
                rd.invocation,
                Error(message=str(err_str), type=Exception),
            )
        except Exception:
            logger.debug("Failed to fail Tool span", exc_info=True)

    # ------------------------------------------------------------------
    # Retriever hooks
    # ------------------------------------------------------------------

    def _on_retriever_start(self, run: Run) -> None:
        try:
            parent_ctx = self._get_parent_context(run)
            inputs = getattr(run, "inputs", None) or {}
            query = inputs.get("query") or ""

            invocation = RetrievalInvocation(query=query)
            self._handler.start_retrieval(invocation, context=parent_ctx)
            rd = _RunData(
                run_kind="retriever",
                span=invocation.span,
                context=set_span_in_context(invocation.span)
                if invocation.span
                else None,
                invocation=invocation,
                parent_run_id=getattr(run, "parent_run_id", None),
            )
            with self._lock:
                self._runs[run.id] = rd
        except Exception:
            logger.debug("Failed to start Retriever span", exc_info=True)

    def _on_retriever_end(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "retriever":
            return
        try:
            inv: RetrievalInvocation = rd.invocation
            outputs = getattr(run, "outputs", None) or {}
            documents = outputs.get("documents") or []
            if documents:
                inv.documents = _documents_to_retrieval_documents(documents)
            self._handler.stop_retrieval(inv)
        except Exception:
            logger.debug("Failed to stop Retriever span", exc_info=True)

    def _on_retriever_error(self, run: Run) -> None:
        with self._lock:
            rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "retriever":
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            self._handler.fail_retrieval(
                rd.invocation,
                Error(message=str(err_str), type=Exception),
            )
        except Exception:
            logger.debug("Failed to fail Retriever span", exc_info=True)

    # ------------------------------------------------------------------
    # LangGraph ReAct Step — callback-based detection
    # ------------------------------------------------------------------

    def _maybe_enter_langgraph_react_step(self, run: Run) -> None:
        """If *run* is a child node of a LangGraph ReAct agent whose name
        equals ``LANGGRAPH_REACT_STEP_NODE`` (``"agent"``), trigger a ReAct
        step transition: end the previous step (with ``"tool_calls"``) and
        start a new one.

        Must be called **before** ``_start_chain`` so that the chain span
        is parented under the step span.
        """
        parent_id = getattr(run, "parent_run_id", None)
        if not parent_id:
            return

        with self._lock:
            parent_rd = self._runs.get(parent_id)
        if parent_rd is None or not parent_rd.is_langgraph_react:
            return

        chain_name = getattr(run, "name", "") or ""
        if chain_name != LANGGRAPH_REACT_STEP_NODE:
            return

        # End previous step (it had tool_calls since another round started)
        if parent_rd.active_step is not None:
            self._exit_react_step(parent_id, "tool_calls")

        self._enter_react_step(parent_id)

    # ------------------------------------------------------------------
    # ReAct Step — called from patch wrapper or callback detection
    # ------------------------------------------------------------------

    def _enter_react_step(self, agent_run_id: UUID) -> None:
        """Create a ReAct Step span and redirect child spans to it."""
        with self._lock:
            agent_rd = self._runs.get(agent_run_id)
        if agent_rd is None or agent_rd.run_kind != "agent":
            return

        if agent_rd.original_context is None:
            agent_rd.original_context = agent_rd.context

        agent_rd.react_round += 1
        inv = ReactStepInvocation(round=agent_rd.react_round)
        self._handler.start_react_step(inv, context=agent_rd.original_context)

        step_ctx = (
            set_span_in_context(inv.span)
            if inv.span
            else agent_rd.original_context
        )
        agent_rd.active_step = _RunData(
            run_kind="react_step",
            span=inv.span,
            context=step_ctx,
            invocation=inv,
        )
        agent_rd.context = step_ctx

    def _exit_react_step(self, agent_run_id: UUID, finish_reason: str) -> None:
        """End the current ReAct Step span and restore Agent context."""
        with self._lock:
            agent_rd = self._runs.get(agent_run_id)
        if agent_rd is None or agent_rd.active_step is None:
            return

        step_inv: ReactStepInvocation = agent_rd.active_step.invocation
        step_inv.finish_reason = finish_reason
        self._handler.stop_react_step(step_inv)
        agent_rd.active_step = None
        if agent_rd.original_context is not None:
            agent_rd.context = agent_rd.original_context

    def _fail_react_step(self, agent_run_id: UUID, error_msg: str) -> None:
        """Fail the current ReAct Step span and restore Agent context."""
        with self._lock:
            agent_rd = self._runs.get(agent_run_id)
        if agent_rd is None or agent_rd.active_step is None:
            return

        step_inv: ReactStepInvocation = agent_rd.active_step.invocation
        self._handler.fail_react_step(
            step_inv, Error(message=error_msg, type=Exception)
        )
        agent_rd.active_step = None
        if agent_rd.original_context is not None:
            agent_rd.context = agent_rd.original_context

    # ------------------------------------------------------------------
    # Deep copy / copy — return self (shared singleton)
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo: dict) -> LoongsuiteTracer:
        return self

    def __copy__(self) -> LoongsuiteTracer:
        return self


def _normalize_run_id(run_id: UUID | str | Any) -> UUID | None:
    if isinstance(run_id, UUID):
        return run_id
    try:
        return UUID(str(run_id))
    except (TypeError, ValueError):
        return None


def get_otel_span_for_run(run_id: UUID | str | Any) -> Span | None:
    """Return the active OTel span for a LangChain run across tracers."""
    for tracer in list(_TRACER_INSTANCES):
        span = tracer.get_otel_span_for_run(run_id)
        if span is not None:
            return span
    return None


def _is_task_tool_span(span: Span | None) -> bool:
    attributes = getattr(span, "attributes", None)
    if not isinstance(attributes, Mapping):
        return False
    return (
        attributes.get(GEN_AI_SPAN_KIND) == "TOOL"
        and attributes.get("gen_ai.tool.name") == "task"
    )


def _is_deepagents_subagent_candidate(run: Run) -> bool:
    metadata = getattr(run, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("ls_agent_type") == "subagent":
        return True
    return (
        metadata.get("ls_integration") == "deepagents"
        and _is_task_tool_span(get_current_span())
    )


def _enrich_deepagents_agent_span(span: Span | None, run: Run) -> None:
    """Write deepagents attributes when LangGraph metadata marks an agent run."""
    if span is None:
        return
    metadata = getattr(run, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return
    if not _is_deepagents_metadata(metadata):
        return

    span.set_attribute(_GEN_AI_FRAMEWORK, _DEEPAGENTS_FRAMEWORK_NAME)
    versions = metadata.get(_DEEPAGENTS_METADATA_VERSIONS)
    if isinstance(versions, Mapping):
        version = versions.get(_DEEPAGENTS_FRAMEWORK_NAME)
        if version:
            span.set_attribute(_GEN_AI_FRAMEWORK_VERSION, str(version))

    agent_name = metadata.get(_DEEPAGENTS_METADATA_AGENT_NAME)
    if agent_name:
        span.set_attribute(_GEN_AI_AGENT_NAME, str(agent_name))

    agent_type = metadata.get(_DEEPAGENTS_METADATA_AGENT_TYPE)
    if agent_type:
        span.set_attribute(_GEN_AI_AGENT_TYPE, str(agent_type))

    description = metadata.get(_DEEPAGENTS_METADATA_SUBAGENT_DESCRIPTION)
    if description:
        span.set_attribute(_GEN_AI_AGENT_DESCRIPTION, str(description))


def _is_deepagents_metadata(metadata: Mapping[str, Any]) -> bool:
    if metadata.get(_DEEPAGENTS_METADATA_INTEGRATION) == _DEEPAGENTS_FRAMEWORK_NAME:
        return True
    if metadata.get(_DEEPAGENTS_METADATA_AGENT_TYPE) == _DEEPAGENTS_SUBAGENT_TYPE:
        return True
    versions = metadata.get(_DEEPAGENTS_METADATA_VERSIONS)
    return isinstance(versions, Mapping) and _DEEPAGENTS_FRAMEWORK_NAME in versions


def _is_deepagents_subagent_root(run: Run) -> bool:
    metadata = getattr(run, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get(_DEEPAGENTS_METADATA_AGENT_TYPE) != _DEEPAGENTS_SUBAGENT_TYPE:
        return False
    agent_name = metadata.get(_DEEPAGENTS_METADATA_AGENT_NAME)
    return bool(agent_name and getattr(run, "name", None) == agent_name)


# ---------------------------------------------------------------------------
# LangGraph message helpers (module-level, used by _start_agent / _stop_agent)
# ---------------------------------------------------------------------------


def _extract_langgraph_input_message(msg: Any) -> InputMessage | None:
    """Convert a LangGraph input message to ``InputMessage``.

    LangGraph inputs may be LangChain message objects, tuples, or dicts.
    """
    # Tuple: ("user", "hello")
    if isinstance(msg, (list, tuple)) and len(msg) == 2:
        role, content = msg
        if isinstance(content, str) and content:
            return InputMessage(role=str(role), parts=[Text(content=content)])
        return None

    # LangChain message object (HumanMessage, AIMessage, etc.)
    content = getattr(msg, "content", None)
    if content and isinstance(content, str):
        role_map = {
            "HumanMessage": "user",
            "AIMessage": "assistant",
            "SystemMessage": "system",
            "ToolMessage": "tool",
        }
        cls_name = type(msg).__name__
        role = role_map.get(cls_name, "user")
        return InputMessage(role=role, parts=[Text(content=content)])

    return None


def _extract_message_content(msg: Any) -> str | None:
    """Extract text content from a LangChain message object or dict."""
    content = getattr(msg, "content", None)
    if content and isinstance(content, str):
        return content
    if isinstance(msg, dict):
        return msg.get("content") or msg.get("text")
    return None
