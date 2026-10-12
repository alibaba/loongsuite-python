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
OpenTelemetry LlamaIndex Instrumentation

Provides automatic instrumentation for LlamaIndex (``llama-index-core``) by
attaching to its **native instrumentation dispatcher**
(``llama_index.core.instrumentation``) rather than monkey-patching call
sites. A ``BaseSpanHandler`` registered on the root ``Dispatcher`` receives
LlamaIndex's own span enter/exit/drop signals and projects the *real* GenAI
operations onto the shared
``opentelemetry.util.genai.extended_handler.ExtendedTelemetryHandler`` -- the
same handler the other LoongSuite agent instrumentations (agno,
hermes-agent, ...) use.

Span lifecycle ownership
------------------------
The shared ``ExtendedTelemetryHandler`` -- not this package -- owns every
agent/tool/LLM span: it starts and ends the span, sets semantic-convention
attributes (including ``gen_ai.span.kind``), records metrics and, when the
GenAI content-capture switches request it, emits the operation-detail log
event. This module only decides *which* LlamaIndex span is which GenAI
operation and builds the matching invocation dataclass:

  * the user-facing agent run (``<Agent>.run`` / ``arun``)
        -> ``InvokeAgentInvocation`` / ``start_invoke_agent`` ...
           exactly one ``invoke_agent`` AGENT span per agent run;
  * the actual tool/function execution (``<Tool>.call`` / ``acall``)
        -> ``ExecuteToolInvocation`` / ``start_execute_tool`` ...
           one ``execute_tool`` TOOL span per tool execution;
  * model calls (``*.chat`` / ``*.complete`` / ``*.predict`` and stream
    forms) -> ``LLMInvocation`` / ``start_llm`` ... LLM spans.
    The LLM turns *inside* an agent loop are LLM, never AGENT spans;
  * embedding / retrieval / rerank calls are routed through the matching
    handler invocations as well.

Agent-internal machinery (``init_run``, ``setup_agent``, ``run_agent_step``,
output parsing, the agent's own ``call_tool`` orchestration, tool-result
aggregation, query-engine ``query`` chains, ...) creates no span: those are
steps *within* one agent invocation, not additional agent or tool
invocations.

Content capture
---------------
Message text, tool arguments/results and operation-detail events are all
produced **by the shared handler**, so the standard
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` switch behaves here
exactly as it does for agno/hermes:

  * ``NO_CONTENT`` (default)  -> structural spans only;
  * ``SPAN_ONLY``             -> content on span attributes, no event;
  * ``EVENT_ONLY``            -> content on a log event only, no span content;
  * ``SPAN_AND_EVENT``        -> content on both.

Fail-safety
-----------
LlamaIndex already swallows span/event-handler exceptions, but this package
does not rely on that: every handler call and every invocation-enrichment
step is additionally guarded, so a telemetry failure can never block the
agent run, alter its result, or replace a business exception.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Collection, Optional

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.llama_index.package import _instruments
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

logger = logging.getLogger(__name__)

# -- Framework identifier -------------------------------------
_FRAMEWORK = "llama_index"

# Re-exported semconv keys/values for tests and downstream users.
_GEN_AI_SPAN_KIND = GEN_AI_SPAN_KIND
_GEN_AI_FRAMEWORK = "gen_ai.framework"
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_SPAN_KIND_LLM = GenAiSpanKindValues.LLM.value
_SPAN_KIND_EMBEDDING = GenAiSpanKindValues.EMBEDDING.value
_SPAN_KIND_RETRIEVER = GenAiSpanKindValues.RETRIEVER.value
_SPAN_KIND_RERANKER = GenAiSpanKindValues.RERANKER.value
_SPAN_KIND_TOOL = GenAiSpanKindValues.TOOL.value
_SPAN_KIND_AGENT = GenAiSpanKindValues.AGENT.value

# Internal operation kinds (mapped 1:1 to ExtendedTelemetryHandler methods).
_KIND_AGENT = "agent"
_KIND_TOOL = "tool"
_KIND_LLM = "llm"
_KIND_EMBEDDING = "embedding"
_KIND_RETRIEVAL = "retrieval"
_KIND_RERANK = "rerank"

# The genuine user-facing agent invocation boundary: the public run
# entrypoint on an Agent / AgentWorkflow class.
_AGENT_RUN_METHODS = frozenset({"run", "arun"})
# The actual tool/function execution on a Tool object. ``__call__`` only
# delegates to ``call`` and is intentionally excluded so one tool execution
# produces exactly one TOOL span; the agent's own ``call_tool`` is
# orchestration and is excluded as well.
_TOOL_EXEC_METHODS = frozenset({"call", "acall"})
_LLM_METHODS = frozenset(
    {
        "chat",
        "achat",
        "complete",
        "acomplete",
        "stream_chat",
        "astream_chat",
        "stream_complete",
        "astream_complete",
        "predict",
        "apredict",
        "structured_predict",
        "astructured_predict",
        "stream_structured_predict",
        "astream_structured_predict",
    }
)
_LLM_METHOD_PREFIXES = (
    "stream_chat",
    "astream_chat",
    "stream_complete",
    "astream_complete",
)


# =============================================================================
# =============================================================================


def _span_id_prefix(id_: str) -> str:
    """Return the ``<Class>.<method>`` portion of a LlamaIndex span id.

    LlamaIndex span ids look like ``MockLLM.chat-8c7b...-uuid``. The class
    and method carry the semantic meaning; the uuid suffix is per-invocation.
    """
    if not id_:
        return ""
    return id_.split("-", 1)[0]


def _classify(prefix: str) -> Optional[str]:
    """Map a ``<Class>.<method>`` prefix to a handler operation kind.

    Returns ``None`` for LlamaIndex spans that must not produce their own
    GenAI span (agent-internal setup/parse/aggregation steps, generic
    query-engine chains, and the ``Tool.__call__`` trampoline).

    Method is classified first; the class name is only used to confirm the
    boundary, so e.g. ``BaseWorkflowAgent.call_tool`` stays skipped
    (orchestration) while ``FunctionTool.acall`` is a TOOL execution, and
    ``MockLLM.astream_chat`` *inside* an agent run stays LLM, not AGENT.
    """
    if not prefix:
        return None
    lower = prefix.lower()
    if "." in lower:
        class_name, method = lower.rsplit(".", 1)
    else:
        class_name, method = "", lower

    # Agent run: only the public run entrypoint on an *Agent class is the
    # invoke_agent boundary.
    if method in _AGENT_RUN_METHODS and (
        "agent" in class_name or "agentworkflow" in class_name
    ):
        return _KIND_AGENT

    # Tool execution: the actual Tool.call / Tool.acall.
    if method in _TOOL_EXEC_METHODS and "tool" in class_name:
        return _KIND_TOOL

    # Embedding calls.
    if "embedding" in method or "embed" in method:
        return _KIND_EMBEDDING

    # Model calls -- LLM regardless of whether they run standalone, inside an
    # agent loop, or inside an indexing/retrieval pipeline.
    if method in _LLM_METHODS or method.startswith(_LLM_METHOD_PREFIXES):
        return _KIND_LLM

    # Retrieval: a real retriever retrieving.
    if "retrieve" in method and "retriever" in class_name:
        return _KIND_RETRIEVAL

    # Rerank / node postprocessing.
    if "rerank" in method or (
        "postprocess" in method and "rerank" in class_name
    ):
        return _KIND_RERANK

    return None


# =============================================================================
# =============================================================================


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a telemetry operation; never let telemetry break business code.

    Returns the call's result on success and ``None`` on failure. Failures
    are logged at debug level because tracing must stay invisible to the host
    application.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - telemetry must never propagate
        logger.debug("LlamaIndex telemetry operation failed: %s", exc)
        return None


def _release_span(invocation: Any) -> None:
    """Best-effort detach/end for an invocation that owns an open span.

    Used when a ``start_*`` raised after the shared handler had already
    created the span and attached its context, and as the fallback when a
    ``stop_*``/``fail_*`` raised. Either way the handler-owned span must not
    stay open and the ambient context must be restored, so a telemetry fault
    cannot corrupt the caller's trace. Everything is guarded because the
    invocation may be only partially initialized.
    """
    token = getattr(invocation, "context_token", None)
    if token is not None:
        try:
            context_api.detach(token)
        except Exception:  # noqa: BLE001
            pass
        invocation.context_token = None
    span = getattr(invocation, "span", None)
    if span is not None:
        try:
            if span.is_recording():
                span.end()
        except Exception:  # noqa: BLE001
            pass


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _input_message(role: Any, content: Any) -> InputMessage:
    role_value = getattr(role, "value", role)
    return InputMessage(
        role=str(role_value) if role_value is not None else "user",
        parts=[Text(content=_text(content))],
    )


def _output_message(content: Any) -> OutputMessage:
    return OutputMessage(
        role="assistant",
        parts=[Text(content=_text(content))],
        finish_reason="stop",
    )


def _messages_from_llama(messages: Any) -> list[InputMessage]:
    """Convert LlamaIndex ``ChatMessage`` objects into util-genai inputs."""
    out: list[InputMessage] = []
    if not messages:
        return out
    for m in messages:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if content is None:
            content = str(m)
        out.append(_input_message(role, content))
    return out


def _extract_usage(raw: Any) -> dict[str, int]:
    """Best-effort token-usage extraction from a provider raw response."""
    usage: dict[str, int] = {}
    if raw is None:
        return usage
    u = (
        raw.get("usage")
        if isinstance(raw, dict)
        else getattr(raw, "usage", None)
    )
    if u is None:
        return usage

    def _get(obj: Any, *names: str) -> Any:
        for n in names:
            if isinstance(obj, dict):
                if n in obj and obj[n] is not None:
                    return obj[n]
            else:
                v = getattr(obj, n, None)
                if v is not None:
                    return v
        return None

    prompt = _get(u, "prompt_tokens", "input_tokens")
    completion = _get(u, "completion_tokens", "output_tokens")
    if isinstance(prompt, int):
        usage["input"] = prompt
    if isinstance(completion, int):
        usage["output"] = completion
    if "input" in usage and "output" in usage:
        usage["total"] = usage["input"] + usage["output"]
    return usage


def _error(exc: BaseException) -> Error:
    return Error(message=str(exc), type=type(exc))


# =============================================================================
# =============================================================================


def _llm_request_model(instance: Any) -> Optional[str]:
    for name in ("model", "id", "name", "model_name"):
        value = getattr(instance, name, None)
        if value:
            return str(value)
    return None


def _build_agent_invocation(bound_args: Any, instance: Any) -> Any:
    from opentelemetry.util.genai.extended_types import InvokeAgentInvocation

    user_msg = None
    if bound_args is not None:
        args = bound_args.arguments
        start_event = args.get("start_event")
        if start_event is not None:
            user_msg = getattr(start_event, "user_msg", None)
        else:
            user_msg = args.get("message") or args.get("user_msg")

    system_instruction = []
    prompt = getattr(instance, "system_prompt", None)
    if prompt:
        system_instruction = [Text(content=_text(prompt))]

    input_messages: list[InputMessage] = []
    if user_msg is not None:
        if hasattr(user_msg, "role"):
            input_messages = _messages_from_llama([user_msg])
        else:
            input_messages = [_input_message("user", user_msg)]

    return InvokeAgentInvocation(
        provider=_FRAMEWORK,
        agent_name=getattr(instance, "name", None) or type(instance).__name__,
        agent_description=getattr(instance, "description", None),
        request_model=_llm_request_model(getattr(instance, "llm", None)),
        input_messages=input_messages,
        system_instruction=system_instruction,
        attributes={_GEN_AI_FRAMEWORK: _FRAMEWORK},
    )


def _build_tool_invocation(bound_args: Any, instance: Any) -> Any:
    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation

    metadata = getattr(instance, "metadata", None)
    tool_name = getattr(metadata, "name", None) or type(instance).__name__
    description = getattr(metadata, "description", None)
    arguments = None
    if bound_args is not None:
        arguments = bound_args.arguments.get("kwargs")

    return ExecuteToolInvocation(
        tool_name=str(tool_name),
        provider=_FRAMEWORK,
        tool_description=str(description) if description else None,
        tool_type="function",
        tool_call_arguments=arguments,
        attributes={_GEN_AI_FRAMEWORK: _FRAMEWORK},
    )


def _build_llm_invocation(
    bound_args: Any, instance: Any, prefix: str = ""
) -> LLMInvocation:
    messages = None
    if bound_args is not None:
        messages = bound_args.arguments.get("messages")
    method = (
        prefix.lower().rsplit(".", 1)[-1] if "." in prefix else prefix.lower()
    )
    # Prompt-based (complete/predict) calls map to completion; every
    # message-based call (chat, structured chat, streaming chat) maps to chat.
    operation_name = (
        "text_completion"
        if "complete" in method or method in {"predict", "apredict"}
        else "chat"
    )
    return LLMInvocation(
        request_model=_llm_request_model(instance) or "",
        operation_name=operation_name,
        provider=_FRAMEWORK,
        input_messages=_messages_from_llama(messages),
        attributes={_GEN_AI_FRAMEWORK: _FRAMEWORK},
    )


def _build_embedding_invocation(instance: Any) -> Any:
    from opentelemetry.util.genai.extended_types import EmbeddingInvocation

    return EmbeddingInvocation(
        request_model=getattr(instance, "model_name", None)
        or type(instance).__name__,
        provider=_FRAMEWORK,
        attributes={_GEN_AI_FRAMEWORK: _FRAMEWORK},
    )


def _build_retrieval_invocation() -> Any:
    from opentelemetry.util.genai.extended_types import RetrievalInvocation

    return RetrievalInvocation(
        provider=_FRAMEWORK,
        attributes={_GEN_AI_FRAMEWORK: _FRAMEWORK},
    )


def _build_rerank_invocation() -> Any:
    from opentelemetry.util.genai.extended_types import RerankInvocation

    return RerankInvocation(
        provider=_FRAMEWORK,
        attributes={_GEN_AI_FRAMEWORK: _FRAMEWORK},
    )


def _enrich_llm_response(invocation: LLMInvocation, response: Any) -> None:
    """Populate an LLM invocation from a LlamaIndex chat/completion response.

    Chat/completion end events pass a response object (``.message.content`` or
    ``.text``); predict/structured-predict end events pass the generated output
    directly, which is commonly a plain string.
    """
    if response is None:
        return
    message = getattr(response, "message", None)
    if message is not None and getattr(message, "content", None) is not None:
        invocation.output_messages = [
            _output_message(getattr(message, "content"))
        ]
    elif getattr(response, "text", None) is not None:
        invocation.output_messages = [
            _output_message(getattr(response, "text"))
        ]
    elif isinstance(response, str):
        invocation.output_messages = [_output_message(response)]
    raw = getattr(response, "raw", None)
    usage = _extract_usage(raw) or _extract_usage(response)
    if "input" in usage:
        invocation.input_tokens = usage["input"]
    if "output" in usage:
        invocation.output_tokens = usage["output"]
    model = getattr(response, "model", None)
    if model:
        invocation.response_model_name = str(model)


def _enrich_agent_result(invocation: Any, result: Any) -> None:
    """Populate an invoke_agent invocation from the run result.

    Workflow agents finish with a ``workflows.events.StopEvent`` whose
    ``result`` is the final agent output.
    """
    if result is None:
        return
    output = getattr(result, "result", None)
    if output is None and not hasattr(result, "result"):
        output = result
    if output is None:
        return
    message = getattr(output, "message", None)
    if message is not None and getattr(message, "content", None) is not None:
        invocation.output_messages = [
            _output_message(getattr(message, "content"))
        ]
    else:
        invocation.output_messages = [_output_message(output)]
    invocation.finish_reasons = ["stop"]


def _enrich_tool_result(invocation: Any, result: Any) -> None:
    """Populate an execute_tool invocation.

    Agent runs finish ``call_tool`` with a ``ToolCallResult``; a direct tool
    call finishes with a LlamaIndex ``ToolOutput``.
    """
    if result is None:
        return
    tool_name = getattr(result, "tool_name", None)
    if tool_name and not invocation.tool_name:
        invocation.tool_name = str(tool_name)
    tool_id = getattr(result, "tool_id", None)
    if tool_id:
        invocation.tool_call_id = str(tool_id)
    kwargs = getattr(result, "tool_kwargs", None)
    if kwargs is not None and invocation.tool_call_arguments is None:
        invocation.tool_call_arguments = kwargs
    output = getattr(result, "tool_output", None)
    if output is None:
        output = getattr(result, "content", None)
    if output is None:
        output = getattr(result, "raw_output", None)
    if output is not None:
        invocation.tool_call_result = (
            output if isinstance(output, str) else _text(output)
        )


def _retrieval_documents(nodes: Any) -> list[Any]:
    from opentelemetry.util.genai.extended_types import RetrievalDocument

    docs: list[Any] = []
    for item in nodes or []:
        node_obj = getattr(item, "node", item)
        node_id = getattr(node_obj, "node_id", None) or getattr(
            node_obj, "id_", None
        )
        score = getattr(item, "score", None)
        getter = getattr(node_obj, "get_content", None)
        text = (
            getter() if callable(getter) else getattr(node_obj, "text", None)
        )
        docs.append(
            RetrievalDocument(
                id=str(node_id) if node_id is not None else None,
                score=float(score)
                if isinstance(score, (int, float))
                else None,
                content=_text(text) if text is not None else None,
            )
        )
    return docs


def _enrich_retrieval_result(invocation: Any, result: Any) -> None:
    nodes = getattr(result, "nodes", None)
    if nodes is None and isinstance(result, (list, tuple)):
        nodes = result
    if nodes:
        invocation.documents = _retrieval_documents(nodes)


def _enrich_embedding_result(invocation: Any, result: Any) -> None:
    if (
        isinstance(result, (list, tuple))
        and result
        and isinstance(result[0], (list, tuple))
    ):
        invocation.dimension_count = len(result[0])


def _enrich_rerank_result(invocation: Any, result: Any) -> None:
    nodes = getattr(result, "nodes", None)
    if nodes is None and isinstance(result, (list, tuple)):
        nodes = result
    if nodes is not None:
        invocation.documents_count = len(nodes)


# =============================================================================
# =============================================================================


class _SpanRecord:
    """The handler invocation owned by one open LlamaIndex span."""

    __slots__ = ("kind", "invocation")

    def __init__(self, kind: str, invocation: Any):
        self.kind = kind
        self.invocation = invocation


_START_METHODS = {
    _KIND_AGENT: "start_invoke_agent",
    _KIND_TOOL: "start_execute_tool",
    _KIND_LLM: "start_llm",
    _KIND_EMBEDDING: "start_embedding",
    _KIND_RETRIEVAL: "start_retrieval",
    _KIND_RERANK: "start_rerank",
}
_STOP_METHODS = {
    _KIND_AGENT: "stop_invoke_agent",
    _KIND_TOOL: "stop_execute_tool",
    _KIND_LLM: "stop_llm",
    _KIND_EMBEDDING: "stop_embedding",
    _KIND_RETRIEVAL: "stop_retrieval",
    _KIND_RERANK: "stop_rerank",
}
_FAIL_METHODS = {
    _KIND_AGENT: "fail_invoke_agent",
    _KIND_TOOL: "fail_execute_tool",
    _KIND_LLM: "fail_llm",
    _KIND_EMBEDDING: "fail_embedding",
    _KIND_RETRIEVAL: "fail_retrieval",
    _KIND_RERANK: "fail_rerank",
}
_EXIT_ENRICHERS = {
    _KIND_AGENT: _enrich_agent_result,
    _KIND_TOOL: _enrich_tool_result,
    _KIND_LLM: lambda inv, result: _enrich_llm_response(inv, result),
    _KIND_EMBEDDING: _enrich_embedding_result,
    _KIND_RETRIEVAL: _enrich_retrieval_result,
    _KIND_RERANK: _enrich_rerank_result,
}


def _build_span_handler(genai_handler: Any):
    """Construct the dispatcher span handler bound to the genai handler."""
    from llama_index.core.instrumentation.span_handlers.base import (
        BaseSpanHandler,
    )

    class _LoongSuiteSpanHandler(BaseSpanHandler):
        """Drive ``ExtendedTelemetryHandler`` from LlamaIndex span signals.

        ``BaseSpanHandler`` is a pydantic model, so mutable per-instance state
        is stored via ``object.__setattr__`` to bypass field validation.
        """

        model_config = {"arbitrary_types_allowed": True}

        def __init__(self, handler: Any, **kwargs: Any):
            super().__init__(**kwargs)
            object.__setattr__(self, "_ls_genai_handler", handler)
            object.__setattr__(self, "_ls_records", {})
            # LlamaIndex parent_span_id for every incoming span id, including
            # spans this handler skips, so parent resolution can walk across
            # skipped agent-internal steps to the nearest handler-owned span.
            object.__setattr__(self, "_ls_parents", {})
            object.__setattr__(self, "_ls_lock", threading.Lock())
            object.__setattr__(self, "_ls_stopped", False)

        # -- helpers -------------------------------------------------------
        def _records_map(self) -> dict[str, _SpanRecord]:
            return object.__getattribute__(self, "_ls_records")

        def _parents_map(self) -> dict[str, Optional[str]]:
            return object.__getattribute__(self, "_ls_parents")

        def _lock(self) -> threading.Lock:
            return object.__getattribute__(self, "_ls_lock")

        def _genai(self) -> Any:
            return object.__getattribute__(self, "_ls_genai_handler")

        def _stopped(self) -> bool:
            return object.__getattribute__(self, "_ls_stopped")

        def class_name(self) -> str:  # pydantic-friendly identity
            return "LoongSuiteSpanHandler"

        def record_for(self, span_id: str) -> Optional[_SpanRecord]:
            """The record for an open LlamaIndex span id, if any."""
            if not span_id:
                return None
            with self._lock():
                return self._records_map().get(span_id)

        def stop_and_drain(self) -> None:
            """Stop creating spans and best-effort finish any left open.

            LlamaIndex only dispatches exit/drop to handlers still attached,
            so uninstrument mid-flight would otherwise strand every open
            span. Draining goes through the normal fail paths so
            handler-owned spans are always detached/ended.
            """
            object.__setattr__(self, "_ls_stopped", True)
            with self._lock():
                ids = list(self._records_map().keys())
            for id_ in ids:
                self._finish(
                    id_,
                    error=RuntimeError("llama_index instrumentation removed"),
                )

        # -- parenting -----------------------------------------------------
        def _parent_context_locked(
            self,
            parent_span_id: Optional[str],
            parents: Optional[dict[str, Optional[str]]] = None,
        ) -> Optional[Any]:
            """Nearest handler-owned ancestor, walking skipped spans.

            Caller MUST hold ``self._lock()``. Skipped LlamaIndex spans
            (agent-internal steps, the agent's ``call_tool`` orchestration)
            have no record of their own, so walk the recorded parent chain
            until the nearest span the handler actually opened. This keeps
            every TOOL span nested directly under the one AGENT span even when
            several un-instrumented orchestration spans sit between them.

            ``parents`` is the in-lock snapshot to walk; ``records`` is read
            straight from the map under the same lock so this is safe to call
            from inside the atomic ``new_span`` critical section.
            """
            records = self._records_map()
            if parents is None:
                parents = self._parents_map()
            seen: set[str] = set()
            current = parent_span_id
            while current and current not in seen:
                seen.add(current)
                record = records.get(current)
                if record is not None and record.invocation.span is not None:
                    return trace_api.set_span_in_context(
                        record.invocation.span
                    )
                current = parents.get(current)
            return None

        def _parent_context(
            self, parent_span_id: Optional[str]
        ) -> Optional[Any]:
            """Lock-acquiring wrapper around ``_parent_context_locked``."""
            with self._lock():
                return self._parent_context_locked(parent_span_id)

        # -- lifecycle -----------------------------------------------------
        def new_span(
            self,
            id_: str,
            bound_args,
            instance: Optional[Any] = None,
            parent_span_id: Optional[str] = None,
            tags: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ):
            # Register the parent edge, gate on the stopped flag and publish
            # the record under one lock. Otherwise a concurrent
            # ``stop_and_drain()`` could run between the stopped check and the
            # record insertion, snapshotting an empty map and stranding this
            # span (exit/drop are only dispatched to attached handlers).
            with self._lock():
                parents = self._parents_map()
                parents[id_] = parent_span_id

                if self._stopped():
                    return None

                prefix = _span_id_prefix(id_)
                kind = _classify(prefix)
                if kind is None:
                    return None

                try:
                    invocation = self._build_invocation(
                        kind, bound_args, instance, prefix
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to build invocation for %s: %s", id_, exc
                    )
                    return None

                parent_ctx = self._parent_context_locked(
                    parent_span_id, parents
                )
                record = _SpanRecord(kind, invocation)
                started = _safe(
                    getattr(self._genai(), _START_METHODS[kind]),
                    invocation,
                    context=parent_ctx,
                )
                if started is None:
                    # start_* raised: it may already have created the span and
                    # attached its context before failing, so release whatever
                    # it created (context restored, no span left open) before
                    # dropping the invocation. No record is published, so
                    # exit/drop stay no-ops and the business call proceeds
                    # completely untouched.
                    _release_span(invocation)
                    return None

                self._records_map()[id_] = record
            return None

        @staticmethod
        def _build_invocation(
            kind: str, bound_args: Any, instance: Any, prefix: str = ""
        ) -> Any:
            if kind == _KIND_AGENT:
                return _build_agent_invocation(bound_args, instance)
            if kind == _KIND_TOOL:
                return _build_tool_invocation(bound_args, instance)
            if kind == _KIND_LLM:
                return _build_llm_invocation(bound_args, instance, prefix)
            if kind == _KIND_EMBEDDING:
                return _build_embedding_invocation(instance)
            if kind == _KIND_RETRIEVAL:
                return _build_retrieval_invocation()
            return _build_rerank_invocation()

        def _finish(
            self,
            id_: str,
            result: Any = None,
            error: Optional[BaseException] = None,
        ) -> None:
            with self._lock():
                record = self._records_map().pop(id_, None)
                self._parents_map().pop(id_, None)
            if record is None:
                return
            invocation = record.invocation
            if error is None:
                _safe(_EXIT_ENRICHERS[record.kind], invocation, result)
                self._finalize(
                    getattr(self._genai(), _STOP_METHODS[record.kind]),
                    invocation,
                )
            else:
                self._finalize(
                    getattr(self._genai(), _FAIL_METHODS[record.kind]),
                    invocation,
                    _error(error),
                )

        @staticmethod
        def _finalize(
            method: Callable[..., Any], invocation: Any, *args: Any
        ) -> None:
            """Call a stop/fail handler method without stranding a span.

            Even if the handler's own attribute/event/metrics code raises
            (e.g. an injected ``set_attribute`` failure), make sure the
            handler-attached context is detached and the span itself ends so a
            telemetry fault cannot corrupt the trace.
            """
            try:
                method(invocation, *args)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Telemetry finalize failed: %s", exc)
                _release_span(invocation)

        def prepare_to_exit_span(
            self,
            id_: str,
            bound_args,
            instance: Optional[Any] = None,
            result: Optional[Any] = None,
            **kwargs: Any,
        ):
            self._finish(id_, result=result)
            return None

        def prepare_to_drop_span(
            self,
            id_: str,
            bound_args,
            instance: Optional[Any] = None,
            err: Optional[BaseException] = None,
            **kwargs: Any,
        ):
            self._finish(id_, error=err)
            return None

    return _LoongSuiteSpanHandler(genai_handler)


# =============================================================================
# =============================================================================


def _build_event_handler(span_handler: Any):
    from llama_index.core.instrumentation.event_handlers.base import (
        BaseEventHandler,
    )

    class _LoongSuiteEventHandler(BaseEventHandler):
        """Fold LlamaIndex events onto the matching open invocation.

        Only invocation *data* fields are set; attributes, content capture
        and operation-detail event emission remain entirely owned by the
        shared handler, which is what makes
        NO_CONTENT/SPAN_ONLY/EVENT_ONLY/SPAN_AND_EVENT behave.
        """

        model_config = {"arbitrary_types_allowed": True}

        def __init__(self, handler: Any, **kwargs: Any):
            super().__init__(**kwargs)
            object.__setattr__(self, "_span_handler", handler)

        @classmethod
        def class_name(cls) -> str:
            return "LoongSuiteEventHandler"

        def handle(self, event: Any, **kwargs: Any) -> Any:
            _safe(self._handle, event)
            return None

        def _handle(self, event: Any) -> None:
            span_id = getattr(event, "id_", None) or getattr(
                event, "span_id", None
            )
            span_handler = object.__getattribute__(self, "_span_handler")
            record = span_handler.record_for(span_id)
            if record is None:
                return
            name = event.class_name()
            if record.kind == _KIND_LLM:
                self._handle_llm(record.invocation, name, event)
            elif record.kind == _KIND_RETRIEVAL and name.endswith("EndEvent"):
                nodes = getattr(event, "nodes", None)
                if nodes:
                    record.invocation.documents = _retrieval_documents(nodes)
            elif record.kind == _KIND_EMBEDDING and name.endswith("EndEvent"):
                chunks = getattr(event, "embeddings", None)
                if chunks and isinstance(chunks[0], (list, tuple)):
                    record.invocation.dimension_count = len(chunks[0])

        @staticmethod
        def _handle_llm(
            invocation: LLMInvocation, name: str, event: Any
        ) -> None:
            if name.endswith("StartEvent"):
                messages = getattr(event, "messages", None)
                if messages and not invocation.input_messages:
                    invocation.input_messages = _messages_from_llama(messages)
                prompt = getattr(event, "prompt", None)
                if prompt and not invocation.input_messages:
                    invocation.input_messages = [
                        _input_message("user", prompt)
                    ]
                model_dict = getattr(event, "model_dict", None)
                if isinstance(model_dict, dict):
                    model = model_dict.get("model") or model_dict.get(
                        "model_name"
                    )
                    if model and not invocation.request_model:
                        invocation.request_model = str(model)
            elif name.endswith("EndEvent"):
                # Chat/completion end events carry the generated response in
                # ``response``; predict/structured-predict end events carry it
                # in ``output``. ``messages`` on a chat end event is the
                # *request* messages, so it must never be used as the output
                # (doing so reported the user prompt as gen_ai.output).
                response = getattr(event, "response", None)
                if response is None:
                    response = getattr(event, "output", None)
                if response is not None:
                    _enrich_llm_response(invocation, response)

    return _LoongSuiteEventHandler(span_handler)


# =============================================================================
# =============================================================================


class LlamaIndexInstrumentor(BaseInstrumentor):
    """Instrumentor for LlamaIndex (``llama-index-core``).

    Registers a span handler and an event handler on LlamaIndex's root
    dispatcher. Agent/tool/LLM span lifecycles are owned by the shared
    ``ExtendedTelemetryHandler``; ``uninstrument`` removes both dispatcher
    handlers and drains any span still open.
    """

    def __init__(self):
        super().__init__()
        self._span_handler = None
        self._event_handler = None
        self._genai_handler = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
                ExtendedTelemetryHandler,
            )
        except ImportError as exc:
            raise RuntimeError(
                "loongsuite-instrumentation-llama-index requires "
                "opentelemetry-util-genai with ExtendedTelemetryHandler support"
            ) from exc

        import llama_index.core.instrumentation as instrumentation  # noqa: PLC0415

        # Build a dedicated handler (as the hermes-agent instrumentation
        # does) so the injected providers are always honored, including in
        # test processes where another package may have cached the singleton.
        genai_handler = ExtendedTelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
        )
        self._genai_handler = genai_handler

        dispatcher = instrumentation.get_dispatcher()
        span_handler = _build_span_handler(genai_handler)
        event_handler = _build_event_handler(span_handler)

        dispatcher.add_span_handler(span_handler)
        dispatcher.add_event_handler(event_handler)

        self._span_handler = span_handler
        self._event_handler = event_handler

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import llama_index.core.instrumentation as instrumentation  # noqa: PLC0415

            dispatcher = instrumentation.get_dispatcher()
            if self._span_handler is not None:
                # Close any spans still open BEFORE detaching, so removing the
                # handler cannot strand them (the dispatcher only routes
                # exit/drop to still-attached handlers).
                try:
                    self._span_handler.stop_and_drain()
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Could not drain open spans: %s", e)
                try:
                    dispatcher.span_handlers = [
                        h
                        for h in dispatcher.span_handlers
                        if h is not self._span_handler
                    ]
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Could not detach span handler: %s", e)
            if self._event_handler is not None:
                try:
                    dispatcher.event_handlers = [
                        h
                        for h in dispatcher.event_handlers
                        if h is not self._event_handler
                    ]
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Could not detach event handler: %s", e)
        finally:
            self._span_handler = None
            self._event_handler = None
            self._genai_handler = None
