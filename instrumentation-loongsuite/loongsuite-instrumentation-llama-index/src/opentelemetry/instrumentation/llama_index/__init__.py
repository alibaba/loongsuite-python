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
sites. LlamaIndex already emits a rich span/event stream through a root
``Dispatcher``; this package registers a ``BaseSpanHandler`` and a
``BaseEventHandler`` on that dispatcher and re-projects the stream onto
OpenTelemetry spans that follow the ARMS gen-ai semantic conventions
(see ``arms_docs/trace/gen-ai.md``).

Why the dispatcher seam (and not wrapt)
---------------------------------------
LlamaIndex assigns every instrumented call a span ``id_`` and, crucially, a
``parent_span_id`` that reflects the *logical* call tree — e.g. an
``llm.chat`` span parents the ``llm.complete`` span it triggers, a
``query`` span parents ``retrieve`` / ``synthesize`` children, and so on.
Reconstructing that tree by hand via ``wrapt`` would be brittle and would
miss the contextvar-propagated relationships LlamaIndex maintains across
threads and async tasks. Consuming ``parent_span_id`` directly yields a
faithful OTel trace whose parent/child structure matches LlamaIndex's own
view.

Span kind mapping (ARMS gen-ai semconv)
---------------------------------------
LlamaIndex span ids are of the form ``<Class>.<method>-<uuid>``. The
``<Class>.<method>`` prefix is classified into an ARMS ``gen_ai.span.kind``:

  * ``*.chat`` / ``*.complete`` / ``*.predict`` / ``*.stream*``  → LLM
  * ``*.get_text_embedding*`` / ``*.get_query_embedding*``       → EMBEDDING
  * ``*.retrieve`` / retriever spans                             → RETRIEVER
  * reranker / postprocessor spans                              → RERANKER
  * ``*.synthesize`` / response-synthesizer spans                → TASK
  * ``*.query`` / query-engine spans                             → CHAIN
  * ``*.chat`` on chat engines / ``*.run`` on agents             → AGENT
  * everything else                                             → CHAIN

Events (``LLMChatEndEvent``, ``EmbeddingEndEvent`` ...) carry request model,
messages and — when the provider returns it — token usage. These are folded
onto the currently-open OTel span as gen-ai attributes.

Content capture
---------------
Message text is written onto span attributes only when the shared GenAI
util's content-capture switch enables it -- set
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to ``SPAN_ONLY`` or
``SPAN_AND_EVENT`` to record ``gen_ai.input.messages`` /
``gen_ai.output.messages``; ``NO_CONTENT`` (the default) keeps the
structural spans and token metrics without message text. This is the same
control every other loongsuite instrumentation uses.
"""

import json
import logging
import threading
from typing import Any, Collection, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.llama_index.package import _instruments
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import get_content_capturing_mode

logger = logging.getLogger(__name__)

# ── Framework identifier ─────────────────────────────────────────────────────
_FRAMEWORK = "llama_index"

# ── GenAI semantic-convention attribute keys (ARMS gen-ai semconv) ───────────
# Strings inlined to avoid a hard dependency on private aliyun packages that
# aren't published to PyPI. Values track the ARMS gen-ai semconv, matching the
# sibling loongsuite instrumentation packages (terminus2, litellm, ...).
_GEN_AI_SPAN_KIND = GEN_AI_SPAN_KIND  # shared GenAI-util semconv key
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_FRAMEWORK = "gen_ai.framework"
_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"

# ── Span kind values ─────────────────────────────────────────────────────────
# Span-kind literals below are sourced from the shared GenAI util
# (GenAiSpanKindValues) so this package and the rest of loongsuite speak one
# span-kind vocabulary. TASK/CHAIN have no shared-util member yet, so they
# stay local literals until one exists.
_SPAN_KIND_LLM = GenAiSpanKindValues.LLM.value
_SPAN_KIND_EMBEDDING = GenAiSpanKindValues.EMBEDDING.value
_SPAN_KIND_RETRIEVER = GenAiSpanKindValues.RETRIEVER.value
_SPAN_KIND_RERANKER = GenAiSpanKindValues.RERANKER.value
_SPAN_KIND_TOOL = GenAiSpanKindValues.TOOL.value
_SPAN_KIND_AGENT = GenAiSpanKindValues.AGENT.value
_SPAN_KIND_TASK = "TASK"
_SPAN_KIND_CHAIN = "CHAIN"

# ── Operation-name values ────────────────────────────────────────────────────
_OP_CHAT = "chat"
_OP_EMBEDDING = "embedding"
_OP_EXECUTE_TOOL = "execute_tool"
_OP_STEP = "step"
_OP_RETRIEVE = "retrieve"
_OP_RERANK = "rerank"
_OP_TASK = "task"
_OP_CHAIN = "chain"
_OP_INVOKE_AGENT = "invoke_agent"

# ── Content capture toggle ───────────────────────────────────────────────────
# Content-message capture is delegated to the shared GenAI util so it is
# governed by the same OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
# switch (and NO_CONTENT/SPAN_ONLY/EVENT_ONLY/SPAN_AND_EVENT modes) as every
# other loongsuite instrumentation, rather than a package-specific env var.
_CONTENT_ON_SPAN_MODES = frozenset(
    {ContentCapturingMode.SPAN_ONLY, ContentCapturingMode.SPAN_AND_EVENT}
)


def _capture_content() -> bool:
    """True when message content should be written onto spans.

    Reads the shared util's content-capturing mode; unset defaults to no
    capture there, so callers that want content set the standard env var.
    """
    try:
        return get_content_capturing_mode() in _CONTENT_ON_SPAN_MODES
    except Exception:  # pragma: no cover - defensive: never break tracing
        return False


def _span_id_prefix(id_: str) -> str:
    """Return the ``<Class>.<method>`` portion of a LlamaIndex span id.

    LlamaIndex span ids look like ``MockLLM.chat-8c7b...-uuid``. The class and
    method carry the semantic meaning; the uuid suffix is per-invocation.
    """
    if not id_:
        return ""
    # The uuid is appended with a '-' separator; the class/method prefix never
    # contains a '-' in LlamaIndex's naming scheme.
    return id_.split("-", 1)[0]


def _classify(prefix: str) -> tuple:
    """Map a ``<Class>.<method>`` prefix to (span_kind, operation_name).

    Classification is **method-first**: the invoked method carries the
    operation's true meaning, and the class name is only a fallback. This
    ordering matters because several class names embed a misleading keyword —
    e.g. ``RetrieverQueryEngine.query`` is a *query engine* (CHAIN), not a
    retriever, despite ``Retriever`` appearing in the class name.
    """
    lower = prefix.lower()
    method = lower.rsplit(".", 1)[-1] if "." in lower else lower

    # ---- method-first mapping (authoritative) ----
    # Tool execution: an agent's ``call_tool`` runs the selected tool, so it is a
    # TOOL span, not an AGENT span -- even though it lives on an ``*Agent`` class.
    # Checked before everything else so the class name can never override it.
    if method in ("call_tool", "acall_tool"):
        return _SPAN_KIND_TOOL, _OP_EXECUTE_TOOL
    # Internal agent-loop machinery (setup/init/step/finalize/tool-result
    # handling). These are steps *inside* an agent invocation, not the agent
    # invocation itself, so they must not inherit AGENT from the class name.
    if method in (
        "setup_agent",
        "init_run",
        "take_step",
        "atake_step",
        "run_step",
        "arun_step",
        "_run_step",
        "finalize",
        "afinalize",
        "handle_tool_call_results",
        "aggregate_tool_results",
    ):
        return _SPAN_KIND_CHAIN, _OP_STEP
    # Embedding
    if "embedding" in method or "embed" in method:
        return _SPAN_KIND_EMBEDDING, _OP_EMBEDDING
    # LLM calls
    if method in (
        "chat",
        "achat",
        "complete",
        "acomplete",
    ) or method.startswith(
        ("stream_chat", "astream_chat", "stream_complete", "astream_complete")
    ):
        # chat/complete invoked on a chat engine or agent is the agent turn.
        if "agent" in lower or "chatengine" in lower or "chat_engine" in lower:
            return _SPAN_KIND_AGENT, _OP_INVOKE_AGENT
        return _SPAN_KIND_LLM, _OP_CHAT
    # Structured prediction is an LLM call; include the async and streaming
    # dispatcher-instrumented variants so they get LLM (not the CHAIN fallback).
    if method in (
        "predict",
        "apredict",
        "structured_predict",
        "astructured_predict",
        "stream_structured_predict",
        "astream_structured_predict",
    ):
        return _SPAN_KIND_LLM, _OP_CHAT
    # Query engine (must precede the retriever class-substring fallback so
    # RetrieverQueryEngine.query is a CHAIN, not a RETRIEVER).
    if method in ("query", "aquery"):
        return _SPAN_KIND_CHAIN, _OP_CHAIN
    # Retrieval
    if "retrieve" in method:
        return _SPAN_KIND_RETRIEVER, _OP_RETRIEVE
    # Rerank / postprocess
    if "rerank" in method or "postprocess" in method:
        return _SPAN_KIND_RERANKER, _OP_RERANK
    # Synthesis
    if "synthesize" in method:
        return _SPAN_KIND_TASK, _OP_TASK
    # Agent / workflow run: the true agent invocation entrypoint.
    if method in ("run", "arun"):
        return _SPAN_KIND_AGENT, _OP_INVOKE_AGENT

    # ---- class-name fallback (method was not decisive) ----
    # NB: no blanket ``"agent" in lower -> AGENT`` here. Only a genuine agent
    # invocation (handled above via ``run``/``chat`` on an agent) is AGENT; an
    # unrecognized method on an ``*Agent`` class is an internal step, so it falls
    # through to CHAIN rather than masquerading as another whole agent turn.
    if "queryengine" in lower or "query_engine" in lower:
        return _SPAN_KIND_CHAIN, _OP_CHAIN
    if "retriever" in lower:
        return _SPAN_KIND_RETRIEVER, _OP_RETRIEVE
    if "rerank" in lower or "postprocessor" in lower:
        return _SPAN_KIND_RERANKER, _OP_RERANK
    if "synthesizer" in lower:
        return _SPAN_KIND_TASK, _OP_TASK
    return _SPAN_KIND_CHAIN, _OP_CHAIN


def _messages_to_json(messages: Any) -> Optional[str]:
    """Serialize LlamaIndex ChatMessage list into the GenAI message schema."""
    if not messages:
        return None
    out = []
    try:
        for m in messages:
            role = getattr(m, "role", None)
            role = getattr(role, "value", role)
            content = getattr(m, "content", None)
            if content is None:
                content = str(m)
            out.append(
                {
                    "role": str(role) if role is not None else "user",
                    "parts": [{"type": "text", "content": str(content)}],
                }
            )
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        try:
            return json.dumps(str(messages), ensure_ascii=False)
        except Exception:
            return None


def _extract_usage(raw: Any) -> Dict[str, int]:
    """Best-effort extraction of token usage from a provider raw response.

    Supports both dict-shaped (``{"usage": {...}}``) and object-shaped
    (``resp.usage.prompt_tokens``) payloads across providers.
    """
    usage: Dict[str, int] = {}
    if raw is None:
        return usage
    u = None
    if isinstance(raw, dict):
        u = raw.get("usage")
    else:
        u = getattr(raw, "usage", None)
    if u is None:
        return usage

    def _get(obj, *names):
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
    total = _get(u, "total_tokens")
    if isinstance(prompt, int):
        usage["input"] = prompt
    if isinstance(completion, int):
        usage["output"] = completion
    if isinstance(total, int):
        usage["total"] = total
    elif "input" in usage and "output" in usage:
        usage["total"] = usage["input"] + usage["output"]
    return usage


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher span handler → OTel spans
# ═══════════════════════════════════════════════════════════════════════════


def _build_span_handler(tracer):
    """Construct the OTel span handler bound to ``tracer``.

    The handler class is built lazily inside this function so importing this
    module does not require ``llama-index-core`` to be installed (mirrors the
    other loongsuite packages, whose instrumentation dependency is optional
    at import time).
    """
    from llama_index.core.instrumentation.span_handlers.base import (
        BaseSpanHandler,
    )

    class _OtelSpanHandler(BaseSpanHandler):
        """Project LlamaIndex spans onto OTel spans, preserving the tree.

        ``BaseSpanHandler`` is a pydantic model, so instance mutable state
        (the tracer and the id→span/token maps) is stored via
        ``object.__setattr__`` to bypass pydantic field validation.
        """

        model_config = {"arbitrary_types_allowed": True}

        def __init__(self, otel_tracer, **kwargs: Any):
            super().__init__(**kwargs)
            object.__setattr__(self, "_otel_tracer", otel_tracer)
            object.__setattr__(self, "_otel_spans", {})
            object.__setattr__(self, "_otel_tokens", {})
            object.__setattr__(self, "_otel_lock", threading.Lock())
            object.__setattr__(self, "_otel_stopped", False)

        # -- helpers -------------------------------------------------------
        def _spans(self) -> Dict[str, Any]:
            return object.__getattribute__(self, "_otel_spans")

        def _tokens(self) -> Dict[str, Any]:
            return object.__getattribute__(self, "_otel_tokens")

        def _lock(self):
            return object.__getattribute__(self, "_otel_lock")

        def _tracer(self):
            return object.__getattribute__(self, "_otel_tracer")

        def _stopped(self) -> bool:
            return object.__getattribute__(self, "_otel_stopped")

        def stop_and_drain(self) -> None:
            """Stop creating new spans and finish any that are still open.

            LlamaIndex only dispatches exit/drop to handlers still attached to
            the dispatcher, so a handler removed mid-flight would strand every
            span open when uninstrument ran (their SDK spans and context tokens
            would never close). Flip the stopped flag first -- so no new spans
            are created while callers wind down -- then end whatever remains.
            """
            object.__setattr__(self, "_otel_stopped", True)
            spans = self._spans()
            with self._lock():
                leftover_ids = list(spans.keys())
            for id_ in leftover_ids:
                self._finish(id_)

        def class_name(self) -> str:  # pydantic-friendly identity
            return "OtelSpanHandler"

        # -- lifecycle -----------------------------------------------------
        def new_span(
            self,
            id_: str,
            bound_args,
            instance: Optional[Any] = None,
            parent_span_id: Optional[str] = None,
            tags: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ):
            if self._stopped():
                # Uninstrument in progress: create no new spans, but leave
                # already-open ones for exit/drop to finish.
                return None
            prefix = _span_id_prefix(id_)
            span_kind, op_name = _classify(prefix)

            spans = self._spans()
            with self._lock():
                parent_span = (
                    spans.get(parent_span_id) if parent_span_id else None
                )
            parent_ctx = (
                trace_api.set_span_in_context(parent_span)
                if parent_span is not None
                else None
            )

            span = self._tracer().start_span(
                prefix or "llama_index.span",
                context=parent_ctx,
                kind=SpanKind.INTERNAL,
            )
            span.set_attribute(_GEN_AI_SPAN_KIND, span_kind)
            span.set_attribute(_GEN_AI_OPERATION_NAME, op_name)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute("llama_index.span.id", id_)

            # Attach so that sibling LlamaIndex spans created via contextvars
            # (and any nested OTel instrumentation) parent correctly even when
            # LlamaIndex does not thread the parent id through.
            ctx = trace_api.set_span_in_context(span)
            token = context_api.attach(ctx)

            with self._lock():
                spans[id_] = span
                self._tokens()[id_] = token
            return None

        def _finish(self, id_: str, err: Optional[BaseException] = None):
            spans = self._spans()
            tokens = self._tokens()
            with self._lock():
                span = spans.pop(id_, None)
                token = tokens.pop(id_, None)
            if token is not None:
                try:
                    context_api.detach(token)
                except Exception:  # pragma: no cover - defensive
                    pass
            if span is not None:
                if err is not None:
                    span.record_exception(err)
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.set_status(Status(StatusCode.OK))
                span.end()

        def prepare_to_exit_span(
            self,
            id_: str,
            bound_args,
            instance: Optional[Any] = None,
            result: Optional[Any] = None,
            **kwargs: Any,
        ):
            self._finish(id_)
            return None

        def prepare_to_drop_span(
            self,
            id_: str,
            bound_args,
            instance: Optional[Any] = None,
            err: Optional[BaseException] = None,
            **kwargs: Any,
        ):
            self._finish(id_, err=err)
            return None

    return _OtelSpanHandler(tracer)


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher event handler → enrich OTel spans
# ═══════════════════════════════════════════════════════════════════════════


def _build_event_handler(span_handler):
    from llama_index.core.instrumentation.event_handlers.base import (
        BaseEventHandler,
    )

    class _OtelEventHandler(BaseEventHandler):
        """Fold LlamaIndex events onto the matching open OTel span."""

        model_config = {"arbitrary_types_allowed": True}

        def __init__(self, handler, **kwargs: Any):
            super().__init__(**kwargs)
            object.__setattr__(self, "_handler", handler)

        @classmethod
        def class_name(cls) -> str:
            return "OtelEventHandler"

        def _span_for(self, event):
            handler = object.__getattribute__(self, "_handler")
            spans = handler._spans()
            span_id = getattr(event, "span_id", None)
            if not span_id:
                return None
            with handler._lock():
                return spans.get(span_id)

        def handle(self, event, **kwargs: Any) -> Any:
            span = self._span_for(event)
            if span is None or not span.is_recording():
                return None

            name = event.class_name()
            capture = _capture_content()

            # ---- request model (start events) ----
            model_dict = getattr(event, "model_dict", None)
            if isinstance(model_dict, dict):
                model = model_dict.get("model") or model_dict.get("model_name")
                if model:
                    span.set_attribute(_GEN_AI_REQUEST_MODEL, str(model))

            # ---- input messages ----
            if capture:
                messages = getattr(event, "messages", None)
                if messages and name.endswith("StartEvent"):
                    js = _messages_to_json(messages)
                    if js:
                        span.set_attribute(_GEN_AI_INPUT_MESSAGES, js)
                prompt = getattr(event, "prompt", None)
                if prompt and name.endswith("StartEvent"):
                    span.set_attribute(
                        _GEN_AI_INPUT_MESSAGES,
                        _messages_to_json(
                            [
                                type(
                                    "M",
                                    (),
                                    {"role": "user", "content": prompt},
                                )()
                            ]
                        )
                        or "",
                    )

            # ---- end events: response + usage ----
            if name.endswith("EndEvent"):
                response = getattr(event, "response", None)
                if response is not None:
                    raw = getattr(response, "raw", None)
                    usage = _extract_usage(raw) or _extract_usage(response)
                    if "input" in usage:
                        span.set_attribute(
                            _GEN_AI_USAGE_INPUT_TOKENS, usage["input"]
                        )
                    if "output" in usage:
                        span.set_attribute(
                            _GEN_AI_USAGE_OUTPUT_TOKENS, usage["output"]
                        )
                    if "total" in usage:
                        span.set_attribute(
                            _GEN_AI_USAGE_TOTAL_TOKENS, usage["total"]
                        )
                    if capture:
                        msg = getattr(response, "message", None)
                        text = None
                        if msg is not None:
                            text = getattr(msg, "content", None)
                        if text is None:
                            text = str(response)
                        span.set_attribute(
                            _GEN_AI_OUTPUT_MESSAGES,
                            _messages_to_json(
                                [
                                    type(
                                        "M",
                                        (),
                                        {"role": "assistant", "content": text},
                                    )()
                                ]
                            )
                            or "",
                        )
                # completion end carries a plain response string
                if capture:
                    messages_out = getattr(event, "messages", None)
                    if messages_out and _GEN_AI_OUTPUT_MESSAGES not in (
                        span.attributes or {}
                    ):
                        js = _messages_to_json(messages_out)
                        if js:
                            span.set_attribute(_GEN_AI_OUTPUT_MESSAGES, js)

                # embedding end: record chunk count
                chunks = getattr(event, "chunks", None)
                if chunks is not None:
                    try:
                        span.set_attribute(
                            "gen_ai.embedding.chunk_count", len(chunks)
                        )
                    except Exception:
                        pass
            return None

    return _OtelEventHandler(span_handler)


# ═══════════════════════════════════════════════════════════════════════════
# Instrumentor
# ═══════════════════════════════════════════════════════════════════════════


class LlamaIndexInstrumentor(BaseInstrumentor):
    """Instrumentor for LlamaIndex (``llama-index-core``).

    Registers a span handler and an event handler on LlamaIndex's root
    dispatcher. ``uninstrument`` removes them so no spans are produced
    afterwards.
    """

    def __init__(self):
        super().__init__()
        self._span_handler = None
        self._event_handler = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        import llama_index.core.instrumentation as instrumentation

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__, "", tracer_provider=tracer_provider
        )

        dispatcher = instrumentation.get_dispatcher()
        span_handler = _build_span_handler(tracer)
        event_handler = _build_event_handler(span_handler)

        dispatcher.add_span_handler(span_handler)
        dispatcher.add_event_handler(event_handler)

        self._span_handler = span_handler
        self._event_handler = event_handler

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import llama_index.core.instrumentation as instrumentation

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
