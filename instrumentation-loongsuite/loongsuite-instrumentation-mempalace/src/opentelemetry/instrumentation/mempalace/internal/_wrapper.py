"""
Wrappers for the 9 stable MemPalace anchor points.

Span kind / attribute layout follows
/apsara/semantic-conventions/arms_docs/trace/{gen-ai,gen-ai_memory,gen-ai_mcp}.md
and execute.md §3.1.

All wrappers use raw ``tracer.start_as_current_span`` (same pattern as
loongsuite-instrumentation-mcp) so the package stays self-contained and does
not depend on the ExtendedTelemetryHandler invocation dataclasses. The
handler is still instantiated by the instrumentor for parity with mem0 and
future event-emit hookups, but the spans/metrics here are written directly.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Callable, Optional

from opentelemetry.context import attach, detach, set_value
from opentelemetry.instrumentation.mempalace.config import (
    get_attr_max_bytes,
    get_embedding_sample_rate,
    get_llm_slow_threshold,
    get_slow_add_threshold,
    get_slow_search_threshold,
    get_user_id,
    is_capture_message_content_enabled,
    is_internal_phases_enabled,
)
from opentelemetry.instrumentation.mempalace.internal._util import (
    distance_to_similarity,
    extract_filter_keys,
    extract_filter_operators,
    get_exception_type,
    infer_provider_from_endpoint,
    normalize_call_parameters,
    palace_path_hash,
    safe_float,
    safe_int,
    safe_json_dumps,
    safe_str,
    sha8,
)
from opentelemetry.instrumentation.mempalace.semconv import (
    TOOL_CHAIN_SET,
    TOOL_MEMORY_MAP,
)
from opentelemetry.instrumentation.mempalace.semconv import (
    OperationNameValues as OpNames,
)
from opentelemetry.instrumentation.mempalace.semconv import (
    SemanticAttributes as SA,
)
from opentelemetry.instrumentation.mempalace.semconv import (
    SpanKindValues as Kind,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)


def _set(span: Any, key: str, value: Any) -> None:
    """Set a span attribute, omitting None / empty values."""
    try:
        if value is None:
            return
        if isinstance(value, str) and value == "":
            return
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return
            # OTel SDK supports sequence attributes of homogeneous scalar type.
            try:
                span.set_attribute(key, list(value))
                return
            except Exception:
                span.set_attribute(key, safe_json_dumps(list(value), 8192))
                return
        if isinstance(value, dict) and len(value) == 0:
            return
        if isinstance(value, (int, float, bool)):
            span.set_attribute(key, value)
            return
        span.set_attribute(key, safe_str(value))
    except Exception as e:
        logger.debug("Failed to set attribute %s: %s", key, e)


def _set_captured(span: Any, key: str, value: Any) -> None:
    """Set a capture-gated attribute (only when capture-message-content on)."""
    if not is_capture_message_content_enabled():
        return
    text = safe_json_dumps(value, get_attr_max_bytes(), redact=True)
    _set(span, key, text)


def _record_exception(span: Any, exc: Exception) -> None:
    try:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        _set(span, SA.ERROR_TYPE, get_exception_type(exc))
    except Exception as e:
        logger.debug("Failed to record exception: %s", e)


# =============================================================================
# P0: MCP SERVER + Tool/Memory
# =============================================================================


class McpServerWrapper:
    """Wrap ``mempalace.mcp_server.handle_request`` — JSON-RPC entry span."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics"):
        self._tracer = tracer
        self._metrics = metrics

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        request = args[0] if args else kwargs.get("request") or {}
        method = ""
        tool_name = ""
        req_id = None
        try:
            if isinstance(request, dict):
                method = safe_str(request.get("method"))
                params = request.get("params") or {}
                if isinstance(params, dict):
                    tool_name = safe_str(params.get("name"))
                req_id = request.get("id")
        except Exception:
            pass

        if not method:
            # Without a method we cannot name the span meaningfully — pass through.
            return wrapped(*args, **kwargs)

        target = tool_name if tool_name else ""
        span_name = f"{method} {target}".strip()

        start = _monotonic()
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.SERVER
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.SERVER)
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.MCP_SERVER)
            _set(span, SA.MCP_METHOD_NAME, method)
            if tool_name:
                _set(span, SA.MCP_TOOL_NAME, tool_name)
            if req_id is not None:
                _set(span, SA.RPC_JSONRPC_REQUEST_ID, safe_str(req_id))
            _set(span, SA.NETWORK_PROTOCOL_VERSION, "2.0")
            _set(span, SA.NETWORK_TRANSPORT, _detect_transport())
            _set(
                span,
                SA.MCP_SESSION_ID,
                _derive_session_id(),
            )
            # mcp.arguments only when capture enabled
            try:
                params = (request or {}).get("params") or {}
                if isinstance(params, dict) and "arguments" in params:
                    _set_captured(span, SA.MCP_ARGUMENTS, params.get("arguments"))
            except Exception:
                pass

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                self._metrics.record_mcp(method, tool_name, _elapsed(start), error=True)
                raise

            # output size + rpc error code
            try:
                if isinstance(result, dict):
                    err = result.get("error")
                    if isinstance(err, dict):
                        _set(span, SA.RPC_JSONRPC_ERROR_CODE, safe_int(err.get("code")))
                    _set(span, SA.MCP_OUTPUT_SIZE, _json_size(result))
                elif result is None:
                    _set(span, SA.MCP_OUTPUT_SIZE, 0)
            except Exception:
                pass
            self._metrics.record_mcp(method, tool_name, _elapsed(start), error=False)
            return result


def _detect_transport() -> str:
    try:
        argv = " ".join(os.sys.argv or [])  # type: ignore[attr-defined]
    except Exception:
        argv = ""
    if "--http" in argv or "--serve-http" in argv or "--port" in argv:
        return "tcp"
    return "stdio"


def _derive_session_id() -> Optional[str]:
    try:
        import time as _t

        return f"pid-{os.getpid()}-{int(_t.time())}"
    except Exception:
        return None


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return 0


def _monotonic() -> float:
    import time as _t

    return _t.monotonic()


def _elapsed(start: float) -> float:
    import time as _t

    return _t.monotonic() - start


class ToolMemoryWrapper:
    """Wrap ``TOOLS[*].handler`` — merged Tool + Memory span."""

    def __init__(
        self,
        tracer: Tracer,
        metrics: "MemPalaceMetrics",
        tool_name: str,
        tool_description: str = "",
    ):
        self._tracer = tracer
        self._metrics = metrics
        self._tool_name = tool_name
        self._tool_description = tool_description

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        normalized = normalize_call_parameters(wrapped, args, kwargs)
        is_chain_tool = self._tool_name in TOOL_CHAIN_SET
        memory_op, memory_type = TOOL_MEMORY_MAP.get(self._tool_name, (None, None))

        if is_chain_tool:
            span_name = f"execute_tool {self._tool_name}"
        elif memory_op:
            span_name = f"execute_tool {self._tool_name}"
        else:
            span_name = f"execute_tool {self._tool_name}"

        start = _monotonic()
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.TOOL)
            _set(span, SA.GEN_AI_TOOL_NAME, self._tool_name)
            _set(span, SA.GEN_AI_TOOL_TYPE, "function")
            if self._tool_description:
                _set(span, SA.GEN_AI_TOOL_DESCRIPTION, self._tool_description)

            if memory_op:
                _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.MEMORY_OPERATION)
                _set(span, SA.GEN_AI_MEMORY_OPERATION, memory_op)
                if memory_type:
                    _set(span, SA.GEN_AI_MEMORY_MEMORY_TYPE, memory_type)
            else:
                _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.EXECUTE_TOOL)

            _apply_memory_attrs(span, normalized, self._tool_name, memory_op)
            _set_captured(span, SA.GEN_AI_TOOL_CALL_ARGUMENTS, normalized)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                if memory_op:
                    self._metrics.record_memory(
                        memory_op, _elapsed(start), error=True
                    )
                raise

            _apply_result_attrs(span, result, self._tool_name, memory_op)
            _set_captured(span, SA.GEN_AI_TOOL_CALL_RESULT, result)
            if memory_op:
                self._metrics.record_memory(memory_op, _elapsed(start), error=False)
            return result


def _apply_memory_attrs(
    span: Any, normalized: dict, tool_name: str, memory_op: Optional[str]
) -> None:
    if not memory_op:
        return
    user_id = get_user_id()
    if user_id:
        _set(span, SA.GEN_AI_MEMORY_USER_ID, user_id)
    agent_id = normalized.get("agent") or normalized.get("agent_name") or normalized.get("added_by")
    if agent_id:
        _set(span, SA.GEN_AI_MEMORY_AGENT_ID, safe_str(agent_id))
    memory_id = (
        normalized.get("drawer_id")
        or normalized.get("triple_id")
        or normalized.get("memory_id")
    )
    if memory_id:
        _set(span, SA.GEN_AI_MEMORY_ID, safe_str(memory_id))
    limit = normalized.get("limit") or normalized.get("n_results")
    if limit is not None:
        _set(span, SA.GEN_AI_MEMORY_LIMIT, safe_int(limit))
        _set(span, SA.GEN_AI_MEMORY_TOP_K, safe_int(limit))
    threshold = normalized.get("threshold") or normalized.get("max_distance")
    if threshold is not None:
        _set(span, SA.GEN_AI_MEMORY_THRESHOLD, safe_float(threshold))
    filter_keys: list[str] = []
    for key in ("wing", "room", "source_file"):
        if normalized.get(key):
            filter_keys.append(key)
    where = normalized.get("where")
    filter_keys.extend(extract_filter_keys(where))
    if filter_keys:
        _set(span, SA.GEN_AI_MEMORY_FILTER_KEYS, filter_keys)
    # hybrid rerank always on in MemPalace searcher
    _set(span, SA.GEN_AI_MEMORY_RERANK, True)
    palace = normalized.get("palace_path")
    if palace:
        _set(span, SA.SERVER_ADDRESS, sha8(safe_str(palace)))


def _apply_result_attrs(
    span: Any, result: Any, tool_name: str, memory_op: Optional[str]
) -> None:
    if not memory_op or result is None:
        return
    try:
        count: Optional[int] = None
        if isinstance(result, dict):
            for key in ("count", "total", "result_count"):
                v = result.get(key)
                if isinstance(v, int):
                    count = v
                    break
            if count is None:
                for key in ("drawers", "items", "results", "triples", "rooms", "wings"):
                    v = result.get(key)
                    if isinstance(v, list):
                        count = len(v)
                        break
            # gen_ai.memory.id from result for add operations (no input id)
            if memory_op == "add":
                rid = result.get("drawer_id") or result.get("triple_id")
                if rid:
                    _set(span, SA.GEN_AI_MEMORY_ID, safe_str(rid))
        elif isinstance(result, list):
            count = len(result)
        if count is not None:
            _set(span, SA.GEN_AI_MEMORY_RESULT_COUNT, count)
    except Exception as e:
        logger.debug("Failed to extract result attrs: %s", e)


# =============================================================================
# P1: Retriever + LLM
# =============================================================================


class RetrieverWrapper:
    """Wrap ``mempalace.searcher.search_memories`` (and ``search``)."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics"):
        self._tracer = tracer
        self._metrics = metrics

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        normalized = normalize_call_parameters(wrapped, args, kwargs)
        palace_path = safe_str(normalized.get("palace_path"))
        ds_id = palace_path_hash(palace_path) or "mempalace"
        span_name = f"retrieval {ds_id}"

        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.RETRIEVER)
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.RETRIEVAL)
            _set(span, SA.GEN_AI_DATA_SOURCE_ID, ds_id)
            _set(span, SA.GEN_AI_PROVIDER_NAME, "chroma")
            try:
                from mempalace import embedding as _emb  # type: ignore

                _set(span, SA.GEN_AI_REQUEST_MODEL, _safe_call(_emb.current_model_name))
            except Exception:
                pass
            n_results = normalized.get("n_results")
            if n_results is not None:
                _set(span, SA.GEN_AI_REQUEST_TOP_K, safe_float(n_results))
            _set_captured(
                span, SA.GEN_AI_RETRIEVAL_QUERY_TEXT, normalized.get("query")
            )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise

            _apply_retrieval_documents(span, result)
            return result


def _apply_retrieval_documents(span: Any, result: Any) -> None:
    if not is_capture_message_content_enabled() or not isinstance(result, dict):
        return
    try:
        ids = result.get("ids") or []
        distances = result.get("distances") or []
        docs: list[Any] = []
        if ids and isinstance(ids, list) and isinstance(ids[0], list):
            ids = ids[0]
            distances = distances[0] if distances and isinstance(distances[0], list) else distances
        for i, doc_id in enumerate(ids):
            score = None
            if i < len(distances):
                score = distance_to_similarity(distances[i])
            docs.append({"id": doc_id, "score": score})
        if docs:
            _set(span, SA.GEN_AI_RETRIEVAL_DOCUMENTS, docs)
    except Exception as e:
        logger.debug("Failed to extract retrieval docs: %s", e)


def _safe_call(fn: Callable, *args: Any) -> Optional[str]:
    try:
        return safe_str(fn(*args))
    except Exception:
        return None


class LlmHttpWrapper:
    """Wrap ``mempalace.llm_client._http_post_json`` — LLM span (main path)."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics"):
        self._tracer = tracer
        self._metrics = metrics

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        url = args[0] if len(args) > 0 else kwargs.get("url")
        body = args[1] if len(args) > 1 else kwargs.get("body")
        model = ""
        if isinstance(body, dict):
            model = safe_str(body.get("model"))
        provider = infer_provider_from_endpoint(safe_str(url)) or "unknown"
        span_name = f"chat {model}" if model else "chat"

        start = _monotonic()
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.LLM)
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.CHAT)
            _set(span, SA.GEN_AI_PROVIDER_NAME, provider)
            _set(span, SA.GEN_AI_REQUEST_MODEL, model)
            _set_captured(span, SA.GEN_AI_INPUT_MESSAGES, body.get("messages") if isinstance(body, dict) else None)

            try:
                data = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                self._metrics.record_llm(provider, model, _elapsed(start), error=True)
                raise

            _apply_llm_response(span, data)
            self._metrics.record_llm(provider, model, _elapsed(start), error=False)
            if isinstance(data, dict):
                self._metrics.record_llm_tokens(
                    model,
                    (data.get("usage") or {}).get("prompt_tokens") if isinstance(data.get("usage"), dict) else None,
                    (data.get("usage") or {}).get("completion_tokens") if isinstance(data.get("usage"), dict) else None,
                )
            return data


class LlmClosetWrapper:
    """Wrap ``mempalace.closet_llm._call_llm`` — LLM span (closet path).

    The wrapped function returns ``(parsed, usage_dict)`` and retries 3x
    internally. Retries are not visible at this signature boundary, so the
    span is a single LLM span covering all retries.
    """

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics"):
        self._tracer = tracer
        self._metrics = metrics

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        cfg = args[0] if len(args) > 0 else kwargs.get("cfg")
        model = _safe_get_attr(cfg, "model") or ""
        endpoint = _safe_get_attr(cfg, "endpoint")
        provider = infer_provider_from_endpoint(endpoint) or "unknown"
        span_name = f"chat {model}" if model else "chat"

        start = _monotonic()
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.LLM)
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.CHAT)
            _set(span, SA.GEN_AI_PROVIDER_NAME, provider)
            _set(span, SA.GEN_AI_REQUEST_MODEL, model)
            max_tokens = _safe_get_attr(cfg, "max_tokens")
            if max_tokens:
                _set(span, SA.GEN_AI_REQUEST_MAX_TOKENS, safe_int(max_tokens))

            try:
                parsed, usage = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                self._metrics.record_llm(provider, model, _elapsed(start), error=True)
                raise

            _apply_llm_usage(span, usage)
            self._metrics.record_llm(provider, model, _elapsed(start), error=False)
            if isinstance(usage, dict):
                self._metrics.record_llm_tokens(
                    model,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                )
            return parsed, usage


def _safe_get_attr(obj: Any, name: str) -> Optional[str]:
    try:
        if obj is None:
            return None
        v = getattr(obj, name, None)
        return safe_str(v) if v is not None else None
    except Exception:
        return None


def _apply_llm_response(span: Any, data: Any) -> None:
    if not isinstance(data, dict):
        return
    _set(span, SA.GEN_AI_RESPONSE_MODEL, data.get("model"))
    try:
        choices = data.get("choices") or []
        if choices and isinstance(choices, list):
            fr = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
            if fr:
                _set(span, SA.GEN_AI_RESPONSE_FINISH_REASONS, [fr])
    except Exception:
        pass
    _set_captured(span, SA.GEN_AI_OUTPUT_MESSAGES, _extract_output_message(data))
    _apply_llm_usage(span, data.get("usage"))


def _extract_output_message(data: Any) -> Any:
    try:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return [choices[0].get("message")]
    except Exception:
        pass
    return None


def _apply_llm_usage(span: Any, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    _set(span, SA.GEN_AI_USAGE_INPUT_TOKENS, safe_int(usage.get("prompt_tokens")))
    _set(span, SA.GEN_AI_USAGE_OUTPUT_TOKENS, safe_int(usage.get("completion_tokens")))
    total = usage.get("total_tokens")
    if total is None:
        pi, po = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if pi is not None and po is not None:
            total = safe_int(pi) + safe_int(po)
    _set(span, SA.GEN_AI_USAGE_TOTAL_TOKENS, safe_int(total))


# =============================================================================
# P2: Vector + Graph sub-phases (gated by OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED)
# =============================================================================


class _SuppressContext:
    """Attach _SUPPRESS_INSTRUMENTATION_KEY for the duration of a block."""

    def __enter__(self):
        self._token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        return self

    def __exit__(self, *exc):
        detach(self._token)
        return False


class VectorWrapper:
    """Wrap ``ChromaCollection.{add,upsert,query,get,delete}``."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics", method_name: str):
        self._tracer = tracer
        self._metrics = metrics
        self._method_name = method_name

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        if not is_internal_phases_enabled():
            return wrapped(*args, **kwargs)

        span_name = f"chroma.{self._method_name}"
        start = _monotonic()
        with _SuppressContext(), self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT
        ) as span:
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.MEMORY_OPERATION)
            _set(span, SA.GEN_AI_MEMORY_INNER_NAME, "vector")
            _set(span, SA.GEN_AI_MEMORY_DATA_SOURCE_TYPE, "chroma")
            _set(span, SA.GEN_AI_MEMORY_VECTOR_METHOD, self._method_name)
            _set(
                span,
                SA.GEN_AI_MEMORY_VECTOR_COLLECTION,
                _safe_get_attr(instance, "name") or _safe_get_attr(instance, "_collection.name"),
            )
            palace = _safe_get_attr(instance, "_palace_path")
            if palace:
                _set(span, SA.GEN_AI_MEMORY_DATA_SOURCE_URL, sha8(palace))
            n_results = kwargs.get("n_results")
            if n_results is not None:
                _set(span, SA.GEN_AI_MEMORY_VECTOR_LIMIT, safe_int(n_results))
            where = kwargs.get("where")
            keys = extract_filter_keys(where)
            if keys:
                _set(span, SA.GEN_AI_MEMORY_VECTOR_FILTERS_KEYS, keys)
            ops = extract_filter_operators(where)
            if ops:
                _set(span, SA.GEN_AI_MEMORY_VECTOR_FILTERS_OPERATORS, ops)
            try:
                from mempalace import embedding as _emb  # type: ignore

                _set(span, SA.GEN_AI_MEMORY_VECTOR_EMBEDDING_DIMS, _safe_call_int(_emb.probe_dimension))
            except Exception:
                pass
            _set(span, SA.GEN_AI_MEMORY_VECTOR_METRIC_TYPE, _detect_metric(instance))

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                self._metrics.record_inner(
                    "vector", self._method_name, _elapsed(start), error=True
                )
                raise

            _set(span, SA.GEN_AI_MEMORY_VECTOR_RESULT_COUNT, _vector_result_count(result, self._method_name))
            self._metrics.record_inner(
                "vector", self._method_name, _elapsed(start), error=False
            )
            return result


def _safe_call_int(fn: Callable, *args: Any) -> Optional[int]:
    try:
        return safe_int(fn(*args))
    except Exception:
        return None


def _detect_metric(instance: Any) -> str:
    try:
        col = getattr(instance, "_collection", None)
        if col is not None:
            meta = getattr(col, "metadata", None) or {}
            if isinstance(meta, dict):
                return safe_str(meta.get("hnsw:space")) or "cosine"
    except Exception:
        pass
    return "cosine"


def _vector_result_count(result: Any, method_name: str) -> Optional[int]:
    try:
        if method_name == "query":
            if isinstance(result, dict):
                ids = result.get("ids") or []
                if ids and isinstance(ids[0], list):
                    return len(ids[0])
                return len(ids)
        if method_name == "get":
            if isinstance(result, dict):
                ids = result.get("ids") or []
                return len(ids) if isinstance(ids, list) else None
        if hasattr(result, "ids"):
            return len(getattr(result, "ids") or [])
    except Exception:
        pass
    return None


class GraphWrapper:
    """Wrap ``mempalace.mcp_server._call_kg`` — Graph sub-phase span."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics"):
        self._tracer = tracer
        self._metrics = metrics

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        if not is_internal_phases_enabled():
            return wrapped(*args, **kwargs)

        op = args[0] if args else kwargs.get("op")
        op_name = safe_str(op) if not callable(op) else "call"
        span_name = f"sqlite.{op_name or 'call'}"
        start = _monotonic()
        with _SuppressContext(), self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT
        ) as span:
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.MEMORY_OPERATION)
            _set(span, SA.GEN_AI_MEMORY_INNER_NAME, "graph")
            _set(span, SA.GEN_AI_MEMORY_DATA_SOURCE_TYPE, "sqlite")
            _set(span, SA.GEN_AI_MEMORY_GRAPH_METHOD, op_name)
            try:
                from mempalace import mcp_server as _ms  # type: ignore

                path_fn = getattr(_ms, "_resolve_kg_path", None)
                if callable(path_fn):
                    _set(span, SA.GEN_AI_MEMORY_DATA_SOURCE_URL, sha8(safe_str(path_fn())))
            except Exception:
                pass

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                self._metrics.record_inner(
                    "graph", op_name, _elapsed(start), error=True
                )
                raise

            _set(span, SA.GEN_AI_MEMORY_GRAPH_RESULT_COUNT, _graph_result_count(result))
            self._metrics.record_inner(
                "graph", op_name, _elapsed(start), error=False
            )
            return result


def _graph_result_count(result: Any) -> Optional[int]:
    try:
        if isinstance(result, list):
            return len(result)
        if isinstance(result, dict):
            for key in ("triples", "results", "entities", "rows"):
                v = result.get(key)
                if isinstance(v, list):
                    return len(v)
    except Exception:
        pass
    return None


# =============================================================================
# P3: Embedding + Task + Chain
# =============================================================================


class EmbeddingWrapper:
    """Wrap ``mempalace.embedding.EmbeddinggemmaONNX.__call__``."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics"):
        self._tracer = tracer
        self._metrics = metrics

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        if not is_internal_phases_enabled():
            return wrapped(*args, **kwargs)
        # Sampling: 10% default, DEBUG level (= 5) → 100%
        rate = get_embedding_sample_rate()
        if logger.isEnabledFor(5):  # DEBUG
            rate = 1.0
        if rate <= 0.0 or random.random() > rate:
            return wrapped(*args, **kwargs)

        model = ""
        try:
            from mempalace import embedding as _emb  # type: ignore

            model = safe_str(_emb.current_model_name())
        except Exception:
            pass
        span_name = f"embeddings {model}" if model else "embeddings"
        with _SuppressContext(), self._tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.EMBEDDING)
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.EMBEDDINGS)
            _set(span, SA.GEN_AI_PROVIDER_NAME, "onnx")
            if model:
                _set(span, SA.GEN_AI_REQUEST_MODEL, model)
            try:
                from mempalace import embedding as _emb  # type: ignore

                _set(span, SA.GEN_AI_EMBEDDINGS_DIMENSION_COUNT, _safe_call_int(_emb.probe_dimension))
            except Exception:
                pass

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise
            return result


class TaskWrapper:
    """Wrap ``mempalace.service.execute_job``."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics"):
        self._tracer = tracer
        self._metrics = metrics

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        kind = args[0] if len(args) > 0 else kwargs.get("kind")
        payload = args[1] if len(args) > 1 else kwargs.get("payload")
        span_name = f"run_task {kind}"

        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.TASK)
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.RUN_TASK)
            _set(span, SA.GEN_AI_TASK_NAME, safe_str(kind))
            _set_captured(span, SA.INPUT_VALUE, payload)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise
            _set_captured(span, SA.OUTPUT_VALUE, result)
            return result


class ChainWrapper:
    """Wrap ``mempalace.miner.mine`` (and ``service.run_mine``)."""

    def __init__(self, tracer: Tracer, metrics: "MemPalaceMetrics", name: str = "mine"):
        self._tracer = tracer
        self._metrics = metrics
        self._name = name

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict):
        normalized = normalize_call_parameters(wrapped, args, kwargs)
        span_name = f"chain {self._name}"

        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL
        ) as span:
            _set(span, SA.GEN_AI_SPAN_KIND, Kind.CHAIN)
            _set(span, SA.GEN_AI_OPERATION_NAME, OpNames.WORKFLOW)
            _set_captured(
                span,
                SA.INPUT_VALUE,
                {
                    k: normalized.get(k)
                    for k in (
                        "project_dir",
                        "palace_path",
                        "wing_override",
                        "wing",
                        "agent",
                        "limit",
                        "dry_run",
                        "extract",
                        "mode",
                    )
                    if k in normalized
                },
            )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise
            _set_captured(span, SA.OUTPUT_VALUE, result)
            return result


# =============================================================================
# Metrics
# =============================================================================


class MemPalaceMetrics:
    def __init__(self, meter):
        self._meter = meter
        self._mcp_duration = meter.create_gauge(
            name="mcp_server_operation_duration",
            description="Duration of MCP server operations (handle_request).",
            unit="s",
        )
        self._mem_duration = meter.create_gauge(
            name="gen_ai_memory_operation_duration",
            description="Duration of MemPalace memory operations.",
            unit="s",
        )
        self._mem_count = meter.create_counter(
            name="gen_ai_memory_operation_count",
            description="Number of MemPalace memory operations.",
        )
        self._mem_error = meter.create_counter(
            name="gen_ai_memory_operation_error_count",
            description="Number of failed MemPalace memory operations.",
        )
        self._mem_slow = meter.create_counter(
            name="gen_ai_memory_operation_slow_count",
            description="Number of slow MemPalace memory operations.",
        )
        self._inner_duration = meter.create_gauge(
            name="gen_ai_memory_inner_operation_duration",
            description="Duration of MemPalace inner (vector/graph) operations.",
            unit="s",
        )
        self._inner_count = meter.create_counter(
            name="gen_ai_memory_inner_operation_count",
            description="Number of MemPalace inner (vector/graph) operations.",
        )
        self._inner_error = meter.create_counter(
            name="gen_ai_memory_inner_operation_error_count",
            description="Number of failed MemPalace inner operations.",
        )
        self._inner_slow = meter.create_counter(
            name="gen_ai_memory_inner_operation_slow_count",
            description="Number of slow MemPalace inner operations.",
        )
        self._llm_calls = meter.create_counter(
            name="genai_calls_count",
            description="Number of LLM calls.",
        )
        self._llm_duration = meter.create_histogram(
            name="genai_calls_duration_seconds",
            description="Duration of LLM calls.",
            unit="s",
        )
        self._llm_error = meter.create_counter(
            name="genai_calls_error_count",
            description="Number of failed LLM calls.",
        )
        self._llm_slow = meter.create_counter(
            name="genai_calls_slow_count",
            description="Number of slow LLM calls.",
        )
        self._llm_tokens = meter.create_counter(
            name="genai_llm_usage_tokens",
            description="Number of LLM usage tokens.",
        )

    def record_mcp(
        self, method: str, tool: str, duration: float, error: bool
    ) -> None:
        try:
            attrs = {SA.MCP_METHOD_NAME: method}
            if tool:
                attrs[SA.MCP_TOOL_NAME] = tool
            if error:
                attrs[SA.ERROR_TYPE] = "error"
            self._mcp_duration.record(max(0.0, duration), attrs)
        except Exception as e:
            logger.debug("Failed to record mcp metric: %s", e)

    def record_memory(
        self, operation: str, duration: float, error: bool
    ) -> None:
        try:
            attrs = {SA.GEN_AI_MEMORY_OPERATION: operation}
            self._mem_duration.record(max(0.0, duration), attrs)
            if error:
                self._mem_error.add(1, attrs)
                return
            self._mem_count.add(1, attrs)
            threshold = (
                get_slow_search_threshold()
                if operation == "search"
                else get_slow_add_threshold()
            )
            if duration > threshold:
                self._mem_slow.add(1, attrs)
        except Exception as e:
            logger.debug("Failed to record memory metric: %s", e)

    def record_inner(
        self, inner_name: str, operation: str, duration: float, error: bool
    ) -> None:
        try:
            attrs = {
                SA.GEN_AI_MEMORY_INNER_NAME: inner_name,
                SA.GEN_AI_MEMORY_OPERATION: operation,
            }
            self._inner_duration.record(max(0.0, duration), attrs)
            if error:
                self._inner_error.add(1, attrs)
                return
            self._inner_count.add(1, attrs)
            threshold = 1.0 if inner_name == "vector" else 0.5
            if duration > threshold:
                self._inner_slow.add(1, attrs)
        except Exception as e:
            logger.debug("Failed to record inner metric: %s", e)

    def record_llm(
        self, provider: str, model: str, duration: float, error: bool
    ) -> None:
        try:
            attrs = {"modelName": model or "unknown", "spanKind": Kind.LLM}
            self._llm_duration.record(max(0.0, duration), attrs)
            if error:
                self._llm_error.add(1, {**attrs, "errorType": "error"})
                return
            self._llm_calls.add(1, attrs)
            if duration > get_llm_slow_threshold():
                self._llm_slow.add(1, attrs)
        except Exception as e:
            logger.debug("Failed to record llm metric: %s", e)

    def record_llm_tokens(
        self, model: str, input_tokens: Optional[int], output_tokens: Optional[int]
    ) -> None:
        try:
            base = {"modelName": model or "unknown"}
            if input_tokens:
                self._llm_tokens.add(int(input_tokens), {**base, "usageType": "input"})
            if output_tokens:
                self._llm_tokens.add(int(output_tokens), {**base, "usageType": "output"})
        except Exception as e:
            logger.debug("Failed to record llm tokens: %s", e)
