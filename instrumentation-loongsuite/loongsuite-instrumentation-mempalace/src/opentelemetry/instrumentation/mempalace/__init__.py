"""
OpenTelemetry MemPalace Instrumentation

Wraps 9 stable anchor points in MemPalace v3.5.x:
  - ``mempalace.mcp_server.handle_request``        (MCP SERVER span)
  - ``mempalace.mcp_server.TOOLS[*].handler``      (Tool + Memory merged span)
  - ``mempalace.searcher.search_memories`` / ``search`` (Retriever span)
  - ``mempalace.backends.chroma.ChromaCollection.{add,upsert,query,get,delete}`` (Vector sub-span)
  - ``mempalace.mcp_server._call_kg``              (Graph sub-span)
  - ``mempalace.embedding.EmbeddinggemmaONNX.__call__`` (Embedding span)
  - ``mempalace.llm_client._http_post_json``       (LLM span, main path)
  - ``mempalace.closet_llm._call_llm``             (LLM span, closet path)
  - ``mempalace.service.execute_job``              (Task span)
  - ``mempalace.miner.mine``                       (Chain span)

Sub-phase spans (Vector / Graph / Embedding) default to OFF and are gated by
``OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED``. Privacy is enforced via
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` (default off) plus a
second-pass redaction using ``mempalace.wal._WAL_REDACT_KEYS``.
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.mempalace.config import (
    is_internal_phases_enabled,
)
from opentelemetry.instrumentation.mempalace.internal._wrapper import (
    ChainWrapper,
    EmbeddingWrapper,
    GraphWrapper,
    LlmClosetWrapper,
    LlmHttpWrapper,
    McpServerWrapper,
    MemPalaceMetrics,
    RetrieverWrapper,
    TaskWrapper,
    ToolMemoryWrapper,
    VectorWrapper,
)
from opentelemetry.instrumentation.mempalace.package import _instruments
from opentelemetry.instrumentation.mempalace.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

_TRACER_NAME = "opentelemetry.instrumentation.mempalace"
_METER_NAME = "opentelemetry.instrumentation.mempalace"

# Anchor (module_path, attr_name) pairs. Each entry is feature-detected
# via hasattr at instrument time; missing anchors are skipped, not fatal.
_MCP_SERVER_MODULE = "mempalace.mcp_server"
_SEARCHER_MODULE = "mempalace.searcher"
_CHROMA_MODULE = "mempalace.backends.chroma"
_EMBEDDING_MODULE = "mempalace.embedding"
_LLM_CLIENT_MODULE = "mempalace.llm_client"
_CLOSET_LLM_MODULE = "mempalace.closet_llm"
_SERVICE_MODULE = "mempalace.service"
_MINER_MODULE = "mempalace.miner"

_VECTOR_METHODS = ("add", "upsert", "query", "get", "delete")


class MemPalaceInstrumentor(BaseInstrumentor):
    """Instrumentor for MemPalace memory platform."""

    def __init__(self):
        self._is_instrumented = False
        self._instrumented_tool_names: list[str] = []
        self._tool_originals: dict[str, Any] = {}
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if self._is_instrumented:
            return
        self._is_instrumented = True

        tracer_provider = kwargs.get("tracer_provider") or trace_api.get_tracer_provider()
        meter_provider = kwargs.get("meter_provider")

        # Handler is instantiated for parity with loongsuite-instrumentation-mem0
        # (future event-emit hookups); spans/metrics are written directly.
        try:
            ExtendedTelemetryHandler(  # noqa: F841 — kept for API parity
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
                logger_provider=kwargs.get("logger_provider"),
            )
        except Exception as e:
            logger.debug("ExtendedTelemetryHandler init failed (non-fatal): %s", e)

        tracer = trace_api.get_tracer(
            _TRACER_NAME,
            __version__,
            tracer_provider=tracer_provider,
            schema_url=Schemas.V1_28_0.value,
        )
        meter = get_meter(
            _METER_NAME,
            __version__,
            meter_provider,
            schema_url=Schemas.V1_28_0.value,
        )
        metrics = MemPalaceMetrics(meter)

        # Verify mempalace is importable before patching any anchor.
        try:
            import mempalace  # type: ignore  # noqa: F401
        except ImportError as e:
            logger.warning(
                "MemPalace not installed, skip instrumentation: %s", e
            )
            return

        self._instrument_mcp_server(tracer, metrics)
        self._instrument_tools(tracer, metrics)
        self._instrument_retriever(tracer, metrics)
        self._instrument_llm(tracer, metrics)

        if is_internal_phases_enabled():
            self._instrument_vector(tracer, metrics)
            self._instrument_graph(tracer, metrics)
            self._instrument_embedding(tracer, metrics)

        self._instrument_task(tracer, metrics)
        self._instrument_chain(tracer, metrics)

    def _uninstrument(self, **kwargs: Any) -> None:
        if not self._is_instrumented:
            return

        for target in (
            (_MCP_SERVER_MODULE, "handle_request"),
            (_SEARCHER_MODULE, "search_memories"),
            (_SEARCHER_MODULE, "search"),
            (_LLM_CLIENT_MODULE, "_http_post_json"),
            (_CLOSET_LLM_MODULE, "_call_llm"),
            (_SERVICE_MODULE, "execute_job"),
            (_MINER_MODULE, "mine"),
        ):
            try:
                unwrap(target[0], target[1])
            except Exception as e:
                logger.debug("Failed to unwrap %s.%s: %s", target[0], target[1], e)

        for method in _VECTOR_METHODS:
            try:
                unwrap(_CHROMA_MODULE, f"ChromaCollection.{method}")
            except Exception as e:
                logger.debug("Failed to unwrap ChromaCollection.%s: %s", method, e)

        for tool_name in self._instrumented_tool_names:
            try:
                import mempalace.mcp_server as _ms  # type: ignore

                entry = _ms.TOOLS.get(tool_name)
                original = self._tool_originals.pop(tool_name, None)
                if entry is not None and original is not None:
                    entry["handler"] = original
            except Exception as e:
                logger.debug("Failed to unwrap tool %s: %s", tool_name, e)
        self._instrumented_tool_names.clear()

        try:
            unwrap(_MCP_SERVER_MODULE, "_call_kg")
        except Exception as e:
            logger.debug("Failed to unwrap _call_kg: %s", e)

        try:
            unwrap(_EMBEDDING_MODULE, "EmbeddinggemmaONNX.__call__")
        except Exception as e:
            logger.debug("Failed to unwrap EmbeddinggemmaONNX.__call__: %s", e)

        self._is_instrumented = False

    # ----- per-anchor helpers ---------------------------------------------

    def _instrument_mcp_server(self, tracer, metrics) -> None:
        try:
            import mempalace.mcp_server as _ms  # type: ignore

            if not hasattr(_ms, "handle_request"):
                return
            wrap_function_wrapper(
                module=_MCP_SERVER_MODULE,
                name="handle_request",
                wrapper=McpServerWrapper(tracer, metrics),
            )
        except Exception as e:
            logger.debug("Failed to instrument mcp_server.handle_request: %s", e)

    def _instrument_tools(self, tracer, metrics) -> None:
        try:
            import mempalace.mcp_server as _ms  # type: ignore

            tools = getattr(_ms, "TOOLS", None)
            if not isinstance(tools, dict):
                return
            for name, entry in tools.items():
                try:
                    if not isinstance(entry, dict):
                        continue
                    description = safe_str(entry.get("description"))
                    handler = entry.get("handler")
                    if not callable(handler):
                        continue
                    tool_name = safe_str(name)
                    # TOOLS stores direct function references, so we must patch
                    # the dict entry in place; otherwise the existing reference
                    # bypasses the wrapper. The original is stashed for unwind.
                    original = getattr(handler, "__mempalace_original__", None) or handler
                    wrapper = ToolMemoryWrapper(tracer, metrics, tool_name, description)
                    entry["handler"] = _wrap_handler_inplace(original, wrapper)
                    self._tool_originals[tool_name] = original
                    self._instrumented_tool_names.append(tool_name)
                except Exception as e:
                    logger.debug("Failed to instrument tool %s: %s", name, e)
        except Exception as e:
            logger.debug("Failed to instrument TOOLS: %s", e)

    def _instrument_retriever(self, tracer, metrics) -> None:
        for fn_name in ("search_memories", "search"):
            try:
                import mempalace.searcher as _se  # type: ignore

                if not hasattr(_se, fn_name):
                    continue
                wrap_function_wrapper(
                    module=_SEARCHER_MODULE,
                    name=fn_name,
                    wrapper=RetrieverWrapper(tracer, metrics),
                )
            except Exception as e:
                logger.debug("Failed to instrument searcher.%s: %s", fn_name, e)

    def _instrument_llm(self, tracer, metrics) -> None:
        try:
            import mempalace.llm_client as _lc  # type: ignore

            if hasattr(_lc, "_http_post_json"):
                wrap_function_wrapper(
                    module=_LLM_CLIENT_MODULE,
                    name="_http_post_json",
                    wrapper=LlmHttpWrapper(tracer, metrics),
                )
        except Exception as e:
            logger.debug("Failed to instrument llm_client._http_post_json: %s", e)
        try:
            import mempalace.closet_llm as _cl  # type: ignore

            if hasattr(_cl, "_call_llm"):
                wrap_function_wrapper(
                    module=_CLOSET_LLM_MODULE,
                    name="_call_llm",
                    wrapper=LlmClosetWrapper(tracer, metrics),
                )
        except Exception as e:
            logger.debug("Failed to instrument closet_llm._call_llm: %s", e)

    def _instrument_vector(self, tracer, metrics) -> None:
        for method in _VECTOR_METHODS:
            try:
                import mempalace.backends.chroma as _ch  # type: ignore

                cls = getattr(_ch, "ChromaCollection", None)
                if cls is None or not hasattr(cls, method):
                    continue
                wrap_function_wrapper(
                    module=_CHROMA_MODULE,
                    name=f"ChromaCollection.{method}",
                    wrapper=VectorWrapper(tracer, metrics, method),
                )
            except Exception as e:
                logger.debug("Failed to instrument ChromaCollection.%s: %s", method, e)

    def _instrument_graph(self, tracer, metrics) -> None:
        try:
            import mempalace.mcp_server as _ms  # type: ignore

            if not hasattr(_ms, "_call_kg"):
                return
            wrap_function_wrapper(
                module=_MCP_SERVER_MODULE,
                name="_call_kg",
                wrapper=GraphWrapper(tracer, metrics),
            )
        except Exception as e:
            logger.debug("Failed to instrument _call_kg: %s", e)

    def _instrument_embedding(self, tracer, metrics) -> None:
        try:
            import mempalace.embedding as _emb  # type: ignore

            cls = getattr(_emb, "EmbeddinggemmaONNX", None)
            if cls is None or not hasattr(cls, "__call__"):
                return
            wrap_function_wrapper(
                module=_EMBEDDING_MODULE,
                name="EmbeddinggemmaONNX.__call__",
                wrapper=EmbeddingWrapper(tracer, metrics),
            )
        except Exception as e:
            logger.debug("Failed to instrument EmbeddinggemmaONNX.__call__: %s", e)

    def _instrument_task(self, tracer, metrics) -> None:
        try:
            import mempalace.service as _svc  # type: ignore

            if not hasattr(_svc, "execute_job"):
                return
            wrap_function_wrapper(
                module=_SERVICE_MODULE,
                name="execute_job",
                wrapper=TaskWrapper(tracer, metrics),
            )
        except Exception as e:
            logger.debug("Failed to instrument service.execute_job: %s", e)

    def _instrument_chain(self, tracer, metrics) -> None:
        try:
            import mempalace.miner as _mn  # type: ignore

            if hasattr(_mn, "mine"):
                wrap_function_wrapper(
                    module=_MINER_MODULE,
                    name="mine",
                    wrapper=ChainWrapper(tracer, metrics, "mine"),
                )
        except Exception as e:
            logger.debug("Failed to instrument miner.mine: %s", e)


def safe_description(entry: Any) -> str:
    try:
        if isinstance(entry, dict):
            return safe_str(entry.get("description"))
    except Exception:
        pass
    return ""


def safe_str(value: Any) -> str:
    try:
        if value is None:
            return ""
        return str(value)
    except Exception:
        return ""


def _wrap_handler_inplace(handler, wrapper):
    """Wrap a TOOLS handler in place using wrapt, preserving the original
    for later uninstrument via ``__mempalace_original__``."""
    import wrapt  # type: ignore

    original = getattr(handler, "__mempalace_original__", None) or handler

    @wrapt.function_wrapper
    def _bound(wrapped, instance, args, kwargs):
        return wrapper(wrapped, instance, args, kwargs)

    bound = _bound(original)
    try:
        bound.__mempalace_original__ = original  # type: ignore[attr-defined]
    except Exception:
        pass
    return bound
