"""``MAFSemanticProcessor`` — a ``SpanProcessor`` that enriches Microsoft Agent
Framework's native OTel spans to align with the ARMS GenAI semantic conventions
(``/apsara/semantic-conventions/arms_docs/trace/gen-ai.md``).

MAF already emits OTel spans via its telemetry layers (``ChatTelemetryLayer``,
``EmbeddingTelemetryLayer``, ``AgentTelemetryLayer``, workflow span helpers).
This processor:

1. Injects ``gen_ai.span.kind`` (MAF does not set it).
2. Renames MAF private-prefix attributes (``workflow.id`` → ``gen_ai.workflow.id``,
   ``executor.id`` → ``gen_ai.task.name`` …) per the local mapping table in
   :mod:`semantic_conventions`.
3. Backfills ``gen_ai.response.time_to_first_token`` from the first streaming
   chunk event timestamp.
4. Normalizes ``gen_ai.provider.name`` (``azure_openai`` → ``openai``).
5. Sets ``StatusCode.OK`` on successful spans (MAF only sets ``ERROR``).
6. Aggregates 6 ARMS gauges (``genai_calls_count`` / ``genai_calls_duration_seconds``
   / ``genai_calls_error_count`` / ``genai_calls_slow_count`` / ``genai_llm_first_token_seconds``
   / ``genai_llm_usage_tokens``) in-process, exposed via observable gauges.

Truncation / PII helpers are reused from ``opentelemetry.util.genai.utils``
(``gen_ai_json_dumps``) — aligned with the pattern at
``instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/.../span_processor.py:27``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from opentelemetry.context import Context
from opentelemetry.metrics import ObservableGauge, get_meter
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider  # noqa: F401  (typing hint)
from opentelemetry.trace import Span as OtelSpan, Status, StatusCode
from opentelemetry.trace.span import TraceState  # noqa: F401
from opentelemetry.util.genai.utils import gen_ai_json_dumps
from opentelemetry.util.types import AttributeValue

from .semantic_conventions import (
    ERROR_TYPE,
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REACT_ROUND,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TTFT,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USER_TTFT,
    GenAIOperation,
    GenAISpanKind,
    MAF_ATTR_RENAME_MAP,
    MAF_SPAN_NAME_PREFIXES,
    PROVIDER_NAME_NORMALIZE,
)

logger = logging.getLogger(__name__)

# Span-name prefixes emitted by MAF (observability.py). Used to classify a span
# when ``gen_ai.operation.name`` is not already set (e.g. workflow spans).
_AGENT_PREFIX = "invoke_agent"
_CHAT_PREFIX = "chat "
_EMBEDDING_PREFIX = "embeddings "
_TOOL_PREFIX = "execute_tool "
_REACT_STEP_NAME = "react step"
_WORKFLOW_RUN = "workflow.run"
_WORKFLOW_BUILD = "workflow.build"
_MESSAGE_SEND = "message.send"
_EXECUTOR_PROCESS = "executor.process"
_EDGE_GROUP_PROCESS = "edge_group.process"


def _attr_value(span: Any, key: str) -> Any:
    """Read an attribute from a live Span or ReadableSpan, tolerating both."""
    attrs = getattr(span, "_attributes", None)
    if attrs is not None:
        try:
            return attrs.get(key)
        except Exception:
            pass
    try:
        return span.attributes.get(key)  # type: ignore[union-attr]
    except Exception:
        return None


def _safe_dumps(obj: Any) -> Optional[str]:
    """Serialize ``obj`` via the shared ``gen_ai_json_dumps`` helper.

    Reuses the truncation / PII / canonicalization logic from
    ``opentelemetry.util.genai.utils`` (the same path as
    ``openai-agents-v2/span_processor.py:27`` / ``safe_json_dumps`` at
    ``:370``). Falls back to ``str(obj)`` if JSON serialization fails.
    """
    try:
        return gen_ai_json_dumps(obj)
    except (TypeError, ValueError):
        return str(obj)


_PRIMITIVE_ATTR_TYPES = (str, bool, int, float)


def _coerce_attr_value(value: Any) -> Any:
    """Coerce ``value`` to an OTel-compatible ``AttributeValue``.

    OTel attributes accept ``str | bool | int | float`` and sequences of those.
    MAF sometimes writes dict / nested-list values under its private prefixes
    (``workflow.definition``, ``message.payload-type`` contents …) which the
    SDK would silently drop. We dump those to a JSON string via the shared
    ``gen_ai_json_dumps`` helper so the data still reaches exporters.
    """
    if value is None:
        return None
    if isinstance(value, _PRIMITIVE_ATTR_TYPES):
        return value
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_attr_value(v) for v in value]
        if all(isinstance(v, _PRIMITIVE_ATTR_TYPES) for v in coerced):
            return coerced
        return _safe_dumps(value)
    if isinstance(value, dict):
        return _safe_dumps(value)
    return _safe_dumps(value)


def _set_attr(live_span: OtelSpan, key: str, value: Any) -> None:
    """Write an attribute onto the span even after it has been ended.

    The OTel SDK's public ``Span.set_attribute`` silently no-ops once
    ``Span.end()`` has been called (``is_recording()`` is False by the time
    ``on_end`` runs). To enrich spans in ``on_end`` we mutate the live span's
    private ``_attributes`` dict directly under its lock — same approach as
    the OpenInference processor. Safe because ``on_end`` runs synchronously
    on the calling thread after the span has ended, so no concurrent writers
    remain.
    """
    coerced = _coerce_attr_value(value)
    if coerced is None:
        return
    attrs = getattr(live_span, "_attributes", None)
    lock = getattr(live_span, "_lock", None)
    if attrs is None:
        try:
            live_span.set_attribute(key, coerced)
        except Exception:
            pass
        return
    try:
        if lock is not None:
            with lock:
                attrs[key] = coerced
        else:
            attrs[key] = coerced
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("set_attribute(%s) failed: %s", key, exc)


def _rename_maf_attrs(live_span: OtelSpan, readable: Any) -> list[str]:
    """Rename MAF-private attributes to ``gen_ai.*`` canonical keys.

    Returns the list of canonical keys that were written. The original MAF key
    is best-effort removed from ``_attributes`` (private API; guarded by
    try/except). Removal failures are harmless — the canonical key is what
    downstream platforms read.
    """
    renamed: list[str] = []
    attrs = getattr(live_span, "_attributes", None)
    lock = getattr(live_span, "_lock", None)
    for old_key, new_key in MAF_ATTR_RENAME_MAP.items():
        if attrs is not None:
            try:
                if lock is not None:
                    with lock:
                        present = old_key in attrs
                        value = attrs.get(old_key) if present else None
                else:
                    present = old_key in attrs
                    value = attrs.get(old_key) if present else None
            except Exception:
                continue
            if not present:
                continue
            _set_attr(live_span, new_key, value)
            renamed.append(new_key)
            # best-effort removal of the old (private) key
            try:
                if lock is not None:
                    with lock:
                        attrs.pop(old_key, None)
                else:
                    attrs.pop(old_key, None)
            except Exception:
                pass
        else:
            # No live attrs; fall back to readable.attributes (read-only)
            readable_attrs = getattr(readable, "attributes", None) or {}
            if old_key in readable_attrs:
                _set_attr(live_span, new_key, readable_attrs[old_key])
                renamed.append(new_key)
    return renamed


def _normalize_provider(value: Any) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, str):
        value = str(value)
    return PROVIDER_NAME_NORMALIZE.get(value, value)


def _classify_span(
    name: str, operation: Optional[str], readable: Any
) -> Tuple[str, str]:
    """Return ``(span_kind, operation_name)`` for a span.

    Classification priority:
    1. Existing ``gen_ai.operation.name`` (set by MAF for chat/embeddings/tool/agent).
    2. Span-name prefix matching (workflow spans have no operation.name from MAF).
    3. ``react step`` literal name (emitted by our react_step patch).
    """
    if operation:
        op = operation
    else:
        op = ""
        for prefix, mapped in MAF_SPAN_NAME_PREFIXES.items():
            if name.startswith(prefix):
                op = mapped
                break
        if not op:
            if name == _REACT_STEP_NAME:
                op = GenAIOperation.REACT
            else:
                op = GenAIOperation.WORKFLOW  # safe default for MAF internal spans

    if op == GenAIOperation.CHAT or op == GenAIOperation.TEXT_COMPLETION or op == GenAIOperation.GENERATE_CONTENT:
        return GenAISpanKind.LLM, op
    if op == GenAIOperation.EMBEDDINGS:
        return GenAISpanKind.EMBEDDING, op
    if op == GenAIOperation.EXECUTE_TOOL:
        return GenAISpanKind.TOOL, op
    if op == GenAIOperation.CREATE_AGENT:
        return GenAISpanKind.AGENT, op
    if op == GenAIOperation.INVOKE_AGENT:
        return GenAISpanKind.AGENT, op
    if op == GenAIOperation.REACT:
        return GenAISpanKind.STEP, op
    if op == GenAIOperation.RETRIEVAL:
        return GenAISpanKind.RETRIEVER, op
    if op == GenAIOperation.WORKFLOW or op == GenAIOperation.TASK:
        # ``executor.process`` splits by ``executor.type``:
        #   FunctionExecutor  -> TASK
        #   AgentExecutor     -> AGENT (kind) + invoke_agent (op)
        #   anything else     -> CHAIN
        if name.startswith(_EXECUTOR_PROCESS):
            executor_type = _attr_value(readable, "executor.type")
            if isinstance(executor_type, str):
                et = executor_type.lower()
                if "function" in et:
                    return GenAISpanKind.TASK, GenAIOperation.TASK
                if "agent" in et:
                    return GenAISpanKind.AGENT, GenAIOperation.INVOKE_AGENT
        return GenAISpanKind.CHAIN, op
    if op == GenAIOperation.MCP:
        return GenAISpanKind.CLIENT, op
    return GenAISpanKind.CHAIN, op


def _ttft_from_events(readable: Any) -> Optional[int]:
    """Backfill ``gen_ai.response.time_to_first_token`` (ns) from the first
    streaming chunk event timestamp.

    MAF emits streaming chunks as span events; the first event's timestamp
    minus the span start time is the TTFT.
    """
    events = getattr(readable, "events", None) or ()
    if not events:
        return None
    start_time = getattr(readable, "start_time", None)
    first_ts = None
    for ev in events:
        ts = getattr(ev, "timestamp", None)
        if ts is None:
            ts = ev.get("timestamp") if isinstance(ev, dict) else None
        if ts is not None:
            first_ts = ts
            break
    if first_ts is None or start_time is None:
        return None
    try:
        return int(first_ts - start_time)
    except (TypeError, ValueError):
        return None


# ---------- Metrics aggregation ----------


class _Counters:
    """In-process counters keyed by ``(model, span_kind[, usage_type])``.

    Designed to be cheap to update in ``on_end`` and read out by observable
    gauge callbacks. Single-process only — multi-process deployments need
    per-process reporters (documented in README).
    """

    __slots__ = (
        "calls_count",
        "calls_duration_ns_sum",
        "calls_error_count",
        "calls_slow_count",
        "llm_first_token_ns_sum",
        "llm_first_token_count",
        "llm_usage_input_tokens",
        "llm_usage_output_tokens",
    )

    def __init__(self) -> None:
        self.calls_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.calls_duration_ns_sum: Dict[Tuple[str, str], int] = defaultdict(int)
        self.calls_error_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.calls_slow_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.llm_first_token_ns_sum: Dict[Tuple[str, str], int] = defaultdict(int)
        self.llm_first_token_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.llm_usage_input_tokens: Dict[Tuple[str, str], int] = defaultdict(int)
        self.llm_usage_output_tokens: Dict[Tuple[str, str], int] = defaultdict(int)


class MAFSemanticProcessor(SpanProcessor):
    """SpanProcessor that injects ARMS GenAI semantic conventions into MAF spans."""

    def __init__(
        self,
        meter_provider: Any = None,
        slow_threshold_ms: int = 1000,
        metrics_enabled: bool = True,
        capture_sensitive_data: bool = False,
    ) -> None:
        self._live_spans: Dict[str, OtelSpan] = {}
        self._span_parents: Dict[str, Optional[str]] = {}
        self._slow_threshold_ns = int(slow_threshold_ms) * 1_000_000
        self._capture_sensitive = capture_sensitive_data
        self._counters = _Counters()
        self._meter = None
        self._gauges: list[ObservableGauge] = []
        self._metrics_enabled = metrics_enabled
        if metrics_enabled:
            try:
                self._init_metrics(meter_provider)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("MAF metrics disabled: %s", exc)
                self._metrics_enabled = False

    # ---- metrics ----
    def _init_metrics(self, meter_provider: Any) -> None:
        self._meter = get_meter(
            "opentelemetry.instrumentation.microsoft_agent_framework",
            meter_provider=meter_provider,
        )
        c = self._counters

        def _calls_cb(options):
            for (model, kind), count in c.calls_count.items():
                yield _obs(
                    count,
                    {"modelName": model or "unknown", "spanKind": kind},
                )

        def _duration_cb(options):
            for (model, kind), total in c.calls_duration_ns_sum.items():
                count = max(c.calls_count.get((model, kind), 0), 1)
                yield _obs(
                    total / count / 1e9,
                    {"modelName": model or "unknown", "spanKind": kind},
                )

        def _error_cb(options):
            for (model, kind), count in c.calls_error_count.items():
                yield _obs(
                    count,
                    {"modelName": model or "unknown", "spanKind": kind},
                )

        def _slow_cb(options):
            for (model, kind), count in c.calls_slow_count.items():
                yield _obs(
                    count,
                    {"modelName": model or "unknown", "spanKind": kind},
                )

        def _ttft_cb(options):
            for (model, kind), total in c.llm_first_token_ns_sum.items():
                count = max(c.llm_first_token_count.get((model, kind), 0), 1)
                yield _obs(
                    total / count / 1e9,
                    {"modelName": model or "unknown", "spanKind": kind},
                )

        def _tokens_input_cb(options):
            for (model, kind), total in c.llm_usage_input_tokens.items():
                yield _obs(
                    total,
                    {
                        "modelName": model or "unknown",
                        "spanKind": kind,
                        "usageType": "input",
                    },
                )

        def _tokens_output_cb(options):
            for (model, kind), total in c.llm_usage_output_tokens.items():
                yield _obs(
                    total,
                    {
                        "modelName": model or "unknown",
                        "spanKind": kind,
                        "usageType": "output",
                    },
                )

        self._gauges.append(
            self._meter.create_observable_gauge(
                name="genai_calls_count",
                callbacks=[_calls_cb],
                description="Number of GenAI calls.",
            )
        )
        self._gauges.append(
            self._meter.create_observable_gauge(
                name="genai_calls_duration_seconds",
                callbacks=[_duration_cb],
                description="Average GenAI call duration in seconds.",
                unit="s",
            )
        )
        self._gauges.append(
            self._meter.create_observable_gauge(
                name="genai_calls_error_count",
                callbacks=[_error_cb],
                description="Number of failed GenAI calls.",
            )
        )
        self._gauges.append(
            self._meter.create_observable_gauge(
                name="genai_calls_slow_count",
                callbacks=[_slow_cb],
                description="Number of slow GenAI calls (threshold-configurable).",
            )
        )
        self._gauges.append(
            self._meter.create_observable_gauge(
                name="genai_llm_first_token_seconds",
                callbacks=[_ttft_cb],
                description="Average LLM time-to-first-token in seconds.",
                unit="s",
            )
        )
        self._gauges.append(
            self._meter.create_observable_gauge(
                name="genai_llm_usage_tokens",
                callbacks=[_tokens_input_cb, _tokens_output_cb],
                description="LLM token usage (input / output).",
                unit="{token}",
            )
        )

    # ---- SpanProcessor interface ----
    def on_start(
        self, span: OtelSpan, parent_context: Optional[Context] = None
    ) -> None:
        try:
            ctx = span.get_span_context()
            sid = ctx.span_id
            # str() hex form, used as dict key
            key = format(sid, "016x")
        except Exception:
            return
        self._live_spans[key] = span
        parent = getattr(span, "_parent", None)
        parent_id = None
        if parent is not None:
            try:
                parent_id = format(parent.span_id, "016x")
            except Exception:
                parent_id = None
        self._span_parents[key] = parent_id

    def on_end(self, span: Any) -> None:
        """Enrich a just-ended MAF span with ARMS GenAI semantic conventions."""
        try:
            ctx = span.get_span_context()
            key = format(ctx.span_id, "016x")
        except Exception:
            return
        live = self._live_spans.pop(key, None)
        parent_id = self._span_parents.pop(key, None)
        if live is None:
            return
        # NOTE: by the time on_end runs, the SDK has already called Span.end(),
        # so is_recording() is False and the public set_attribute / set_status
        # are no-ops. We mutate ``_attributes`` / ``_status`` directly (see
        # ``_set_attr``). Same approach as the OpenInference processor.

        try:
            name = span.name or ""
            existing_op = _attr_value(span, GEN_AI_OPERATION_NAME)
            existing_op = existing_op if isinstance(existing_op, str) else None
            span_kind, op_name = _classify_span(name, existing_op, span)

            # 1) gen_ai.span.kind (only set if not already present)
            if not _attr_value(span, GEN_AI_SPAN_KIND):
                _set_attr(live, GEN_AI_SPAN_KIND, span_kind)

            # 2) gen_ai.operation.name (set if missing or freshly derived for
            #    workflow spans where MAF does not write it)
            if not existing_op:
                _set_attr(live, GEN_AI_OPERATION_NAME, op_name)
            elif existing_op != op_name and span_kind in {
                GenAISpanKind.TASK,
                GenAISpanKind.AGENT,
            }:
                # executor.process reclassification (FunctionExecutor -> TASK,
                # AgentExecutor -> AGENT/invoke_agent)
                _set_attr(live, GEN_AI_OPERATION_NAME, op_name)

            # 3) gen_ai.framework (always — ARMS extension)
            if not _attr_value(span, GEN_AI_FRAMEWORK):
                _set_attr(live, GEN_AI_FRAMEWORK, FRAMEWORK_NAME)

            # 4) Rename MAF private-prefix attributes
            _rename_maf_attrs(live, span)

            # 5) Normalize provider.name
            provider = _attr_value(span, GEN_AI_PROVIDER_NAME)
            normalized = _normalize_provider(provider)
            if normalized is not None and normalized != provider:
                _set_attr(live, GEN_AI_PROVIDER_NAME, normalized)

            # 6) TTFT backfill for LLM spans with streaming events
            if span_kind == GenAISpanKind.LLM:
                ttft = _ttft_from_events(span)
                if ttft is not None and not _attr_value(
                    span, GEN_AI_RESPONSE_TTFT
                ):
                    _set_attr(live, GEN_AI_RESPONSE_TTFT, ttft)
                    _set_attr(live, GEN_AI_USER_TTFT, ttft)

            # 7) ENTRY detection: a root invoke_agent span with no parent becomes
            #    the trace entry point.
            if (
                span_kind == GenAISpanKind.AGENT
                and op_name == GenAIOperation.INVOKE_AGENT
                and parent_id is None
            ):
                # Only reclassify if there's no explicit ENTRY span above us
                # (we cannot see siblings; this is best-effort). We keep AGENT
                # kind on the actual agent span — ENTRY is represented by the
                # AGENT span itself being the root, in line with the spec note
                # that ENTRY is an entry-point identifier, not a separate kind
                # unless an application-level ENTRY span exists.
                pass

            # 8) Status: set OK on success (MAF only sets ERROR). The SDK
            #    passes a ``ReadableSpan`` snapshot to ``on_end``; its
            #    ``_status`` is a separate field from the live Span's, so we
            #    mutate the ReadableSpan's ``_status`` directly (and the live
            #    Span's too, for symmetry). The shared ``_attributes`` dict is
            #    mutated via ``_set_attr`` above and is visible to exporters.
            current_status = getattr(span, "status", None)
            status_code = getattr(current_status, "status_code", None)
            if status_code != StatusCode.ERROR:
                ok_status = Status(StatusCode.OK)
                try:
                    span._status = ok_status  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("set_status OK on readable failed: %s", exc)
                try:
                    live._status = ok_status  # type: ignore[attr-defined]
                except Exception:
                    pass

            # 9) error.type already set by MAF via capture_exception; nothing to do.

            # 10) Aggregate metrics
            if self._metrics_enabled:
                self._aggregate_metrics(span, span_kind, op_name)

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MAFSemanticProcessor.on_end failed: %s", exc)
        finally:
            # Span has already been ended by the SDK; nothing to do here.
            pass

    def _aggregate_metrics(
        self, readable: Any, span_kind: str, op_name: str
    ) -> None:
        try:
            model = _attr_value(readable, GEN_AI_REQUEST_MODEL)
            if not model:
                model = _attr_value(readable, GEN_AI_RESPONSE_MODEL)
            model = model if isinstance(model, str) else "unknown"
            key = (model, span_kind)
            self._counters.calls_count[key] += 1

            start = getattr(readable, "start_time", None)
            end = getattr(readable, "end_time", None)
            if start is not None and end is not None:
                try:
                    duration_ns = int(end - start)
                    self._counters.calls_duration_ns_sum[key] += duration_ns
                    if duration_ns >= self._slow_threshold_ns:
                        self._counters.calls_slow_count[key] += 1
                except (TypeError, ValueError):
                    pass

            current_status = getattr(readable, "status", None)
            status_code = getattr(current_status, "status_code", None)
            if status_code == StatusCode.ERROR:
                self._counters.calls_error_count[key] += 1
            elif _attr_value(readable, ERROR_TYPE):
                self._counters.calls_error_count[key] += 1

            if span_kind == GenAISpanKind.LLM:
                ttft = _attr_value(readable, GEN_AI_RESPONSE_TTFT)
                if isinstance(ttft, (int, float)) and ttft > 0:
                    self._counters.llm_first_token_ns_sum[key] += int(ttft)
                    self._counters.llm_first_token_count[key] += 1
                input_tokens = _attr_value(readable, GEN_AI_USAGE_INPUT_TOKENS)
                if isinstance(input_tokens, (int, float)):
                    self._counters.llm_usage_input_tokens[key] += int(input_tokens)
                output_tokens = _attr_value(readable, GEN_AI_USAGE_OUTPUT_TOKENS)
                if isinstance(output_tokens, (int, float)):
                    self._counters.llm_usage_output_tokens[key] += int(output_tokens)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("MAF metrics aggregation failed: %s", exc)

    def shutdown(self) -> None:
        self._live_spans.clear()
        self._span_parents.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _obs(value: float, attrs: Dict[str, str]):
    """Build an Observation compatible with OTel callbacks."""
    from opentelemetry.metrics import Observation

    return Observation(value, attrs)


__all__ = ["MAFSemanticProcessor"]
