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

"""``MAFSemanticProcessor`` — a ``SpanProcessor`` that enriches Microsoft Agent
Framework's native OTel spans to align with the ARMS GenAI semantic conventions
(``/apsara/semantic-conventions/arms_docs/trace/gen-ai.md``).

MAF already emits OTel spans via its telemetry layers (``ChatTelemetryLayer``,
``EmbeddingTelemetryLayer``, ``AgentTelemetryLayer``, workflow span helpers).
This processor:

1. Injects ``gen_ai.span.kind`` (MAF does not set it).
2. Copies registry-defined MAF private-prefix attributes
   (``workflow.name`` → ``gen_ai.workflow.name``) per the local mapping table in
   :mod:`semantic_conventions`; other MAF private attributes are kept under
   their original names to avoid extending the ``gen_ai`` namespace with
   unregistered keys.
3. Backfills ``gen_ai.response.time_to_first_token`` from the first streaming
   chunk event timestamp.
4. Normalizes ``gen_ai.provider.name`` (``azure_openai`` → ``openai``).
5. Leaves metric emission to Microsoft Agent Framework's native counter and
   histogram instruments instead of re-exporting process-cumulative gauges.

Truncation / JSON serialization are reused from ``opentelemetry.util.genai.utils``
(``gen_ai_json_dumps``) — aligned with the pattern at
``instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/.../span_processor.py:27``.
``gen_ai_json_dumps`` itself only serializes (it does not truncate), so the
single-field 4 KB cap from execute.md is enforced in :func:`_safe_dumps`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from opentelemetry.context import Context
from opentelemetry.sdk.trace import (
    SpanProcessor,
    TracerProvider,  # noqa: F401  (typing hint)
)
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import SpanKind
from opentelemetry.trace.span import TraceState  # noqa: F401
from opentelemetry.util.genai.utils import gen_ai_json_dumps

from .semantic_conventions import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_TTFT,
    GEN_AI_SPAN_KIND,
    MAF_ATTR_RENAME_MAP,
    MAF_LIVE_SPAN_MARKER,
    MAF_PROVIDER_NAME,
    MAF_SPAN_NAME_PREFIXES,
    PROVIDER_NAME_NORMALIZE,
    GenAIOperation,
    GenAISpanKind,
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
_LIVE_SPAN_MAX_AGE_NS = 60 * 1_000_000_000
_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
_FRAMEWORK_PROVIDER_NAME = MAF_PROVIDER_NAME
_GEN_AI_SYSTEM = "gen_ai.system"
_MAF_LIVE_SPAN_MARKER = MAF_LIVE_SPAN_MARKER
_MAF_INTERNAL_NAME_PREFIXES = (
    _WORKFLOW_RUN,
    _WORKFLOW_BUILD,
    _MESSAGE_SEND,
    _EXECUTOR_PROCESS,
    _EDGE_GROUP_PROCESS,
)
_TOOL_CALL_FINISH_REASONS = frozenset(
    {"tool_call", "tool_calls", "function_call", "function_calls"}
)
_AGENT_OUTPUT_ROLES = frozenset({"assistant", "ai", "model"})
_AGENT_INTERMEDIATE_PART_TYPES = frozenset(
    {"tool_call", "tool_call_response", "reasoning"}
)


def _attr_value(span: Any, key: str) -> Any:
    """Read an attribute from a live Span or ReadableSpan, tolerating both."""
    if isinstance(span, Mapping):
        return span.get(key)
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


def _attr_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = (value,)
    return tuple(str(item).lower() for item in values if item is not None)


def _span_attrs(readable: Any) -> Mapping[Any, Any]:
    if isinstance(readable, Mapping):
        return readable
    attrs = getattr(readable, "_attributes", None)
    if attrs is not None:
        return attrs
    attrs = getattr(readable, "attributes", None)
    if attrs is not None:
        return attrs
    return {}


def _has_live_maf_marker(live: Any | None) -> bool:
    if live is None:
        return False
    if isinstance(live, Mapping):
        return live.get(_MAF_LIVE_SPAN_MARKER) == _FRAMEWORK_PROVIDER_NAME
    return (
        getattr(live, _MAF_LIVE_SPAN_MARKER, None) == _FRAMEWORK_PROVIDER_NAME
    )


def _scope_name(span: Any) -> str:
    for attr in ("instrumentation_scope", "instrumentation_library"):
        scope = getattr(span, attr, None)
        name = getattr(scope, "name", None)
        if name:
            return str(name).lower()
    return ""


def _has_maf_scope(readable: Any, live: Any | None = None) -> bool:
    return any(
        "agent_framework" in _scope_name(span)
        or "microsoft_agent_framework" in _scope_name(span)
        for span in (readable, live)
        if span is not None
    )


def _has_maf_provider_marker(readable: Any) -> bool:
    for key in (_GEN_AI_SYSTEM, GEN_AI_PROVIDER_NAME):
        if _FRAMEWORK_PROVIDER_NAME in _attr_values(
            _attr_value(readable, key)
        ):
            return True
    return False


def _has_maf_private_marker(readable: Any) -> bool:
    attrs = _span_attrs(readable)
    return any(key in attrs for key in MAF_ATTR_RENAME_MAP)


def _safe_dumps(obj: Any) -> Optional[str]:
    """Serialize ``obj`` to a JSON string capped at 4 KB.

    Uses the shared ``gen_ai_json_dumps`` helper from
    ``opentelemetry.util.genai.utils`` (the same path as
    ``openai-agents-v2/span_processor.py:27``) for compact, ASCII-preserving
    JSON serialization of arbitrary objects (bytes / datetimes / nested
    dicts). Note that ``gen_ai_json_dumps`` itself does *not* truncate — it is
    just ``json.dumps`` with a custom encoder — so we cap the output at 4 KB
    here to honour the execute.md single-field cap (per-attribute budget
    shared with the rename path). Falls back to ``str(obj)`` if JSON
    serialization fails.
    """
    try:
        out = gen_ai_json_dumps(obj)
    except (TypeError, ValueError):
        out = str(obj)
    return out[:4096]


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
                _set_attr_value(attrs, key, coerced)
        else:
            _set_attr_value(attrs, key, coerced)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("set_attribute(%s) failed: %s", key, exc)


def _set_attr_value(attrs: Any, key: str, value: Any) -> None:
    try:
        attrs[key] = value
        return
    except TypeError:
        pass
    backing_dict = getattr(attrs, "_dict", None)
    if backing_dict is not None:
        backing_dict[key] = value


def _set_attr_on_both(
    live_span: OtelSpan, readable: Any, key: str, value: Any
) -> None:
    _set_attr(live_span, key, value)
    _set_attr(readable, key, value)


def _delete_attr(span: Any, key: str) -> None:
    attrs = getattr(span, "_attributes", None)
    if attrs is None:
        return
    lock = getattr(span, "_lock", None)
    try:
        if lock is not None:
            with lock:
                attrs.pop(key, None)
        else:
            attrs.pop(key, None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("delete_attribute(%s) failed: %s", key, exc)


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
            _set_attr_on_both(live_span, readable, new_key, value)
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
            _delete_attr(readable, old_key)
        else:
            # No live attrs; fall back to readable.attributes (read-only)
            readable_attrs = getattr(readable, "attributes", None) or {}
            if old_key in readable_attrs:
                _set_attr_on_both(
                    live_span, readable, new_key, readable_attrs[old_key]
                )
                renamed.append(new_key)
    return renamed


def _copy_maf_attrs(live_span: OtelSpan, readable: Any) -> list[str]:
    """Copy MAF-private attributes to canonical keys without removing sources."""
    copied: list[str] = []
    for old_key, new_key in MAF_ATTR_RENAME_MAP.items():
        value = _attr_value(readable, old_key)
        if value is None:
            continue
        _set_attr_on_both(live_span, readable, new_key, value)
        copied.append(new_key)
    return copied


def _normalize_provider(value: Any) -> Optional[str]:
    """Normalize ``gen_ai.provider.name`` to the ARMS canonical value.

    MAF can write OpenAI aliases or framework-level values, and may wrap the
    value in a sequence for some span types. We:

    1. Unwrap sequence attribute values (OTel allows ``str | sequence[str]``).
    2. Try an exact match against ``PROVIDER_NAME_NORMALIZE``.
    3. Fall back to a case-insensitive match.
    4. Return the lower-cased raw value for unknown providers. We intentionally
       do not map ``microsoft.agent_framework`` to ``openai`` because MAF can
       route to multiple underlying model providers.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, str):
        value = str(value)
    if value in PROVIDER_NAME_NORMALIZE:
        return PROVIDER_NAME_NORMALIZE[value]
    lowered = value.lower()
    if lowered in PROVIDER_NAME_NORMALIZE:
        return PROVIDER_NAME_NORMALIZE[lowered]
    return lowered


def _normalize_finish_reasons(live_span: OtelSpan, readable: Any) -> None:
    """Normalize JSON-encoded finish reasons to an OTel string array."""
    value = _attr_value(readable, GEN_AI_RESPONSE_FINISH_REASONS)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        normalized = [_normalize_finish_reason_value(item) for item in value]
        if list(value) != normalized:
            _set_attr_on_both(
                live_span,
                readable,
                GEN_AI_RESPONSE_FINISH_REASONS,
                normalized,
            )
        return
    if not isinstance(value, str):
        return
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return
    if isinstance(parsed, list) and all(
        isinstance(item, str) for item in parsed
    ):
        parsed = [_normalize_finish_reason_value(item) for item in parsed]
        _set_attr_on_both(
            live_span, readable, GEN_AI_RESPONSE_FINISH_REASONS, parsed
        )


def _normalize_input_messages(
    live_span: OtelSpan, readable: Any, *, agent_boundary: bool = False
) -> None:
    if not agent_boundary:
        return
    handled, normalized = _agent_boundary_input_messages_value(
        _attr_value(readable, _GEN_AI_INPUT_MESSAGES)
    )
    if not handled:
        return
    if normalized is None:
        _delete_attr(live_span, _GEN_AI_INPUT_MESSAGES)
        _delete_attr(readable, _GEN_AI_INPUT_MESSAGES)
    else:
        _set_attr_on_both(
            live_span, readable, _GEN_AI_INPUT_MESSAGES, normalized
        )


def _normalize_output_messages(
    live_span: OtelSpan,
    readable: Any,
    *,
    agent_boundary: bool = False,
) -> None:
    if agent_boundary:
        handled, normalized = _agent_boundary_output_messages_value(
            _attr_value(readable, _GEN_AI_OUTPUT_MESSAGES)
        )
        if not handled:
            return
        if normalized is None:
            _delete_attr(live_span, _GEN_AI_OUTPUT_MESSAGES)
            _delete_attr(readable, _GEN_AI_OUTPUT_MESSAGES)
        else:
            _set_attr_on_both(
                live_span, readable, _GEN_AI_OUTPUT_MESSAGES, normalized
            )
        return

    normalized = _normalized_output_messages_value(
        _attr_value(readable, _GEN_AI_OUTPUT_MESSAGES)
    )
    if normalized is not None:
        _set_attr_on_both(
            live_span, readable, _GEN_AI_OUTPUT_MESSAGES, normalized
        )


def _normalized_output_messages_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        messages = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    if not isinstance(messages, list):
        return None
    changed = False
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue
        normalized = dict(message)
        current = normalized.get("finish_reason")
        default = _default_finish_reason_for_message(normalized)
        finish_reason = _normalize_finish_reason_value(
            current, default=default
        )
        if current != finish_reason:
            normalized["finish_reason"] = finish_reason
            changed = True
        normalized_messages.append(normalized)
    if not changed and isinstance(value, str):
        return None
    try:
        return json.dumps(
            normalized_messages, ensure_ascii=False, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return None


def _parse_message_list(value: Any) -> tuple[bool, list[Any]]:
    if value is None:
        return False, []
    try:
        messages = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return False, []
    if not isinstance(messages, list):
        return False, []
    return True, messages


def _agent_boundary_input_messages_value(
    value: Any,
) -> tuple[bool, Optional[str]]:
    handled, messages = _parse_message_list(value)
    if not handled:
        return False, None
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != "user":
            continue
        normalized = _message_with_visible_parts(message)
        if normalized is not None:
            normalized_messages.append(normalized)
    if not normalized_messages:
        return True, None
    return True, _dump_messages(normalized_messages)


def _agent_boundary_output_messages_value(
    value: Any,
) -> tuple[bool, Optional[str]]:
    handled, messages = _parse_message_list(value)
    if not handled:
        return False, None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role not in _AGENT_OUTPUT_ROLES:
            continue
        normalized = _message_with_visible_parts(message)
        if normalized is None:
            continue
        current = normalized.get("finish_reason")
        normalized["finish_reason"] = _normalize_finish_reason_value(
            current, default="stop"
        )
        return True, _dump_messages([normalized])
    return True, None


def _message_with_visible_parts(
    message: Mapping[Any, Any],
) -> Optional[dict[Any, Any]]:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return None
    visible_parts = []
    for part in parts:
        if isinstance(part, Mapping):
            part_type = part.get("type")
            if part_type in _AGENT_INTERMEDIATE_PART_TYPES:
                continue
            visible_parts.append(dict(part))
        else:
            visible_parts.append(part)
    if not visible_parts:
        return None
    normalized = dict(message)
    normalized["parts"] = visible_parts
    return normalized


def _dump_messages(messages: list[Any]) -> Optional[str]:
    try:
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _default_finish_reason_for_message(message: Mapping[Any, Any]) -> str:
    parts = message.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, Mapping) and part.get("type") == "tool_call":
                return "tool_calls"
    return "stop"


def _normalize_finish_reason_value(
    value: Any, *, default: str = "stop"
) -> str:
    if value is None or value == "":
        return default
    reason = str(value)
    if reason in _TOOL_CALL_FINISH_REASONS:
        return "tool_calls"
    return reason


def _set_span_kind(live_span: OtelSpan, readable: Any, kind: SpanKind) -> None:
    """Mutate the SDK span kind before downstream exporters receive the span."""
    for target in (readable, live_span):
        try:
            target._kind = kind  # type: ignore[attr-defined]
        except Exception:
            pass


_MCP_METHOD_NAME_ATTR = "mcp.method.name"
_MCP_TOOL_CALL_METHOD = "tools/call"


def _is_mcp_span(readable: Any) -> bool:
    """Return True if ``readable`` is an MCP client span.

    MAF's ``create_mcp_client_span`` (``observability.py:2101``) always sets
    ``mcp.method.name``. We also accept ``SpanKind.CLIENT`` + any ``mcp.*``
    attribute as a fallback in case MAF later renames the attribute.
    """
    if _attr_value(readable, _MCP_METHOD_NAME_ATTR) is not None:
        return True
    try:
        from opentelemetry.trace import SpanKind

        kind = getattr(readable, "kind", None)
        if kind == SpanKind.CLIENT:
            attrs = getattr(readable, "attributes", None) or {}
            if any(str(k).startswith("mcp.") for k in attrs):
                return True
    except Exception:
        pass
    return False


def _mcp_tool_name(name: str, readable: Any) -> Optional[str]:
    """Return a low-cardinality tool name for an MCP client span."""
    tool_call_prefix = f"{_MCP_TOOL_CALL_METHOD} "
    if name.startswith(tool_call_prefix):
        target = name[len(tool_call_prefix) :].strip()
        if target:
            return target
    method = _attr_value(readable, _MCP_METHOD_NAME_ATTR)
    method = method if isinstance(method, str) else None
    if method and name.startswith(f"{method} "):
        target = name[len(method) + 1 :].strip()
        if target:
            return target
    return method or (name.strip() if name else None)


def _is_mcp_tool_call_span(name: str, readable: Any) -> bool:
    """Return True only for MCP ``tools/call`` spans.

    MCP lifecycle operations such as ``initialize`` and ``tools/list`` are MCP
    protocol spans, but they are not GenAI tool executions. Upstream MCP
    semantic conventions only align MCP tool calls with GenAI ``execute_tool``.
    """
    method = _attr_value(readable, _MCP_METHOD_NAME_ATTR)
    if method == _MCP_TOOL_CALL_METHOD:
        return True
    return name.startswith(f"{_MCP_TOOL_CALL_METHOD} ")


def _is_maf_span(
    name: str,
    _operation: Optional[str],
    readable: Any,
    live: Any | None = None,
    *,
    allow_name_prefix: bool = True,
) -> bool:
    """Return True when the span carries a Microsoft Agent Framework signal."""
    if _is_mcp_span(readable) and not _is_mcp_tool_call_span(name, readable):
        return False
    if (
        _has_live_maf_marker(live)
        or _has_live_maf_marker(readable)
        or _has_maf_scope(readable, live)
    ):
        return True
    if _has_maf_provider_marker(readable):
        return True
    if allow_name_prefix and any(
        name.startswith(prefix) for prefix in _MAF_INTERNAL_NAME_PREFIXES
    ):
        return True
    return _has_maf_private_marker(readable)


def _classify_span(
    name: str, operation: Optional[str], readable: Any
) -> Tuple[str, str]:
    """Return ``(span_kind, operation_name)`` for a span.

    Classification priority:
    1. Existing ``gen_ai.operation.name`` (set by MAF for chat/embeddings/tool/agent).
    2. MCP tool-call detection — MAF's ``create_mcp_client_span``
       (``observability.py:2083``) emits spans named ``{mcp.method.name} {target}``
       with no ``gen_ai.operation.name`` set. Only ``tools/call`` spans are
       GenAI tool executions; lifecycle spans such as ``initialize`` keep only
       their ``mcp.*`` protocol attributes.
    3. Span-name prefix matching (workflow spans have no operation.name from MAF).
    4. ``react step`` literal name (emitted by our react_step patch).
    """
    # MCP tool calls are GenAI tool executions. Other MCP protocol lifecycle
    # spans are filtered out by ``_is_maf_span`` and keep only their ``mcp.*``
    # attributes.
    if _is_mcp_tool_call_span(name, readable):
        return GenAISpanKind.MCP, GenAIOperation.EXECUTE_TOOL

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
                op = (
                    GenAIOperation.WORKFLOW
                )  # safe default for MAF internal spans

    if (
        op == GenAIOperation.CHAT
        or op == GenAIOperation.TEXT_COMPLETION
        or op == GenAIOperation.GENERATE_CONTENT
    ):
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
    if op == GenAIOperation.WORKFLOW:
        # ``executor.process`` splits by ``executor.type`` when it represents
        # an agent invocation. Other executor/message/edge spans remain part of
        # the workflow operation.
        if name.startswith(_EXECUTOR_PROCESS):
            executor_type = _attr_value(readable, "executor.type")
            if isinstance(executor_type, str):
                et = executor_type.lower()
                if "agent" in et:
                    return GenAISpanKind.AGENT, GenAIOperation.INVOKE_AGENT
        return GenAISpanKind.WORKFLOW, op
    return GenAISpanKind.WORKFLOW, op


def _ttft_from_events(readable: Any) -> Optional[int]:
    """Backfill ``gen_ai.response.time_to_first_token`` (ns) from the first
    streaming chunk event timestamp.

    MAF emits streaming chunks as span events; the first non-exception event's
    timestamp minus the span start time is the TTFT.
    """
    events = getattr(readable, "events", None) or ()
    if not events:
        return None
    start_time = getattr(readable, "start_time", None)
    first_ts = None
    for ev in events:
        if _is_exception_event(ev):
            continue
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


def _is_exception_event(event: Any) -> bool:
    name = getattr(event, "name", None)
    if name is None and isinstance(event, dict):
        name = event.get("name")
    if name == "exception":
        return True
    attributes = getattr(event, "attributes", None)
    if attributes is None and isinstance(event, dict):
        attributes = event.get("attributes")
    return bool(
        isinstance(attributes, dict)
        and (
            "exception.type" in attributes or "exception.message" in attributes
        )
    )


class MAFSemanticProcessor(SpanProcessor):
    """SpanProcessor that injects ARMS GenAI semantic conventions into MAF spans."""

    def __init__(
        self,
        meter_provider: Any = None,
        slow_threshold_ms: int = 1000,
        metrics_enabled: bool = True,
        capture_sensitive_data: bool = False,
    ) -> None:
        # Deprecated compatibility parameters. MAF emits native metrics;
        # retain the old constructor surface for direct processor users.
        _ = meter_provider, slow_threshold_ms, metrics_enabled
        self._live_spans: Dict[str, OtelSpan] = {}
        self._span_parents: Dict[str, Optional[str]] = {}
        self._live_span_lock = threading.Lock()
        self._capture_sensitive = capture_sensitive_data

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
        parent = getattr(span, "_parent", None)
        parent_id = None
        if parent is not None:
            try:
                parent_id = format(parent.span_id, "016x")
            except Exception:
                parent_id = None
        with self._live_span_lock:
            self._live_spans[key] = span
            self._span_parents[key] = parent_id
        self._apply_semantic_attributes(
            span,
            span,
            remove_private_attrs=True,
            allow_name_prefix=False,
        )

    def on_end(self, span: Any) -> None:
        """Enrich a just-ended MAF span with ARMS GenAI semantic conventions."""
        try:
            ctx = span.get_span_context()
            key = format(ctx.span_id, "016x")
        except Exception:
            return
        with self._live_span_lock:
            live = self._live_spans.pop(key, None)
            parent_id = self._span_parents.pop(key, None)
        if live is None:
            return
        # NOTE: by the time on_end runs, the SDK has already called Span.end(),
        # so is_recording() is False and the public set_attribute / set_status
        # are no-ops. We mutate ``_attributes`` / ``_status`` directly (see
        # ``_set_attr``). Same approach as the OpenInference processor.

        try:
            classified = self._apply_semantic_attributes(
                live, span, remove_private_attrs=True
            )
            if classified is None:
                return
            span_kind, op_name = classified

            # TTFT backfill for LLM spans with streaming events
            if span_kind == GenAISpanKind.LLM:
                ttft = _ttft_from_events(span)
                if ttft is not None and not _attr_value(
                    span, GEN_AI_RESPONSE_TTFT
                ):
                    _set_attr_on_both(live, span, GEN_AI_RESPONSE_TTFT, ttft)

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

            # 8) Status: MAF already sets ERROR on failed spans. Successful
            # spans are left UNSET, matching the OTel SDK default and Weaver's
            # validation model.

            # 9) error.type already set by MAF via capture_exception; nothing to do.

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MAFSemanticProcessor.on_end failed: %s", exc)
        finally:
            # Span has already been ended by the SDK; nothing to do here.
            pass

    def _apply_semantic_attributes(
        self,
        live: OtelSpan,
        readable: Any,
        *,
        remove_private_attrs: bool,
        allow_name_prefix: bool = True,
    ) -> Optional[Tuple[str, str]]:
        """Write GenAI semantic attributes while the span is still exportable.

        ``on_start`` runs before any downstream exporter receives a completed
        span, so workflow spans must be normalized there. ``on_end`` keeps the
        same logic as a fallback for attributes written after span creation.
        """
        name = getattr(readable, "name", "") or getattr(live, "name", "") or ""
        existing_op = _attr_value(readable, GEN_AI_OPERATION_NAME)
        existing_op = existing_op if isinstance(existing_op, str) else None
        if not _is_maf_span(
            name,
            existing_op,
            readable,
            live,
            allow_name_prefix=allow_name_prefix,
        ):
            return None

        span_kind, op_name = _classify_span(name, existing_op, readable)
        existing_kind = _attr_value(readable, GEN_AI_SPAN_KIND)

        if (
            isinstance(existing_kind, str)
            and existing_kind
            and span_kind == GenAISpanKind.WORKFLOW
            and existing_kind != GenAISpanKind.WORKFLOW
        ):
            span_kind = existing_kind

        # 1) gen_ai.span.kind (only set if not already present, or when a
        #    later-written MAF attribute lets us refine our own WORKFLOW
        #    fallback into a more specific kind).
        if not existing_kind or (
            existing_kind == GenAISpanKind.WORKFLOW
            and span_kind != GenAISpanKind.WORKFLOW
        ):
            _set_attr_on_both(live, readable, GEN_AI_SPAN_KIND, span_kind)

        # 2) gen_ai.operation.name (set if missing or freshly derived for
        #    workflow spans where MAF does not write it). For spans MAF
        #    mislabels (e.g. MCP ``tools/call`` written by MAF as
        #    ``execute_tool`` — see ``create_mcp_client_span`` at
        #    ``observability.py:2101``) we also override when our
        #    classification disagrees, provided the span is one of the
        #    kinds whose operation.name we own (AGENT reclassification of
        #    ``executor.process``).
        if not existing_op:
            _set_attr_on_both(live, readable, GEN_AI_OPERATION_NAME, op_name)
        elif existing_op != op_name and span_kind == GenAISpanKind.AGENT:
            _set_attr_on_both(live, readable, GEN_AI_OPERATION_NAME, op_name)

        if span_kind == GenAISpanKind.MCP and not _attr_value(
            readable, _GEN_AI_TOOL_NAME
        ):
            tool_name = _mcp_tool_name(name, readable)
            if tool_name:
                _set_attr_on_both(live, readable, _GEN_AI_TOOL_NAME, tool_name)

        # 3) Rename MAF private-prefix attributes
        if remove_private_attrs:
            _rename_maf_attrs(live, readable)
        else:
            _copy_maf_attrs(live, readable)

        # 4) Normalize provider.name
        provider = _attr_value(readable, GEN_AI_PROVIDER_NAME)
        normalized = _normalize_provider(provider)
        if normalized is not None and normalized != provider:
            _set_attr_on_both(live, readable, GEN_AI_PROVIDER_NAME, normalized)
        elif (
            normalized is None
            and span_kind == GenAISpanKind.AGENT
            and op_name
            in {
                GenAIOperation.CREATE_AGENT,
                GenAIOperation.INVOKE_AGENT,
            }
        ):
            _set_attr_on_both(
                live, readable, GEN_AI_PROVIDER_NAME, _FRAMEWORK_PROVIDER_NAME
            )

        # 5) Normalize finish reasons and content-capture messages. AGENT
        # input/output is the user-visible boundary; intermediate ReAct/tool
        # messages stay on child STEP/LLM/TOOL spans.
        _normalize_finish_reasons(live, readable)
        agent_boundary = (
            span_kind == GenAISpanKind.AGENT
            and op_name == GenAIOperation.INVOKE_AGENT
        )
        _normalize_input_messages(
            live, readable, agent_boundary=agent_boundary
        )
        _normalize_output_messages(
            live, readable, agent_boundary=agent_boundary
        )

        # 6) LLM/embedding spans should use CLIENT OTel span kind.
        if span_kind == GenAISpanKind.LLM:
            _set_span_kind(live, readable, SpanKind.CLIENT)

        return span_kind, op_name

    def shutdown(self) -> None:
        with self._live_span_lock:
            self._live_spans.clear()
            self._span_parents.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._sweep_stale_live_spans()
        return True

    def _sweep_stale_live_spans(
        self, max_age_ns: int = _LIVE_SPAN_MAX_AGE_NS
    ) -> None:
        """Bound live-span bookkeeping if a span is started but never ended."""
        now_ns = time.time_ns()
        stale_keys = []
        with self._live_span_lock:
            for key, live_span in list(self._live_spans.items()):
                start_time = getattr(live_span, "start_time", None)
                if start_time is None:
                    continue
                try:
                    if now_ns - int(start_time) > max_age_ns:
                        stale_keys.append(key)
                except (TypeError, ValueError):
                    continue
            for key in stale_keys:
                self._live_spans.pop(key, None)
                self._span_parents.pop(key, None)


__all__ = ["MAFSemanticProcessor"]
