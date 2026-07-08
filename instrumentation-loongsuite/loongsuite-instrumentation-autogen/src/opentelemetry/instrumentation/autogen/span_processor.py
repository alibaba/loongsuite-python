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

"""Normalize AutoGen native spans to LoongSuite GenAI semantics."""

from __future__ import annotations

import logging
import threading
from typing import Any, Mapping, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import SpanKind

from .semantic_conventions import (
    AUTOGEN_LIVE_SPAN_MARKER,
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    GEN_AI_SYSTEM,
    GenAIOperation,
    GenAISpanKind,
)

logger = logging.getLogger(__name__)

_AUTOGEN_NAME_PREFIXES = (
    f"{GenAIOperation.CREATE_AGENT} ",
    f"{GenAIOperation.EXECUTE_TOOL} ",
    f"{GenAIOperation.INVOKE_AGENT} ",
)
_AUTOGEN_PRIVATE_ATTRS = frozenset(
    {
        "autogen.code.block_count",
        "autogen.code.exit_code",
        "autogen.code.retry_attempt",
        "autogen.handoff.context_count",
        "autogen.handoff.source",
        "autogen.handoff.target",
        "autogen.memory.result_count",
        "autogen.team.message_count",
        "autogen.team.participants",
        "autogen.team.stop_reason",
        "autogen.team.type",
        "autogen.user_input.request_id",
    }
)
_CREATE_AGENT_LIFECYCLE_ATTRS = (
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    GEN_AI_SYSTEM,
    "gen_ai.agent.version",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
)


def _attr_value(span: Any, key: str) -> Any:
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


def _span_attrs(span: Any) -> Mapping[Any, Any]:
    if isinstance(span, Mapping):
        return span
    attrs = getattr(span, "_attributes", None)
    if attrs is not None:
        return attrs
    attrs = getattr(span, "attributes", None)
    if attrs is not None:
        return attrs
    return {}


def _has_live_autogen_marker(live: Any | None) -> bool:
    if live is None:
        return False
    return (
        getattr(live, AUTOGEN_LIVE_SPAN_MARKER, None) == AUTOGEN_PROVIDER_NAME
    )


def _has_autogen_marker(readable: Any, live: Any | None = None) -> bool:
    if _has_live_autogen_marker(live):
        return True
    for key in (GEN_AI_SYSTEM, GEN_AI_PROVIDER_NAME):
        if AUTOGEN_PROVIDER_NAME in _attr_values(_attr_value(readable, key)):
            return True
    attrs = _span_attrs(readable)
    return any(str(key) in _AUTOGEN_PRIVATE_ATTRS for key in attrs)


def _set_attr(live_span: OtelSpan, key: str, value: Any) -> None:
    if value is None:
        return
    attrs = getattr(live_span, "_attributes", None)
    lock = getattr(live_span, "_lock", None)
    if attrs is None:
        try:
            live_span.set_attribute(key, value)
        except Exception:
            pass
        return
    try:
        if lock is not None:
            with lock:
                _set_attr_value(attrs, key, value)
        else:
            _set_attr_value(attrs, key, value)
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


def _delete_attr(live_span: OtelSpan, key: str) -> None:
    attrs = getattr(live_span, "_attributes", None)
    if attrs is None:
        return
    lock = getattr(live_span, "_lock", None)
    try:
        if lock is not None:
            with lock:
                _pop_attr_value(attrs, key)
        else:
            _pop_attr_value(attrs, key)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("delete_attribute(%s) failed: %s", key, exc)


def _pop_attr_value(attrs: Any, key: str) -> None:
    try:
        attrs.pop(key, None)
        return
    except TypeError:
        pass
    backing_dict = getattr(attrs, "_dict", None)
    if backing_dict is not None:
        backing_dict.pop(key, None)


def _set_attr_on_both(
    live_span: OtelSpan, readable: Any, key: str, value: Any
) -> None:
    _set_attr(live_span, key, value)
    _set_attr(readable, key, value)


def _delete_attr_on_both(live_span: OtelSpan, readable: Any, key: str) -> None:
    _delete_attr(live_span, key)
    _delete_attr(readable, key)


def _set_otel_span_kind(
    live_span: OtelSpan, readable: Any, kind: SpanKind
) -> None:
    for target in (live_span, readable):
        try:
            target._kind = kind  # type: ignore[attr-defined]
        except Exception:
            pass


def _is_autogen_span(
    _name: str, _operation: Optional[str], readable: Any, live: Any | None
) -> bool:
    return _has_autogen_marker(readable, live)


def _classify_span(
    name: str, operation: Optional[str]
) -> tuple[Optional[str], str]:
    op = operation or ""
    if not op:
        for prefix in _AUTOGEN_NAME_PREFIXES:
            if name.startswith(prefix):
                op = prefix.strip()
                break
    if op in {
        GenAIOperation.CHAT,
        GenAIOperation.GENERATE_CONTENT,
        GenAIOperation.TEXT_COMPLETION,
    }:
        return GenAISpanKind.LLM, op
    if op == GenAIOperation.EXECUTE_TOOL:
        return GenAISpanKind.TOOL, op
    if op == GenAIOperation.CREATE_AGENT:
        return None, op
    if op == GenAIOperation.INVOKE_AGENT:
        return GenAISpanKind.AGENT, op
    return GenAISpanKind.CHAIN, op or GenAIOperation.INVOKE_AGENT


class AutoGenSemanticProcessor(SpanProcessor):
    """SpanProcessor that enriches AutoGen native GenAI spans on end."""

    def __init__(self) -> None:
        self._live_spans: dict[str, OtelSpan] = {}
        self._lock = threading.Lock()

    def on_start(
        self, span: OtelSpan, parent_context: Optional[Context] = None
    ) -> None:
        try:
            key = format(span.get_span_context().span_id, "016x")
        except Exception:
            return
        with self._lock:
            self._live_spans[key] = span

    def on_end(self, span: Any) -> None:
        try:
            key = format(span.get_span_context().span_id, "016x")
        except Exception:
            return
        with self._lock:
            live = self._live_spans.pop(key, None)
        if live is None:
            return

        try:
            name = span.name or ""
            existing_op = _attr_value(span, GEN_AI_OPERATION_NAME)
            operation = existing_op if isinstance(existing_op, str) else None
            if not _is_autogen_span(name, operation, span, live):
                return

            span_kind, op_name = _classify_span(name, operation)
            if op_name == GenAIOperation.CREATE_AGENT:
                for attr_name in _CREATE_AGENT_LIFECYCLE_ATTRS:
                    _delete_attr_on_both(live, span, attr_name)
                return

            if span_kind and not _attr_value(span, GEN_AI_SPAN_KIND):
                _set_attr_on_both(live, span, GEN_AI_SPAN_KIND, span_kind)
            if not operation:
                _set_attr_on_both(live, span, GEN_AI_OPERATION_NAME, op_name)

            # AutoGen 0.7.x native spans write gen_ai.system=autogen. The
            # LoongSuite GenAI profile expects provider.name instead.
            if _attr_value(span, GEN_AI_SYSTEM) == AUTOGEN_PROVIDER_NAME:
                if not _attr_value(span, GEN_AI_PROVIDER_NAME):
                    _set_attr_on_both(
                        live,
                        span,
                        GEN_AI_PROVIDER_NAME,
                        AUTOGEN_PROVIDER_NAME,
                    )
                _delete_attr_on_both(live, span, GEN_AI_SYSTEM)
            elif not _attr_value(span, GEN_AI_PROVIDER_NAME):
                _set_attr_on_both(
                    live, span, GEN_AI_PROVIDER_NAME, AUTOGEN_PROVIDER_NAME
                )

            if span_kind == GenAISpanKind.LLM:
                _set_otel_span_kind(live, span, SpanKind.CLIENT)
            elif span_kind == GenAISpanKind.AGENT and op_name == (
                GenAIOperation.INVOKE_AGENT
            ):
                _set_otel_span_kind(live, span, SpanKind.INTERNAL)
            elif span_kind == GenAISpanKind.TOOL:
                _set_otel_span_kind(live, span, SpanKind.INTERNAL)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("AutoGenSemanticProcessor.on_end failed: %s", exc)

    def shutdown(self) -> None:
        with self._lock:
            self._live_spans.clear()
