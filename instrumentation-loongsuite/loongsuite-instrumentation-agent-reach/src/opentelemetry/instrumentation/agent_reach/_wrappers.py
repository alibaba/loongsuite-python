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

"""wrapt wrappers for Agent-Reach instrumentation.

Each wrapper creates a GenAI span via ``ExtendedTelemetryHandler``:

* ``EntryWrapper``           → ENTRY span for ``cli.main``
* ``DoctorWrapper``          → TOOL span for ``doctor.check_all``
* ``ChannelCheckWrapper``    → TOOL span for each ``Channel.check`` override
* ``ProbeWrapper``           → TOOL span for ``probe.probe_command``
* ``TranscribeWrapper``      → TOOL span for ``transcribe.transcribe``
* ``TranscribeChunkWrapper`` → TOOL span for ``transcribe.transcribe_chunk``
* ``MCPServerWrapper``       → patches ``create_server`` so the inner
  ``call_tool`` closure produces a TOOL span (only enabled when the
  standalone ``loongsuite-instrumentation-mcp`` is NOT active, to avoid
  duplicate spans).

All wrappers are exception-safe: a failure inside attribute collection
downgrades to omitting that attribute rather than raising into user code.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from typing import Any, Optional

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
)
from opentelemetry.util.genai.types import Error

from . import semconv as sc
from ._utils import (
    _get_attr,
    _is_content_capture_enabled,
    _safe_json_dumps,
    _truncate,
)

logger = logging.getLogger(__name__)


def _set_attr(span: Any, key: str, value: Any) -> None:
    if span is None or value is None:
        return
    if not hasattr(span, "is_recording") or not span.is_recording():
        return
    try:
        span.set_attribute(key, value)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to set attribute %s: %s", key, e)


def _apply_tool_invocation_attrs(
    invocation: ExecuteToolInvocation,
    *,
    tool_name: str,
    tool_description: Optional[str] = None,
    provider: Optional[str] = None,
    extra_attributes: Optional[dict] = None,
) -> None:
    """Populate standard TOOL attributes on an ExecuteToolInvocation."""
    invocation.tool_name = tool_name
    if tool_description is not None:
        invocation.tool_description = tool_description
    if provider is not None:
        invocation.provider = provider
    if extra_attributes:
        invocation.attributes.update(extra_attributes)


def _set_tool_arguments(invocation: ExecuteToolInvocation, value: Any) -> None:
    if not _is_content_capture_enabled():
        return
    payload = _safe_json_dumps(value)
    if payload is not None:
        invocation.tool_call_arguments = payload


def _set_tool_result(invocation: ExecuteToolInvocation, value: Any) -> None:
    if not _is_content_capture_enabled():
        return
    payload = _safe_json_dumps(value)
    if payload is not None:
        invocation.tool_call_result = payload


class EntryWrapper:
    """Wrap ``agent_reach.cli.main`` to create an ENTRY span."""

    _FRAMEWORK = sc.GEN_AI_FRAMEWORK

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        session_id = str(uuid.uuid4())
        entry_inv = EntryInvocation(session_id=session_id)
        entry_inv.attributes.update(
            {
                sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_ENTRY,
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_ENTER,
                sc.GEN_AI_FRAMEWORK_ATTR: self._FRAMEWORK,
            }
        )
        if _is_content_capture_enabled():
            argv = _safe_json_dumps(list(sys.argv[1:]))
            if argv is not None:
                entry_inv.attributes["input.value"] = argv

        with self._handler.entry(entry_inv):
            return wrapped(*args, **kwargs)


class DoctorWrapper:
    """Wrap ``agent_reach.doctor.check_all`` to create a TOOL span."""

    _TOOL_NAME = "agent-reach-doctor"
    _TOOL_DESCRIPTION = "Check all platform channel availability"

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        channels_count = None
        try:
            from agent_reach.channels import get_all_channels  # noqa: PLC0415

            channels_count = len(get_all_channels())
        except Exception:  # noqa: BLE001
            channels_count = None

        inv = ExecuteToolInvocation(tool_name=self._TOOL_NAME)
        _apply_tool_invocation_attrs(
            inv,
            tool_name=self._TOOL_NAME,
            tool_description=self._TOOL_DESCRIPTION,
            extra_attributes={
                sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_TOOL,
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                sc.GEN_AI_FRAMEWORK_ATTR: sc.GEN_AI_FRAMEWORK,
            },
        )
        if _is_content_capture_enabled() and channels_count is not None:
            _set_tool_arguments(inv, {"channels_count": channels_count})

        try:
            with self._handler.execute_tool(inv):
                result = wrapped(*args, **kwargs)
                if _is_content_capture_enabled() and isinstance(result, dict):
                    ok = sum(
                        1
                        for r in result.values()
                        if isinstance(r, dict) and r.get("status") == "ok"
                    )
                    _set_tool_result(
                        inv,
                        {
                            "ok": ok,
                            "total": len(result),
                            "channels": sorted(list(result.keys())),
                        },
                    )
                return result
        except Exception as exc:
            try:
                self._handler.fail_execute_tool(
                    inv, Error(message=str(exc), type=type(exc))
                )
            except Exception:  # noqa: BLE001
                pass
            raise


class ChannelCheckWrapper:
    """Wrap a concrete ``Channel`` subclass' ``check`` method."""

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        channel_name = _get_attr(instance, "name", "unknown") or "unknown"
        tool_name = f"agent-reach-channel-{channel_name}"
        span_name = f"execute_tool channel-{channel_name}"
        description = _get_attr(instance, "description", "") or ""
        tier = _get_attr(instance, "tier", 0)
        backends = _get_attr(instance, "backends", []) or []

        inv = ExecuteToolInvocation(tool_name=channel_name)
        _apply_tool_invocation_attrs(
            inv,
            tool_name=tool_name,
            tool_description=description,
            extra_attributes={
                sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_TOOL,
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                sc.GEN_AI_FRAMEWORK_ATTR: sc.GEN_AI_FRAMEWORK,
                sc.AGENT_REACH_CHANNEL_TIER: int(tier)
                if isinstance(tier, int)
                else 0,
            },
        )
        if _is_content_capture_enabled():
            _set_tool_arguments(
                inv,
                {
                    "channel": channel_name,
                    "backends": list(backends),
                    "tier": tier,
                },
            )

        try:
            with self._handler.execute_tool(inv):
                # Override the span name (start_execute_tool sets it to
                # ``execute_tool {tool_name}``; we want
                # ``execute_tool channel-{name}`` per execute.md while keeping
                # gen_ai.tool.name = agent-reach-channel-{name}).
                if inv.span is not None:
                    try:
                        inv.span.update_name(span_name)
                    except Exception:  # noqa: BLE001
                        pass
                result = wrapped(*args, **kwargs)
                status = "ok"
                message: Optional[str] = None
                if isinstance(result, tuple) and len(result) >= 2:
                    status = str(result[0] or "ok")
                    message = (
                        result[1] if isinstance(result[1], str) else str(result[1])
                    )
                elif isinstance(result, str):
                    status = "ok"
                    message = result
                _set_attr(inv.span, sc.AGENT_REACH_CHANNEL_STATUS, status)
                active_backend = _get_attr(instance, "active_backend", None)
                if active_backend:
                    _set_attr(
                        inv.span,
                        sc.AGENT_REACH_CHANNEL_ACTIVE_BACKEND,
                        str(active_backend),
                    )
                if _is_content_capture_enabled():
                    _set_tool_result(
                        inv,
                        {
                            "status": status,
                            "message": _truncate(
                                message, sc.MESSAGE_TRUNCATE_CHARS
                            ),
                        },
                    )
                metrics = self._handler.__dict__.get("_agent_reach_metrics")
                if metrics is not None and hasattr(
                    metrics, "record_channel_status"
                ):
                    metrics.record_channel_status(channel_name, status)
                return result
        except Exception as exc:
            try:
                self._handler.fail_execute_tool(
                    inv, Error(message=str(exc), type=type(exc))
                )
            except Exception:  # noqa: BLE001
                pass
            raise


class ProbeWrapper:
    """Wrap ``agent_reach.probe.probe_command``."""

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        cmd = args[0] if args else kwargs.get("cmd")
        probe_args = (
            args[1] if len(args) > 1 else kwargs.get("args", ("--version",))
        )
        timeout = args[2] if len(args) > 2 else kwargs.get("timeout", 10)
        retries = args[3] if len(args) > 3 else kwargs.get("retries", 0)
        package = args[4] if len(args) > 4 else kwargs.get("package")

        tool_name = "agent-reach-probe"
        span_name = (
            f"execute_tool probe-{cmd}" if cmd else "execute_tool probe"
        )

        inv = ExecuteToolInvocation(tool_name=tool_name)
        _apply_tool_invocation_attrs(
            inv,
            tool_name=tool_name,
            extra_attributes={
                sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_TOOL,
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                sc.GEN_AI_FRAMEWORK_ATTR: sc.GEN_AI_FRAMEWORK,
                sc.AGENT_REACH_PROBE_CMD: str(cmd) if cmd else "",
            },
        )
        if _is_content_capture_enabled():
            _set_tool_arguments(
                inv,
                {
                    "cmd": str(cmd) if cmd else "",
                    "args": list(probe_args) if probe_args else [],
                    "timeout": timeout,
                    "retries": retries,
                    "package": package,
                },
            )

        start = time.monotonic()
        try:
            with self._handler.execute_tool(inv):
                if inv.span is not None:
                    try:
                        inv.span.update_name(span_name)
                    except Exception:  # noqa: BLE001
                        pass
                result = wrapped(*args, **kwargs)
                status = _get_attr(result, "status", "ok") or "ok"
                output = _get_attr(result, "output", "")
                hint = _get_attr(result, "hint", "")
                _set_attr(inv.span, sc.AGENT_REACH_PROBE_STATUS, str(status))
                if _is_content_capture_enabled():
                    _set_tool_result(
                        inv,
                        {
                            "status": str(status),
                            "output": _truncate(
                                output, sc.MESSAGE_TRUNCATE_CHARS
                            ),
                            "hint": _truncate(hint, sc.MESSAGE_TRUNCATE_CHARS),
                        },
                    )
                metrics = self._handler.__dict__.get("_agent_reach_metrics")
                if metrics is not None and hasattr(metrics, "record_probe"):
                    metrics.record_probe(
                        str(cmd) if cmd else "",
                        str(status),
                        time.monotonic() - start,
                    )
                return result
        except Exception as exc:
            try:
                self._handler.fail_execute_tool(
                    inv, Error(message=str(exc), type=type(exc))
                )
            except Exception:  # noqa: BLE001
                pass
            raise


class TranscribeWrapper:
    """Wrap ``agent_reach.transcribe.transcribe``."""

    _TOOL_NAME = "agent-reach-transcribe"

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        source = args[0] if args else kwargs.get("source")
        provider = (
            args[1]
            if len(args) > 1
            else kwargs.get("provider", "auto")
        )

        inv = ExecuteToolInvocation(tool_name=self._TOOL_NAME)
        _apply_tool_invocation_attrs(
            inv,
            tool_name=self._TOOL_NAME,
            extra_attributes={
                sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_TOOL,
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                sc.GEN_AI_FRAMEWORK_ATTR: sc.GEN_AI_FRAMEWORK,
                sc.AGENT_REACH_TRANSCRIBE_SOURCE: str(source) if source else "",
                sc.AGENT_REACH_TRANSCRIBE_PROVIDER: str(provider)
                if provider
                else "auto",
            },
        )
        if _is_content_capture_enabled():
            _set_tool_arguments(
                inv,
                {
                    "source": str(source) if source else "",
                    "provider": str(provider) if provider else "auto",
                },
            )

        try:
            with self._handler.execute_tool(inv):
                result = wrapped(*args, **kwargs)
                if isinstance(result, str):
                    line_count = len(
                        [ln for ln in result.split("\n") if ln.strip()]
                    )
                    if line_count > 0:
                        _set_attr(
                            inv.span,
                            sc.AGENT_REACH_TRANSCRIBE_CHUNK_COUNT,
                            line_count,
                        )
                    if _is_content_capture_enabled():
                        _set_tool_result(
                            inv, _truncate(result, sc.MESSAGE_TRUNCATE_CHARS)
                        )
                return result
        except Exception as exc:
            try:
                self._handler.fail_execute_tool(
                    inv, Error(message=str(exc), type=type(exc))
                )
            except Exception:  # noqa: BLE001
                pass
            raise


class TranscribeChunkWrapper:
    """Wrap ``agent_reach.transcribe.transcribe_chunk``."""

    _TOOL_NAME = "agent-reach-whisper"

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        chunk = args[0] if args else kwargs.get("chunk")
        provider = args[1] if len(args) > 1 else kwargs.get("provider")
        model = None
        try:
            from agent_reach.transcribe import PROVIDERS  # noqa: PLC0415

            info = PROVIDERS.get(provider) if provider else None
            if isinstance(info, dict):
                model = info.get("model")
        except Exception:  # noqa: BLE001
            model = None

        chunk_name = _get_attr(chunk, "name", "") or ""

        inv = ExecuteToolInvocation(tool_name=self._TOOL_NAME)
        _apply_tool_invocation_attrs(
            inv,
            tool_name=self._TOOL_NAME,
            provider=str(provider) if provider else None,
            extra_attributes={
                sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_TOOL,
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                sc.GEN_AI_FRAMEWORK_ATTR: sc.GEN_AI_FRAMEWORK,
                sc.GEN_AI_PROVIDER_NAME: str(provider) if provider else "",
                sc.AGENT_REACH_TRANSCRIBE_PROVIDER: str(provider)
                if provider
                else "",
            },
        )
        if model:
            _set_attr(inv.span, sc.AGENT_REACH_TRANSCRIBE_MODEL, str(model))
        if _is_content_capture_enabled():
            _set_tool_arguments(
                inv,
                {
                    "provider": str(provider) if provider else "",
                    "chunk": str(chunk_name),
                    "model": model,
                },
            )

        try:
            with self._handler.execute_tool(inv):
                result = wrapped(*args, **kwargs)
                if _is_content_capture_enabled() and isinstance(result, str):
                    _set_tool_result(
                        inv, _truncate(result, sc.MESSAGE_TRUNCATE_CHARS)
                    )
                return result
        except Exception as exc:
            try:
                self._handler.fail_execute_tool(
                    inv, Error(message=str(exc), type=type(exc))
                )
            except Exception:  # noqa: BLE001
                pass
            raise


class MCPServerWrapper:
    """Wrap ``agent_reach.integrations.mcp_server.create_server``.

    Only installed when ``loongsuite-instrumentation-mcp`` is NOT active.
    """

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self._handler = handler

    def __call__(self, wrapped, instance, args, kwargs):
        server = wrapped(*args, **kwargs)
        if server is None:
            return server

        handler = self._handler
        original_call_tool_decorator = server.call_tool

        def call_tool_decorator(fn):
            decorated = original_call_tool_decorator(fn)

            async def wrapped_call_tool(name, arguments):
                tool_name = f"agent-reach-mcp-{name}" if name else (
                    "agent-reach-mcp"
                )
                span_name = (
                    f"execute_tool mcp-{name}" if name else "execute_tool mcp"
                )
                inv = ExecuteToolInvocation(
                    tool_name=name if name else "mcp"
                )
                _apply_tool_invocation_attrs(
                    inv,
                    tool_name=tool_name,
                    extra_attributes={
                        sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_TOOL,
                        sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                        sc.GEN_AI_FRAMEWORK_ATTR: sc.GEN_AI_FRAMEWORK,
                    },
                )
                if inv.span is not None:
                    try:
                        inv.span.update_name(span_name)
                    except Exception:  # noqa: BLE001
                        pass
                if _is_content_capture_enabled():
                    _set_tool_arguments(
                        inv, {"name": name, "arguments": arguments}
                    )
                try:
                    with handler.execute_tool(inv):
                        if inv.span is not None:
                            try:
                                inv.span.update_name(span_name)
                            except Exception:  # noqa: BLE001
                                pass
                        result = await decorated(name, arguments)
                        if _is_content_capture_enabled():
                            _set_tool_result(
                                inv, _serialize_mcp_result(result)
                            )
                        return result
                except Exception as exc:
                    try:
                        handler.fail_execute_tool(
                            inv, Error(message=str(exc), type=type(exc))
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    raise

            return wrapped_call_tool

        try:
            server.call_tool = call_tool_decorator  # type: ignore[assignment]
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to patch server.call_tool: %s", e)
        return server


def _serialize_mcp_result(result: Any) -> Any:
    """Best-effort serialization of MCP ``TextContent`` lists."""
    if result is None:
        return None
    if isinstance(result, list):
        out = []
        for item in result:
            text = _get_attr(item, "text", None)
            if text is not None:
                out.append({"type": "text", "text": text})
            else:
                out.append(str(item))
        return out
    return str(result)


__all__ = [
    "ChannelCheckWrapper",
    "DoctorWrapper",
    "EntryWrapper",
    "MCPServerWrapper",
    "ProbeWrapper",
    "TranscribeChunkWrapper",
    "TranscribeWrapper",
]
