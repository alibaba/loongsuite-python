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

"""DeerFlow graph and application-entry patches."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import timeit
import uuid
from collections.abc import Generator, Iterator
from contextvars import Context, ContextVar, Token, copy_context
from typing import Any, Callable
from weakref import WeakKeyDictionary

from wrapt import wrap_function_wrapper

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
)
from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.types import Error

from .constants import (
    AGENT_FLAVOR_ATTR,
    CLIENT_STREAM,
    CREATE_AGENT_ALIASES,
    DEERFLOW_AGENT_FLAVOR,
    DEERFLOW_RUN_STATUS,
    GATEWAY_LOADED_RUN_AGENT_ALIASES,
    GATEWAY_RUN_AGENT_ALIASES,
    GEN_AI_AGENT_NAME,
)
from .utils import (
    create_entry_invocation,
    has_active_host_entry,
    non_empty_string,
    should_capture_content,
    to_output_messages,
    trace_id_from_sources,
)

logger = logging.getLogger(__name__)

_ENTRY_DEPTH: ContextVar[int] = ContextVar(
    "loongsuite_deerflow_entry_depth", default=0
)
_MISSING = object()
_LEGACY_GRAPH_ATTRS = (
    "_loongsuite_react_agent",
    "_loongsuite_deepagents_agent",
)

_patched_locations: list[tuple[str, str]] = []
_owned_gateway_wrappers: list[Any] = []
_original_graph_markers: WeakKeyDictionary[Any, dict[str, Any]] = (
    WeakKeyDictionary()
)
_strong_graph_markers: list[tuple[Any, dict[str, Any]]] = []


class DeerFlowRunError(RuntimeError):
    """A DeerFlow Gateway run ended in its error state."""


class DeerFlowRunTimeout(TimeoutError):
    """A DeerFlow Gateway run ended in its timeout state."""


class DeerFlowRunInterrupted(InterruptedError):
    """A DeerFlow Gateway run was interrupted."""


def _track_graph(graph: Any, original_markers: dict[str, Any]) -> None:
    try:
        if graph not in _original_graph_markers:
            _original_graph_markers[graph] = original_markers
        return
    except TypeError:
        pass

    if not any(existing is graph for existing, _ in _strong_graph_markers):
        _strong_graph_markers.append((graph, original_markers))


def mark_deerflow_graph(graph: Any) -> Any:
    """Mark a graph with the DeerFlow flavor and no legacy ReAct flags."""
    marker_attrs = (AGENT_FLAVOR_ATTR, *_LEGACY_GRAPH_ATTRS)
    original_markers = {
        name: getattr(graph, name, _MISSING) for name in marker_attrs
    }
    try:
        setattr(graph, AGENT_FLAVOR_ATTR, DEERFLOW_AGENT_FLAVOR)
        for legacy_attr in _LEGACY_GRAPH_ATTRS:
            if hasattr(graph, legacy_attr):
                delattr(graph, legacy_attr)
    except Exception:  # noqa: BLE001
        logger.debug("Could not mark DeerFlow graph", exc_info=True)
        return graph
    _track_graph(graph, original_markers)
    return graph


def _create_agent_alias_wrapper(
    wrapped: Callable[..., Any],
    _instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    return mark_deerflow_graph(wrapped(*args, **kwargs))


def _entry_is_active() -> bool:
    return _ENTRY_DEPTH.get() > 0 or has_active_host_entry()


def _call_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    position: int,
    name: str,
    default: Any = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def _effective_user_id(fallback: Any = None) -> str | None:
    if fallback is not None:
        return non_empty_string(fallback)
    try:
        from deerflow.runtime.user_context import (  # noqa: PLC0415
            get_effective_user_id,
        )

        return non_empty_string(get_effective_user_id())
    except Exception:  # noqa: BLE001
        return None


def _client_trace_id(instance: Any) -> str | None:
    """Resolve or pre-generate current DeerFlow embedded correlation id."""
    trace_id = trace_id_from_sources()
    if trace_id:
        return trace_id
    try:
        from deerflow.config.app_config import (  # noqa: PLC0415
            is_trace_correlation_enabled,
        )
        from deerflow.trace_context import generate_trace_id  # noqa: PLC0415

        if is_trace_correlation_enabled(
            getattr(instance, "_app_config", None)
        ):
            return generate_trace_id()
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def _bind_deerflow_trace_id(trace_id: str | None) -> Any:
    if not trace_id:
        return None
    try:
        from deerflow.trace_context import (  # noqa: PLC0415
            set_current_trace_id,
        )

        return set_current_trace_id(trace_id)
    except (ImportError, ModuleNotFoundError):
        return None


def _reset_deerflow_trace_id(token: Any) -> None:
    if token is None:
        return
    try:
        from deerflow.trace_context import (  # noqa: PLC0415
            reset_current_trace_id,
        )

        reset_current_trace_id(token)
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError):
        logger.debug("Failed to reset DeerFlow trace context", exc_info=True)


def _gateway_user_id(config: dict[str, Any]) -> str | None:
    """Resolve the Gateway user through DeerFlow's runtime contract."""
    runtime_context = config.get("context")
    runtime_context = (
        runtime_context if isinstance(runtime_context, dict) else {}
    )
    try:
        from deerflow.runtime.user_context import (  # noqa: PLC0415
            resolve_runtime_user_id,
        )

        runtime = type("_Runtime", (), {"context": runtime_context})()
        return non_empty_string(resolve_runtime_user_id(runtime))
    except Exception:  # noqa: BLE001
        return _effective_user_id()


def _gateway_invocation(
    record: Any,
    graph_input: Any,
    config: Any,
) -> EntryInvocation:
    config = config if isinstance(config, dict) else {}
    metadata = config.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    record_metadata = getattr(record, "metadata", None)
    record_metadata = (
        record_metadata if isinstance(record_metadata, dict) else {}
    )
    context = config.get("context")
    context = context if isinstance(context, dict) else {}
    configurable = config.get("configurable")
    configurable = configurable if isinstance(configurable, dict) else {}
    run_name = config.get("run_name") or metadata.get("run_name")
    assistant_id = getattr(record, "assistant_id", None)
    agent_name = (
        context.get("agent_name")
        or configurable.get("agent_name")
        or metadata.get("agent_name")
        or run_name
        or "lead-agent"
    )

    return create_entry_invocation(
        thread_id=getattr(record, "thread_id", None),
        user_id=_gateway_user_id(config),
        agent_name=agent_name,
        assistant_id=assistant_id,
        run_id=getattr(record, "run_id", None),
        deerflow_trace_id=trace_id_from_sources(
            metadata,
            record_metadata,
        ),
        input_value=graph_input,
    )


def _error_from_exception(exc: BaseException) -> Error:
    return Error(message=str(exc), type=type(exc))


def _run_status_value(record: Any) -> str:
    status = getattr(record, "status", None)
    value = getattr(status, "value", status)
    return non_empty_string(value) or "unknown"


def _run_status_error(record: Any, status: str) -> BaseException:
    message = non_empty_string(getattr(record, "error", None))
    if status == "timeout":
        return DeerFlowRunTimeout(message or "DeerFlow run timed out")
    if status == "interrupted":
        return DeerFlowRunInterrupted(
            message or "DeerFlow run was interrupted"
        )
    return DeerFlowRunError(
        message or f"DeerFlow run ended with status {status}"
    )


def _safe_stop_entry(
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
) -> None:
    try:
        handler.stop_entry(invocation)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to stop DeerFlow ENTRY", exc_info=True)


def _safe_fail_entry(
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
    exc: BaseException,
) -> None:
    try:
        handler.fail_entry(invocation, _error_from_exception(exc))
    except Exception:  # noqa: BLE001
        logger.debug("Failed to fail DeerFlow ENTRY", exc_info=True)


def _start_entry_with_agent_baggage(
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
) -> None:
    """Start ENTRY while exposing its resolved agent name to child spans."""
    parent_context = otel_context.get_current()
    agent_name = invocation.attributes.get(GEN_AI_AGENT_NAME)
    if agent_name:
        parent_context = baggage.set_baggage(
            GEN_AI_AGENT_NAME,
            agent_name,
            parent_context,
        )
    handler.start_entry(invocation, context=parent_context)


class _GatewayRunAgentWrapper:
    """Create one ENTRY around the asynchronous Gateway worker run."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _entry_is_active():
            return await wrapped(*args, **kwargs)

        record = _call_arg(args, kwargs, 2, "record")
        graph_input = _call_arg(args, kwargs, 5, "graph_input")
        config = _call_arg(args, kwargs, 6, "config", {})
        if record is None:
            return await wrapped(*args, **kwargs)

        invocation = _gateway_invocation(record, graph_input, config)
        depth_token = _ENTRY_DEPTH.set(_ENTRY_DEPTH.get() + 1)
        try:
            _start_entry_with_agent_baggage(self._handler, invocation)
        except Exception:  # noqa: BLE001
            _ENTRY_DEPTH.reset(depth_token)
            logger.debug(
                "Failed to start DeerFlow Gateway ENTRY", exc_info=True
            )
            return await wrapped(*args, **kwargs)

        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            status = _run_status_value(record)
            if status in {"unknown", "pending", "running"}:
                status = (
                    "interrupted"
                    if isinstance(exc, asyncio.CancelledError)
                    else "error"
                )
            invocation.attributes[DEERFLOW_RUN_STATUS] = status
            _safe_fail_entry(self._handler, invocation, exc)
            raise
        else:
            status = _run_status_value(record)
            invocation.attributes[DEERFLOW_RUN_STATUS] = status
            if should_capture_content():
                invocation.output_messages = to_output_messages(
                    getattr(record, "last_ai_message", None)
                )
            if status == "success":
                _safe_stop_entry(self._handler, invocation)
            else:
                _safe_fail_entry(
                    self._handler,
                    invocation,
                    _run_status_error(record, status),
                )
            return result
        finally:
            _ENTRY_DEPTH.reset(depth_token)


class _StreamOutput:
    """Accumulate the final embedded assistant message when capture is on."""

    def __init__(self) -> None:
        self._chunks: dict[str, list[str]] = {}
        self._last_id: str | None = None

    def observe(
        self,
        event: Any,
        invocation: EntryInvocation,
    ) -> None:
        if getattr(event, "type", None) != "messages-tuple":
            return
        data = getattr(event, "data", None)
        if not isinstance(data, dict) or data.get("type") != "ai":
            return
        if invocation.response_time_to_first_token is None:
            invocation.response_time_to_first_token = int(
                (timeit.default_timer() - invocation.monotonic_start_s)
                * 1_000_000_000
            )
        raw_content = data.get("content")
        if not isinstance(raw_content, str) or not raw_content:
            return
        if not should_capture_content():
            return
        content = raw_content
        message_id = non_empty_string(data.get("id")) or ""
        self._chunks.setdefault(message_id, []).append(content)
        self._last_id = message_id

    def finish(self, invocation: EntryInvocation) -> None:
        if not should_capture_content() or self._last_id is None:
            return
        output = "".join(self._chunks.get(self._last_id, ()))
        invocation.output_messages = to_output_messages(output)


def _start_isolated_entry(
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
) -> Token[int]:
    depth_token = _ENTRY_DEPTH.set(_ENTRY_DEPTH.get() + 1)
    try:
        _start_entry_with_agent_baggage(handler, invocation)
    except BaseException:
        _ENTRY_DEPTH.reset(depth_token)
        raise
    return depth_token


def _finish_isolated_entry(
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
    depth_token: Token[int],
    deerflow_trace_token: Any,
    error: BaseException | None,
) -> None:
    try:
        if error is None:
            _safe_stop_entry(handler, invocation)
        else:
            if isinstance(error, (GeneratorExit, asyncio.CancelledError)):
                status = "interrupted"
            elif isinstance(error, TimeoutError):
                status = "timeout"
            else:
                status = "error"
            invocation.attributes[DEERFLOW_RUN_STATUS] = status
            _safe_fail_entry(handler, invocation, error)
    finally:
        _ENTRY_DEPTH.reset(depth_token)
        _reset_deerflow_trace_id(deerflow_trace_token)


def _close_iterator(inner: Iterator[Any]) -> None:
    close = getattr(inner, "close", None)
    if callable(close):
        close()


def _iterate_in_context(
    inner: Iterator[Any],
    stream_context: Context,
) -> Generator[Any, None, None]:
    """Advance and close an iterator inside its call-time context."""
    try:
        while True:
            try:
                event = stream_context.run(next, inner)
            except StopIteration:
                break
            yield event
    finally:
        try:
            stream_context.run(_close_iterator, inner)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to close DeerFlow stream", exc_info=True)


def _entry_stream(
    inner: Iterator[Any],
    handler: ExtendedTelemetryHandler,
    invocation: EntryInvocation,
    entry_context: Context,
    deerflow_trace_id: str | None,
) -> Generator[Any, None, None]:
    """Advance a DeerFlow iterator only inside an isolated OTel context."""
    output = _StreamOutput()
    deerflow_trace_token = entry_context.run(
        _bind_deerflow_trace_id,
        deerflow_trace_id,
    )
    try:
        depth_token = entry_context.run(
            _start_isolated_entry,
            handler,
            invocation,
        )
    except Exception:  # noqa: BLE001
        entry_context.run(
            _reset_deerflow_trace_id,
            deerflow_trace_token,
        )
        logger.debug("Failed to start DeerFlow Client ENTRY", exc_info=True)
        yield from _iterate_in_context(inner, entry_context)
        return

    try:
        while True:
            try:
                event = entry_context.run(next, inner)
            except StopIteration:
                break
            output.observe(event, invocation)
            yield event
    except GeneratorExit as exc:
        try:
            entry_context.run(_close_iterator, inner)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to close DeerFlow stream", exc_info=True)
        entry_context.run(
            _finish_isolated_entry,
            handler,
            invocation,
            depth_token,
            deerflow_trace_token,
            exc,
        )
        raise
    except BaseException as exc:
        entry_context.run(
            _finish_isolated_entry,
            handler,
            invocation,
            depth_token,
            deerflow_trace_token,
            exc,
        )
        raise
    else:
        output.finish(invocation)
        invocation.attributes[DEERFLOW_RUN_STATUS] = "success"
        entry_context.run(
            _finish_isolated_entry,
            handler,
            invocation,
            depth_token,
            deerflow_trace_token,
            None,
        )


class _ClientStreamWrapper:
    """Create an isolated, iterator-lifecycle ENTRY for embedded streams."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        entry_context = copy_context()
        if _entry_is_active():
            inner = iter(wrapped(*args, **kwargs))
            return _iterate_in_context(inner, entry_context)

        call_kwargs = dict(kwargs)
        thread_id = _call_arg(args, call_kwargs, 1, "thread_id")
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            call_kwargs["thread_id"] = thread_id

        message = _call_arg(args, call_kwargs, 0, "message")
        assistant_id = getattr(instance, "_agent_name", None)
        agent_name = (
            assistant_id or call_kwargs.get("run_name") or "lead-agent"
        )
        deerflow_trace_id = _client_trace_id(instance)
        invocation = create_entry_invocation(
            thread_id=thread_id,
            user_id=_effective_user_id(),
            agent_name=agent_name,
            assistant_id=assistant_id,
            deerflow_trace_id=deerflow_trace_id,
            input_value=message,
        )
        inner = iter(wrapped(*args, **call_kwargs))
        return _entry_stream(
            inner,
            self._handler,
            invocation,
            entry_context,
            deerflow_trace_id,
        )


def _patch_location(
    module_name: str,
    target: str,
    wrapper: Any,
) -> bool:
    try:
        importlib.import_module(module_name)
        wrap_function_wrapper(module_name, target, wrapper)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Could not patch DeerFlow target %s.%s",
            module_name,
            target,
            exc_info=True,
        )
        return False
    _patched_locations.append((module_name, target))
    return True


def instrument_deerflow(handler: ExtendedTelemetryHandler) -> bool:
    """Patch installed DeerFlow 2.x creation and application entry points."""
    patched = False
    for module_name, target in CREATE_AGENT_ALIASES:
        patched = (
            _patch_location(
                module_name,
                target,
                _create_agent_alias_wrapper,
            )
            or patched
        )

    gateway_wrapper = _GatewayRunAgentWrapper(handler)
    for module_name, target in GATEWAY_RUN_AGENT_ALIASES:
        location_patched = _patch_location(
            module_name,
            target,
            gateway_wrapper,
        )
        if location_patched:
            owner, attribute = _resolve_patch_owner(module_name, target)
            _owned_gateway_wrappers.append(getattr(owner, attribute))
        patched = location_patched or patched
    for module_name, target in GATEWAY_LOADED_RUN_AGENT_ALIASES:
        if module_name in sys.modules:
            location_patched = _patch_location(
                module_name,
                target,
                gateway_wrapper,
            )
            if location_patched:
                owner, attribute = _resolve_patch_owner(module_name, target)
                _owned_gateway_wrappers.append(getattr(owner, attribute))
            patched = location_patched or patched
    patched = (
        _patch_location(
            *CLIENT_STREAM,
            _ClientStreamWrapper(handler),
        )
        or patched
    )
    return patched


def _resolve_patch_owner(module_name: str, target: str) -> tuple[Any, str]:
    owner: Any = importlib.import_module(module_name)
    parts = target.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def _restore_graphs() -> None:
    tracked = [
        *_original_graph_markers.items(),
        *_strong_graph_markers,
    ]
    for graph, original_markers in tracked:
        for attr_name, original_value in original_markers.items():
            try:
                if original_value is _MISSING:
                    if hasattr(graph, attr_name):
                        delattr(graph, attr_name)
                else:
                    setattr(graph, attr_name, original_value)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Could not restore DeerFlow graph marker",
                    exc_info=True,
                )
    _original_graph_markers.clear()
    _strong_graph_markers.clear()


def _restore_late_gateway_aliases() -> None:
    """Unwrap Gateway aliases cached after instrumentation completed."""
    patched_locations = set(_patched_locations)
    for module_name, target in GATEWAY_LOADED_RUN_AGENT_ALIASES:
        if (
            module_name not in sys.modules
            or (module_name, target) in patched_locations
        ):
            continue
        try:
            owner, attribute = _resolve_patch_owner(module_name, target)
            candidate = getattr(owner, attribute)
            if any(
                candidate is wrapper for wrapper in _owned_gateway_wrappers
            ):
                unwrap(owner, attribute)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not restore late DeerFlow Gateway alias %s.%s",
                module_name,
                target,
                exc_info=True,
            )


def uninstrument_deerflow() -> None:
    """Restore all DeerFlow module targets and existing graph markers."""
    _restore_late_gateway_aliases()
    for module_name, target in reversed(_patched_locations):
        try:
            owner, attribute = _resolve_patch_owner(module_name, target)
            unwrap(owner, attribute)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not restore DeerFlow target %s.%s",
                module_name,
                target,
                exc_info=True,
            )
    _patched_locations.clear()
    _owned_gateway_wrappers.clear()
    _restore_graphs()
