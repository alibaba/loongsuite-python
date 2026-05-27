# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""ENTRY span patch for ``deepagents.graph.create_deep_agent``."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import suppress
from importlib import import_module
from typing import Any, Callable

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.types import Error

from ._attributes import (
    BUILD_TASK_TOOL_MODULE,
    BUILD_TASK_TOOL_NAME,
    CREATE_DEEP_AGENT_MODULE,
    CREATE_DEEP_AGENT_NAME,
    FRAMEWORK_NAME,
    GEN_AI_AGENT_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_FRAMEWORK_VERSION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_SPAN_KIND,
    GRAPH_ATTR,
    GRAPH_METADATA_ATTR,
    GRAPH_METHODS_WRAPPED_ATTR,
    GRAPH_ORIGINAL_METHODS_ATTR,
    GRAPH_REGISTRY_ATTR,
    GRAPH_VERSION_ATTR,
    LANGGRAPH_REACT_AGENT_METADATA_KEY,
    METADATA_LS_AGENT_TYPE,
    METADATA_SUBAGENT_DESCRIPTION,
    SPAN_KIND_ENTRY,
    SUBAGENT_TYPE,
)
from ._utils import (
    active_span_is_entry_or_agent,
    config_from_call,
    config_with_langgraph_react_metadata,
    create_graph_metadata,
    detect_deepagents_version,
    entry_attributes,
    extract_subagent_registry,
    inject_langgraph_react_metadata,
    input_messages_from_value,
    input_value_from_call,
    output_messages_from_value,
    prime_entry_span,
    reset_current_subagent_registry,
    root_agent_name,
    safe_set_attribute,
    session_id_from_config,
    set_current_subagent_registry,
)

_logger = logging.getLogger(__name__)
_handler: ExtendedTelemetryHandler | None = None
_is_entry_patched = False
_TOP_LEVEL_MODULE = "deepagents"
_MISSING = object()
_top_level_original: Any = _MISSING
_top_level_patched = False
_is_subagent_task_patched = False


def instrument_entry_patch(handler: ExtendedTelemetryHandler) -> None:
    """Patch ``create_deep_agent`` and retain the util-genai handler."""
    global _handler, _is_entry_patched  # noqa: PLW0603
    _handler = handler
    if _is_entry_patched:
        return
    try:
        wrap_function_wrapper(
            CREATE_DEEP_AGENT_MODULE,
            CREATE_DEEP_AGENT_NAME,
            _create_deep_agent_wrapper,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "deepagents" or exc.name == CREATE_DEEP_AGENT_MODULE:
            _logger.warning(
                "deepagents is not installed; create_deep_agent ENTRY patch skipped."
            )
            return
        raise
    except AttributeError:
        _logger.warning(
            "%s.%s not found; ENTRY patch skipped.",
            CREATE_DEEP_AGENT_MODULE,
            CREATE_DEEP_AGENT_NAME,
        )
        return
    _sync_top_level_create_deep_agent()
    _instrument_subagent_task_tool()
    _is_entry_patched = True


def uninstrument_entry_patch() -> None:
    """Remove the create_deep_agent patch.

    Graph instances returned while instrumented keep their instance-level
    wrappers. They are disabled by losing the global handler and are not
    reachable from here without retaining application object references.
    """
    global _handler, _is_entry_patched  # noqa: PLW0603
    _handler = None
    if not _is_entry_patched:
        return
    with suppress(Exception):
        module = import_module(CREATE_DEEP_AGENT_MODULE)
        unwrap(module, CREATE_DEEP_AGENT_NAME)
    _uninstrument_subagent_task_tool()
    _restore_top_level_create_deep_agent()
    _is_entry_patched = False


def _sync_top_level_create_deep_agent() -> None:
    """Point ``deepagents.create_deep_agent`` at the wrapped graph export."""
    global _top_level_original, _top_level_patched  # noqa: PLW0603
    top_level_module = sys.modules.get(_TOP_LEVEL_MODULE)
    graph_module = sys.modules.get(CREATE_DEEP_AGENT_MODULE)
    if top_level_module is None or graph_module is None:
        return

    _top_level_original = getattr(
        top_level_module,
        CREATE_DEEP_AGENT_NAME,
        _MISSING,
    )
    wrapped_create_deep_agent = getattr(graph_module, CREATE_DEEP_AGENT_NAME, None)
    if wrapped_create_deep_agent is None:
        return
    try:
        setattr(top_level_module, CREATE_DEEP_AGENT_NAME, wrapped_create_deep_agent)
    except Exception:  # noqa: BLE001
        _logger.debug("Failed to sync deepagents top-level export", exc_info=True)
        return
    _top_level_patched = True


def _restore_top_level_create_deep_agent() -> None:
    """Restore the top-level ``create_deep_agent`` export after unwrap."""
    global _top_level_original, _top_level_patched  # noqa: PLW0603
    if not _top_level_patched:
        return
    top_level_module = sys.modules.get(_TOP_LEVEL_MODULE)
    if top_level_module is None:
        _top_level_original = _MISSING
        _top_level_patched = False
        return
    try:
        if _top_level_original is _MISSING:
            delattr(top_level_module, CREATE_DEEP_AGENT_NAME)
        else:
            setattr(top_level_module, CREATE_DEEP_AGENT_NAME, _top_level_original)
    except Exception:  # noqa: BLE001
        _logger.debug("Failed to restore deepagents top-level export", exc_info=True)
    finally:
        _top_level_original = _MISSING
        _top_level_patched = False


def _create_deep_agent_wrapper(
    wrapped: Callable[..., Any],
    _instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    graph = wrapped(*args, **kwargs)
    metadata = create_graph_metadata(graph, name=kwargs.get("name"))
    registry = extract_subagent_registry(kwargs.get("subagents"))
    _mark_graph(graph, metadata, registry)
    _wrap_graph_methods(graph, metadata, registry)
    return graph


def _instrument_subagent_task_tool() -> None:
    """Patch deepagents subagent task construction to mark nested graphs."""
    global _is_subagent_task_patched  # noqa: PLW0603
    if _is_subagent_task_patched:
        return
    try:
        wrap_function_wrapper(
            BUILD_TASK_TOOL_MODULE,
            BUILD_TASK_TOOL_NAME,
            _build_task_tool_wrapper,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "deepagents" or exc.name == BUILD_TASK_TOOL_MODULE:
            _logger.debug(
                "deepagents subagent middleware is not installed; "
                "SubAgent task patch skipped."
            )
            return
        raise
    except AttributeError:
        _logger.debug(
            "%s.%s not found; SubAgent task patch skipped.",
            BUILD_TASK_TOOL_MODULE,
            BUILD_TASK_TOOL_NAME,
        )
        return
    _is_subagent_task_patched = True


def _uninstrument_subagent_task_tool() -> None:
    global _is_subagent_task_patched  # noqa: PLW0603
    if not _is_subagent_task_patched:
        return
    with suppress(Exception):
        module = import_module(BUILD_TASK_TOOL_MODULE)
        unwrap(module, BUILD_TASK_TOOL_NAME)
    _is_subagent_task_patched = False


def _build_task_tool_wrapper(
    wrapped: Callable[..., Any],
    _instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    _mark_subagent_specs(_subagent_specs_from_call(args, kwargs))
    return wrapped(*args, **kwargs)


def _subagent_specs_from_call(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    if args:
        return args[0]
    return kwargs.get("subagents")


def _mark_subagent_specs(subagents: Any) -> None:
    for spec in subagents or ():
        try:
            name = spec.get("name") if isinstance(spec, dict) else None
            description = (
                spec.get("description") if isinstance(spec, dict) else None
            )
            runnable = spec.get("runnable") if isinstance(spec, dict) else None
            if not name or runnable is None:
                continue
            spec["runnable"] = _mark_subagent_runnable(
                runnable,
                name=str(name),
                description=str(description) if description else None,
            )
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to mark deepagents SubAgent graph", exc_info=True)


def _mark_subagent_runnable(
    runnable: Any,
    *,
    name: str,
    description: str | None,
) -> Any:
    metadata = create_graph_metadata(runnable, name=name)
    metadata.setdefault(METADATA_LS_AGENT_TYPE, SUBAGENT_TYPE)
    metadata.setdefault(LANGGRAPH_REACT_AGENT_METADATA_KEY, True)
    if description:
        metadata.setdefault(METADATA_SUBAGENT_DESCRIPTION, description)

    registry = {name: description} if description else {}
    _mark_graph(runnable, metadata, registry)
    proxy = _SubagentRunnableProxy(runnable, metadata)
    _mark_graph(proxy, metadata, registry)
    return proxy


class _SubagentRunnableProxy:
    """Proxy that injects deepagents metadata before nested SubAgent calls."""

    __slots__ = ("_metadata", "_runnable")

    def __init__(self, runnable: Any, metadata: Mapping[str, Any]) -> None:
        self._runnable = runnable
        self._metadata = dict(metadata)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)

    def invoke(self, value: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._runnable.invoke(
            value,
            config_with_langgraph_react_metadata(config, self._metadata),
            **kwargs,
        )

    async def ainvoke(
        self,
        value: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._runnable.ainvoke(
            value,
            config_with_langgraph_react_metadata(config, self._metadata),
            **kwargs,
        )


def _mark_graph(
    graph: Any,
    metadata: dict[str, Any],
    registry: dict[str, str],
) -> None:
    version = detect_deepagents_version(metadata)
    with suppress(Exception):
        setattr(graph, GRAPH_ATTR, True)
        setattr(graph, LANGGRAPH_REACT_AGENT_METADATA_KEY, True)
        setattr(graph, GRAPH_METADATA_ATTR, metadata)
        setattr(graph, GRAPH_REGISTRY_ATTR, registry)
        if version:
            setattr(graph, GRAPH_VERSION_ATTR, version)


def _wrap_graph_methods(
    graph: Any,
    metadata: dict[str, Any],
    registry: dict[str, str],
) -> None:
    if getattr(graph, GRAPH_METHODS_WRAPPED_ATTR, False):
        return

    originals: dict[str, Any] = {}
    for method_name in ("invoke", "ainvoke", "stream", "astream"):
        original = getattr(graph, method_name, None)
        if original is None:
            continue
        originals[method_name] = original
        wrapper = _make_method_wrapper(
            graph=graph,
            method_name=method_name,
            original=original,
            metadata=metadata,
            registry=registry,
        )
        try:
            setattr(graph, method_name, wrapper)
        except Exception:  # noqa: BLE001
            _logger.debug(
                "Failed to wrap deepagents graph method %s", method_name, exc_info=True
            )

    with suppress(Exception):
        setattr(graph, GRAPH_ORIGINAL_METHODS_ATTR, originals)
        setattr(graph, GRAPH_METHODS_WRAPPED_ATTR, True)


def _make_method_wrapper(
    *,
    graph: Any,
    method_name: str,
    original: Callable[..., Any],
    metadata: dict[str, Any],
    registry: dict[str, str],
) -> Callable[..., Any]:
    if method_name == "ainvoke":

        async def ainvoke_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _call_async_with_entry(
                graph, method_name, original, metadata, registry, args, kwargs
            )

        return ainvoke_wrapper

    if method_name == "stream":

        def stream_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
            yield from _call_stream_with_entry(
                graph, method_name, original, metadata, registry, args, kwargs
            )

        return stream_wrapper

    if method_name == "astream":

        async def astream_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            async for chunk in _call_astream_with_entry(
                graph, method_name, original, metadata, registry, args, kwargs
            ):
                yield chunk

        return astream_wrapper

    def invoke_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _call_sync_with_entry(
            graph, method_name, original, metadata, registry, args, kwargs
        )

    return invoke_wrapper


def _create_entry_invocation(
    graph: Any,
    method_name: str,
    metadata: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> EntryInvocation:
    config = config_from_call(args, kwargs)
    invocation = EntryInvocation(
        session_id=session_id_from_config(config),
        input_messages=input_messages_from_value(input_value_from_call(args, kwargs)),
        attributes=entry_attributes(method_name=method_name, metadata=metadata),
    )
    agent_name = root_agent_name(graph, metadata)
    if agent_name:
        invocation.attributes[GEN_AI_AGENT_NAME] = agent_name
    return invocation


def _start_entry(
    graph: Any,
    method_name: str,
    metadata: dict[str, Any],
    registry: dict[str, str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[EntryInvocation | None, Any]:
    if _handler is None or active_span_is_entry_or_agent():
        return None, None

    invocation = _create_entry_invocation(graph, method_name, metadata, args, kwargs)
    token = set_current_subagent_registry(registry)
    _handler.start_entry(invocation)
    prime_entry_span(invocation.span, method_name=method_name, metadata=metadata)
    _rename_entry_span(invocation, graph, method_name, metadata)
    return invocation, token


def _rename_entry_span(
    invocation: EntryInvocation,
    graph: Any,
    method_name: str,
    metadata: dict[str, Any],
) -> None:
    span = invocation.span
    if span is None:
        return
    version = detect_deepagents_version(metadata)
    with suppress(Exception):
        span.update_name(f"deepagents.{method_name}")
    safe_set_attribute(span, GEN_AI_SPAN_KIND, SPAN_KIND_ENTRY)
    safe_set_attribute(span, GEN_AI_OPERATION_NAME, method_name)
    safe_set_attribute(span, GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
    safe_set_attribute(span, GEN_AI_FRAMEWORK_VERSION, version)
    safe_set_attribute(span, GEN_AI_AGENT_NAME, root_agent_name(graph, metadata))


def _finish_entry(
    invocation: EntryInvocation | None,
    token: Any,
    result: Any = None,
    exc: Exception | None = None,
) -> None:
    if token is not None:
        reset_current_subagent_registry(token)
    if invocation is None or _handler is None:
        return
    if exc is None:
        invocation.output_messages = output_messages_from_value(result)
        _handler.stop_entry(invocation)
        return
    _handler.fail_entry(invocation, Error(message=str(exc), type=type(exc)))


def _call_sync_with_entry(
    graph: Any,
    method_name: str,
    original: Callable[..., Any],
    metadata: dict[str, Any],
    registry: dict[str, str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    invocation, token = _start_entry(
        graph, method_name, metadata, registry, args, kwargs
    )
    args, kwargs = inject_langgraph_react_metadata(args, kwargs, metadata)
    try:
        result = original(*args, **kwargs)
    except Exception as exc:
        _finish_entry(invocation, token, exc=exc)
        raise
    _finish_entry(invocation, token, result=result)
    return result


async def _call_async_with_entry(
    graph: Any,
    method_name: str,
    original: Callable[..., Any],
    metadata: dict[str, Any],
    registry: dict[str, str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    invocation, token = _start_entry(
        graph, method_name, metadata, registry, args, kwargs
    )
    args, kwargs = inject_langgraph_react_metadata(args, kwargs, metadata)
    try:
        result = await original(*args, **kwargs)
    except Exception as exc:
        _finish_entry(invocation, token, exc=exc)
        raise
    _finish_entry(invocation, token, result=result)
    return result


def _call_stream_with_entry(
    graph: Any,
    method_name: str,
    original: Callable[..., Any],
    metadata: dict[str, Any],
    registry: dict[str, str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Iterator[Any]:
    invocation, token = _start_entry(
        graph, method_name, metadata, registry, args, kwargs
    )
    args, kwargs = inject_langgraph_react_metadata(args, kwargs, metadata)
    last_chunk = None
    try:
        for chunk in original(*args, **kwargs):
            last_chunk = chunk
            yield chunk
    except Exception as exc:
        _finish_entry(invocation, token, exc=exc)
        raise
    _finish_entry(invocation, token, result=last_chunk)


async def _call_astream_with_entry(
    graph: Any,
    method_name: str,
    original: Callable[..., Any],
    metadata: dict[str, Any],
    registry: dict[str, str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AsyncIterator[Any]:
    invocation, token = _start_entry(
        graph, method_name, metadata, registry, args, kwargs
    )
    args, kwargs = inject_langgraph_react_metadata(args, kwargs, metadata)
    last_chunk = None
    try:
        async for chunk in original(*args, **kwargs):
            last_chunk = chunk
            yield chunk
    except Exception as exc:
        _finish_entry(invocation, token, exc=exc)
        raise
    _finish_entry(invocation, token, result=last_chunk)
