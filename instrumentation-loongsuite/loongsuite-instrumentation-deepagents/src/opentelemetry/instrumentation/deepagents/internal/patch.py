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

"""Patch helpers for DeepAgents instrumentation.

DeepAgents builds on ``langchain.agents.create_agent`` and returns
``create_agent(...).with_config(...)``. That final ``with_config`` call creates
a new graph object and drops the marker that LangChain instrumentation places
on the original graph. This module restores the marker on the returned graph
and injects the marker into call-time config, mirroring LangGraph's lightweight
metadata handoff.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from typing import Any, Callable
from weakref import WeakSet

from wrapt import ObjectProxy, wrap_function_wrapper

from opentelemetry.instrumentation.utils import unwrap

logger = logging.getLogger(__name__)

CREATE_DEEP_AGENT_MODULE = "deepagents.graph"
CREATE_DEEP_AGENT_NAME = "create_deep_agent"
REACT_AGENT_METADATA_KEY = "_loongsuite_react_agent"
DEEPAGENTS_METADATA_KEY = "_loongsuite_deepagents_agent"
GRAPH_METHODS_WRAPPED_ATTR = "_loongsuite_deepagents_methods_wrapped"
GRAPH_ORIGINAL_METHODS_ATTR = "_loongsuite_deepagents_original_methods"

_TOP_LEVEL_MODULE = "deepagents"
_MISSING = object()
_top_level_original: Any = _MISSING
_top_level_patched = False
_wrapped_graphs = WeakSet()
_strong_wrapped_graphs: list[Any] = []


def _create_deep_agent_wrapper(
    wrapped: Callable[..., Any],
    _instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    graph = wrapped(*args, **kwargs)
    _mark_graph(graph)
    _wrap_graph_methods(graph)
    return graph


def instrument_create_deep_agent() -> bool:
    """Patch ``deepagents.graph.create_deep_agent`` when available."""
    try:
        module = importlib.import_module(CREATE_DEEP_AGENT_MODULE)
        target = getattr(module, CREATE_DEEP_AGENT_NAME)
    except ModuleNotFoundError as exc:
        if exc.name == "deepagents" or exc.name == CREATE_DEEP_AGENT_MODULE:
            logger.warning(
                "deepagents is not installed; DeepAgents instrumentation skipped."
            )
            return False
        raise
    except AttributeError:
        logger.warning(
            "%s.%s not found; DeepAgents instrumentation skipped.",
            CREATE_DEEP_AGENT_MODULE,
            CREATE_DEEP_AGENT_NAME,
        )
        return False

    if isinstance(target, ObjectProxy):
        logger.debug(
            "Skipping %s.%s (already wrapped)",
            CREATE_DEEP_AGENT_MODULE,
            CREATE_DEEP_AGENT_NAME,
        )
    else:
        wrap_function_wrapper(
            CREATE_DEEP_AGENT_MODULE,
            CREATE_DEEP_AGENT_NAME,
            _create_deep_agent_wrapper,
        )
        logger.debug(
            "Patched %s.%s",
            CREATE_DEEP_AGENT_MODULE,
            CREATE_DEEP_AGENT_NAME,
        )

    _sync_top_level_create_deep_agent()
    return True


def uninstrument_create_deep_agent() -> None:
    """Restore the create_deep_agent patch and top-level export."""
    with suppress(Exception):
        module = importlib.import_module(CREATE_DEEP_AGENT_MODULE)
        unwrap(module, CREATE_DEEP_AGENT_NAME)
    _restore_wrapped_graph_methods()
    _restore_top_level_create_deep_agent()


def _sync_top_level_create_deep_agent() -> None:
    global _top_level_original, _top_level_patched  # noqa: PLW0603

    top_level_module = sys.modules.get(_TOP_LEVEL_MODULE)
    graph_module = sys.modules.get(CREATE_DEEP_AGENT_MODULE)
    if top_level_module is None or graph_module is None:
        return

    wrapped_create_deep_agent = getattr(
        graph_module, CREATE_DEEP_AGENT_NAME, None
    )
    if wrapped_create_deep_agent is None:
        return

    if not _top_level_patched:
        _top_level_original = getattr(
            top_level_module, CREATE_DEEP_AGENT_NAME, _MISSING
        )

    try:
        setattr(
            top_level_module,
            CREATE_DEEP_AGENT_NAME,
            wrapped_create_deep_agent,
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "Failed to sync deepagents top-level export", exc_info=True
        )
        return
    _top_level_patched = True


def _restore_top_level_create_deep_agent() -> None:
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
            setattr(
                top_level_module,
                CREATE_DEEP_AGENT_NAME,
                _top_level_original,
            )
    except Exception:  # noqa: BLE001
        logger.debug(
            "Failed to restore deepagents top-level export", exc_info=True
        )
    finally:
        _top_level_original = _MISSING
        _top_level_patched = False


def _mark_graph(graph: Any) -> None:
    with suppress(Exception):
        setattr(graph, REACT_AGENT_METADATA_KEY, True)
        setattr(graph, DEEPAGENTS_METADATA_KEY, True)


def _wrap_graph_methods(graph: Any) -> None:
    if getattr(graph, GRAPH_METHODS_WRAPPED_ATTR, False):
        return

    originals: dict[str, Callable[..., Any]] = {}
    for method_name in (
        "invoke",
        "ainvoke",
        "stream",
        "astream",
        "with_config",
    ):
        original = getattr(graph, method_name, None)
        if original is None:
            continue
        originals[method_name] = original
        wrapper = _make_method_wrapper(method_name, original)
        try:
            setattr(graph, method_name, wrapper)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to wrap deepagents graph method %s",
                method_name,
                exc_info=True,
            )

    with suppress(Exception):
        setattr(graph, GRAPH_ORIGINAL_METHODS_ATTR, originals)
        setattr(graph, GRAPH_METHODS_WRAPPED_ATTR, True)
    _track_wrapped_graph(graph)


def _track_wrapped_graph(graph: Any) -> None:
    try:
        _wrapped_graphs.add(graph)
    except TypeError:
        _strong_wrapped_graphs.append(graph)


def _restore_wrapped_graph_methods() -> None:
    for graph in [*list(_wrapped_graphs), *_strong_wrapped_graphs]:
        _restore_graph_methods(graph)
    _wrapped_graphs.clear()
    _strong_wrapped_graphs.clear()


def _restore_graph_methods(graph: Any) -> None:
    originals = getattr(graph, GRAPH_ORIGINAL_METHODS_ATTR, None)
    if isinstance(originals, dict):
        for method_name, original in originals.items():
            with suppress(Exception):
                setattr(graph, method_name, original)

    for attr_name in (
        GRAPH_ORIGINAL_METHODS_ATTR,
        GRAPH_METHODS_WRAPPED_ATTR,
        REACT_AGENT_METADATA_KEY,
        DEEPAGENTS_METADATA_KEY,
    ):
        with suppress(Exception):
            delattr(graph, attr_name)


def _make_method_wrapper(
    method_name: str,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    if method_name == "ainvoke":

        async def ainvoke_wrapper(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = _rewrite_config(args, kwargs)
            return await original(*args, **kwargs)

        return ainvoke_wrapper

    if method_name == "stream":

        def stream_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
            args, kwargs = _rewrite_config(args, kwargs)
            return original(*args, **kwargs)

        return stream_wrapper

    if method_name == "astream":

        def astream_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            args, kwargs = _rewrite_config(args, kwargs)
            return original(*args, **kwargs)

        return astream_wrapper

    if method_name == "with_config":

        def with_config_wrapper(*args: Any, **kwargs: Any) -> Any:
            graph = original(*args, **kwargs)
            _mark_graph(graph)
            _wrap_graph_methods(graph)
            return graph

        return with_config_wrapper

    def invoke_wrapper(*args: Any, **kwargs: Any) -> Any:
        args, kwargs = _rewrite_config(args, kwargs)
        return original(*args, **kwargs)

    return invoke_wrapper


def _rewrite_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if len(args) > 1:
        config = _inject_react_metadata(args[1])
        args = (args[0], config) + args[2:]
        return args, kwargs

    kwargs = {**kwargs}
    kwargs["config"] = _inject_react_metadata(kwargs.get("config"))
    return args, kwargs


def _inject_react_metadata(config: Any) -> Any:
    from langchain_core.runnables.config import ensure_config  # noqa: PLC0415

    config = ensure_config(config)
    config = {**config}
    metadata = dict(config.get("metadata") or {})
    metadata.setdefault(DEEPAGENTS_METADATA_KEY, True)
    config["metadata"] = metadata
    return config
