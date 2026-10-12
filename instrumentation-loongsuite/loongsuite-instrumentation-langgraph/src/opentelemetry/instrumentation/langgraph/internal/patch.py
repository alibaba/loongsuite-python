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

"""``wrapt``-based wrappers for LangGraph instrumentation.

All wrappers follow the ``wrapt`` convention::

    def wrapper(wrapped, instance, args, kwargs) -> ...

Three patch targets:

1. ``create_react_agent`` — sets ``_loongsuite_react_agent = True`` on the
   compiled ``CompiledStateGraph`` so downstream instrumentation recognises it.

2. ``Pregel.stream`` / ``Pregel.astream`` — injects
   either the existing ReAct marker or an opt-in harness's framework and
   decision-node semantics into a copy of ``RunnableConfig``. The metadata
   then flows through LangChain callbacks to ``Run.metadata``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

AGENT_FRAMEWORK_METADATA_KEY = "_loongsuite_agent_framework"
AGENT_STEP_NODE_METADATA_KEY = "_loongsuite_agent_step_node"
REACT_AGENT_METADATA_KEY = "_loongsuite_react_agent"


# ---------------------------------------------------------------------------
# create_react_agent
# ---------------------------------------------------------------------------


def _create_react_agent_wrapper(
    wrapped: Any, _instance: Any, args: Any, kwargs: Any
) -> Any:
    """``wrapt`` wrapper for ``create_react_agent``.

    Calls the original function, then marks the returned graph with
    ``_loongsuite_react_agent = True``.
    """
    graph = wrapped(*args, **kwargs)
    setattr(graph, REACT_AGENT_METADATA_KEY, True)
    logger.debug(
        "[INSTRUMENTATION] create_react_agent patched graph: name=%r, %s=%r",
        getattr(graph, "name", None),
        REACT_AGENT_METADATA_KEY,
        True,
    )
    return graph


# ---------------------------------------------------------------------------
# Pregel.stream / astream — metadata injection
# ---------------------------------------------------------------------------


def _copy_config_and_metadata(
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return copy-on-write config and metadata dictionaries."""
    # Inline import: langchain_core is a transitive dependency of langgraph;
    # importing here avoids module-level coupling.
    from langchain_core.runnables.config import (  # noqa: PLC0415
        ensure_config,
    )

    config = ensure_config(config)
    config = {**config}
    return config, dict(config.get("metadata") or {})


def _inject_react_metadata(config: Any) -> Any:
    """Return a new config carrying the existing prebuilt ReAct marker."""
    config, metadata = _copy_config_and_metadata(config)
    metadata.setdefault(REACT_AGENT_METADATA_KEY, True)
    config["metadata"] = metadata
    return config


def _inject_agent_semantics(config: Any, semantics: tuple[str, str]) -> Any:
    """Return a new config carrying an opt-in harness's scalar semantics."""
    config, metadata = _copy_config_and_metadata(config)
    metadata[AGENT_FRAMEWORK_METADATA_KEY] = semantics[0]
    metadata[AGENT_STEP_NODE_METADATA_KEY] = semantics[1]
    config["metadata"] = metadata
    return config


def _get_graph_agent_semantics(graph: Any) -> tuple[str, str] | None:
    """Read validated opt-in agent semantics from graph attributes."""

    framework = getattr(graph, AGENT_FRAMEWORK_METADATA_KEY, None)
    step_node = getattr(graph, AGENT_STEP_NODE_METADATA_KEY, None)
    if not isinstance(framework, str) or not framework.strip():
        return None
    if not isinstance(step_node, str) or not step_node.strip():
        return None
    return framework.strip(), step_node.strip()


def _stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):  # type: ignore[return]
    """``wrapt`` wrapper for ``Pregel.stream``."""
    semantics = _get_graph_agent_semantics(instance)
    if semantics is not None:
        args, kwargs = _rewrite_config(args, kwargs, semantics)
    elif getattr(instance, REACT_AGENT_METADATA_KEY, False):
        args, kwargs = _rewrite_config(args, kwargs)
    yield from wrapped(*args, **kwargs)


async def _astream_wrapper(
    wrapped: Any, instance: Any, args: Any, kwargs: Any
):  # type: ignore[return]
    """``wrapt`` wrapper for ``Pregel.astream``."""
    semantics = _get_graph_agent_semantics(instance)
    if semantics is not None:
        args, kwargs = _rewrite_config(args, kwargs, semantics)
    elif getattr(instance, REACT_AGENT_METADATA_KEY, False):
        args, kwargs = _rewrite_config(args, kwargs)
    async for chunk in wrapped(*args, **kwargs):
        yield chunk


def _rewrite_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    semantics: tuple[str, str] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Extract ``config`` from *args*/*kwargs*, inject metadata, put it back."""
    if len(args) > 1:
        config = (
            _inject_agent_semantics(args[1], semantics)
            if semantics is not None
            else _inject_react_metadata(args[1])
        )
        args = (args[0], config) + args[2:]
    else:
        config = (
            _inject_agent_semantics(kwargs.get("config"), semantics)
            if semantics is not None
            else _inject_react_metadata(kwargs.get("config"))
        )
        kwargs = {**kwargs, "config": config}
    return args, kwargs
