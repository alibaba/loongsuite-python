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

1. ``create_react_agent`` — marks the compiled ``CompiledStateGraph`` with the
   explicit ``langgraph-prebuilt`` agent flavor.

2. ``Pregel.stream`` / ``Pregel.astream`` — injects
   ``metadata["_loongsuite_agent_flavor"]`` into a copy of the
   ``RunnableConfig`` when the graph is a marked agent. This metadata flows
   through LangChain's callback system to ``Run.metadata``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

AGENT_FLAVOR_METADATA_KEY = "_loongsuite_agent_flavor"
REACT_AGENT_METADATA_KEY = "_loongsuite_react_agent"
LANGGRAPH_PREBUILT_AGENT_FLAVOR = "langgraph-prebuilt"


# ---------------------------------------------------------------------------
# create_react_agent
# ---------------------------------------------------------------------------


def _create_react_agent_wrapper(
    wrapped: Any, _instance: Any, args: Any, kwargs: Any
) -> Any:
    """``wrapt`` wrapper for ``create_react_agent``.

    Calls the original function, then marks the returned graph with its
    explicit agent flavor. The legacy boolean remains for compatibility.
    """
    graph = wrapped(*args, **kwargs)
    setattr(
        graph,
        AGENT_FLAVOR_METADATA_KEY,
        LANGGRAPH_PREBUILT_AGENT_FLAVOR,
    )
    setattr(graph, REACT_AGENT_METADATA_KEY, True)
    logger.debug(
        "[INSTRUMENTATION] create_react_agent patched graph: name=%r, %s=%r",
        getattr(graph, "name", None),
        AGENT_FLAVOR_METADATA_KEY,
        LANGGRAPH_PREBUILT_AGENT_FLAVOR,
    )
    return graph


# ---------------------------------------------------------------------------
# Pregel.stream / astream — metadata injection
# ---------------------------------------------------------------------------


def _inject_agent_metadata(config: Any, agent_flavor: str) -> Any:
    """Return a new config with the graph's single agent flavor metadata."""
    # Inline import: langchain_core is a transitive dependency of langgraph;
    # importing here avoids module-level coupling.
    from langchain_core.runnables.config import (  # noqa: PLC0415
        ensure_config,
    )

    config = ensure_config(config)
    config = {**config}
    metadata = dict(config.get("metadata") or {})
    metadata[AGENT_FLAVOR_METADATA_KEY] = agent_flavor
    metadata.setdefault(REACT_AGENT_METADATA_KEY, True)
    config["metadata"] = metadata
    return config


def _get_graph_agent_flavor(graph: Any) -> str | None:
    """Prefer an explicit graph flavor, falling back to the legacy marker."""

    explicit_flavor = getattr(graph, AGENT_FLAVOR_METADATA_KEY, None)
    if isinstance(explicit_flavor, str) and explicit_flavor:
        return explicit_flavor
    if getattr(graph, REACT_AGENT_METADATA_KEY, False):
        return LANGGRAPH_PREBUILT_AGENT_FLAVOR
    return None


def _stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):  # type: ignore[return]
    """``wrapt`` wrapper for ``Pregel.stream``."""
    agent_flavor = _get_graph_agent_flavor(instance)
    if agent_flavor is not None:
        args, kwargs = _rewrite_config(args, kwargs, agent_flavor)
    yield from wrapped(*args, **kwargs)


async def _astream_wrapper(
    wrapped: Any, instance: Any, args: Any, kwargs: Any
):  # type: ignore[return]
    """``wrapt`` wrapper for ``Pregel.astream``."""
    agent_flavor = _get_graph_agent_flavor(instance)
    if agent_flavor is not None:
        args, kwargs = _rewrite_config(args, kwargs, agent_flavor)
    async for chunk in wrapped(*args, **kwargs):
        yield chunk


def _rewrite_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    agent_flavor: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Extract ``config`` from *args*/*kwargs*, inject metadata, put it back."""
    if len(args) > 1:
        config = _inject_agent_metadata(args[1], agent_flavor)
        args = (args[0], config) + args[2:]
    else:
        config = _inject_agent_metadata(kwargs.get("config"), agent_flavor)
        kwargs = {**kwargs, "config": config}
    return args, kwargs
