# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""LangChain callback sidecar that enriches existing LoongSuite spans."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from importlib import import_module
from typing import Any, Callable

from wrapt import wrap_function_wrapper

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap

from ._attributes import (
    FRAMEWORK_NAME,
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_NAME,
    GEN_AI_AGENT_TYPE,
    GEN_AI_FRAMEWORK,
    GEN_AI_FRAMEWORK_VERSION,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    METADATA_DEEPAGENTS_VERSION,
    METADATA_LC_AGENT_NAME,
    METADATA_LS_AGENT_TYPE,
    METADATA_LS_INTEGRATION,
    METADATA_VERSIONS,
    SUBAGENT_TYPE,
    TASK_TOOL_NAME,
    TOOL_TYPE_AGENT,
)
from ._utils import (
    current_subagent_registry,
    obj_get,
    safe_set_attribute,
)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ModuleNotFoundError:
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

_logger = logging.getLogger(__name__)
_is_enricher_patched = False
_handler: "DeepAgentsEnricherCallbackHandler | None" = None


class DeepAgentsEnricherCallbackHandler(BaseCallbackHandler):  # type: ignore[misc,valid-type]
    """Sidecar callback that writes deepagents metadata to active spans."""

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del serialized, inputs, run_id, parent_run_id, tags, kwargs
        self._enrich_agent_or_chain(metadata or {})

    async def on_chain_start_async(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del serialized, inputs, run_id, parent_run_id, tags, kwargs
        self._enrich_agent_or_chain(metadata or {})

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del input_str, run_id, parent_run_id, tags, metadata
        self._enrich_tool(serialized, kwargs)

    async def on_tool_start_async(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del input_str, run_id, parent_run_id, tags, metadata
        self._enrich_tool(serialized, kwargs)

    def _enrich_agent_or_chain(self, metadata: Mapping[str, Any]) -> None:
        if obj_get(metadata, METADATA_LS_INTEGRATION) != FRAMEWORK_NAME:
            return

        current_span = trace.get_current_span()
        safe_set_attribute(current_span, GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
        version = _version_from_metadata(metadata)
        if version:
            safe_set_attribute(current_span, GEN_AI_FRAMEWORK_VERSION, version)

        agent_name = obj_get(metadata, METADATA_LC_AGENT_NAME)
        if agent_name:
            safe_set_attribute(current_span, GEN_AI_AGENT_NAME, str(agent_name))

        if obj_get(metadata, METADATA_LS_AGENT_TYPE) == SUBAGENT_TYPE:
            safe_set_attribute(current_span, GEN_AI_AGENT_TYPE, SUBAGENT_TYPE)
            description = current_subagent_registry().get(str(agent_name))
            if description:
                safe_set_attribute(
                    current_span,
                    GEN_AI_AGENT_DESCRIPTION,
                    description,
                )

    def _enrich_tool(
        self,
        serialized: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> None:
        name = (
            obj_get(serialized or {}, "name")
            or obj_get(kwargs, "name")
            or obj_get(obj_get(serialized or {}, "kwargs", {}), "name")
        )
        if name != TASK_TOOL_NAME:
            return
        current_span = trace.get_current_span()
        safe_set_attribute(current_span, GEN_AI_TOOL_NAME, TASK_TOOL_NAME)
        safe_set_attribute(current_span, GEN_AI_TOOL_TYPE, TOOL_TYPE_AGENT)


def install_enricher_callback() -> None:
    global _handler, _is_enricher_patched  # noqa: PLW0603
    if _is_enricher_patched:
        return

    try:
        import_module("langchain_core.callbacks")
    except ModuleNotFoundError as exc:
        if exc.name == "langchain_core":
            _logger.warning(
                "langchain_core is not installed; deepagents enricher skipped."
            )
            return
        raise

    _handler = DeepAgentsEnricherCallbackHandler()
    wrap_function_wrapper(
        module="langchain_core.callbacks",
        name="BaseCallbackManager.__init__",
        wrapper=_BaseCallbackManagerInit(_handler),
    )
    _is_enricher_patched = True


def uninstall_enricher_callback() -> None:
    global _handler, _is_enricher_patched  # noqa: PLW0603
    if not _is_enricher_patched:
        _handler = None
        return
    with suppress(Exception):
        import langchain_core.callbacks  # noqa: PLC0415

        unwrap(langchain_core.callbacks.BaseCallbackManager, "__init__")
    _handler = None
    _is_enricher_patched = False


class _BaseCallbackManagerInit:
    __slots__ = ("_handler",)

    def __init__(self, handler: DeepAgentsEnricherCallbackHandler) -> None:
        self._handler = handler

    def __call__(
        self,
        wrapped: Callable[..., None],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in getattr(instance, "inheritable_handlers", ()):
            if isinstance(handler, DeepAgentsEnricherCallbackHandler):
                return
        instance.add_handler(self._handler, True)


def _version_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    versions = obj_get(metadata, METADATA_VERSIONS, {})
    version = obj_get(versions, METADATA_DEEPAGENTS_VERSION)
    return str(version) if version else None
