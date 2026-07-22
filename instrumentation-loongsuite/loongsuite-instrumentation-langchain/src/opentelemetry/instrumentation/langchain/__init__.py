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

"""
LoongSuite LangChain instrumentation supporting ``langchain_core >= 0.1.0``.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument()

    # ... use LangChain as normal ...

    LangChainInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Collection, Type

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.langchain.package import _instruments
from opentelemetry.instrumentation.utils import unwrap

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackManager

    from opentelemetry.instrumentation.langchain.internal._tracer import (
        LoongsuiteTracer,
    )

logger = logging.getLogger(__name__)

__all__ = ["LangChainInstrumentor"]

# Stored originals for AgentExecutor uninstrument: {class: (iter, aiter)}
_patched_agent_executors: dict[type, tuple[Any, Any]] = {}

# Module paths that were successfully patched for create_agent (for uninstrument).
_patched_create_agent_locations: list[tuple[str, str]] = []


def _get_agent_executor_classes() -> list[type]:
    """Return all available AgentExecutor classes to patch.

    Both 0.x and 1.x may coexist:
    - langchain 0.x: AgentExecutor in langchain.agents
    - langchain 1.x + langchain-classic: AgentExecutor in langchain_classic.agents

    We patch both when both exist so either import path works.
    """
    classes: list[type] = []
    try:
        from langchain.agents import AgentExecutor  # noqa: PLC0415

        classes.append(AgentExecutor)
    except ImportError as e:
        logger.debug("langchain.agents.AgentExecutor not available: %s", e)
    try:
        from langchain_classic.agents import AgentExecutor  # noqa: PLC0415

        if AgentExecutor not in classes:
            classes.append(AgentExecutor)
    except ImportError as e:
        logger.debug(
            "langchain_classic.agents.AgentExecutor not available: %s", e
        )
    return classes


def _instrument_agent_executor() -> bool:
    """Apply ReAct step patch to AgentExecutor. Returns True if patched."""
    classes = _get_agent_executor_classes()
    if not classes:
        logger.debug("AgentExecutor not available, skipping ReAct patch")
        return False

    from opentelemetry.instrumentation.langchain.internal.patch import (  # noqa: PLC0415
        _make_aiter_next_step_wrapper,
        _make_iter_next_step_wrapper,
    )

    for cls in classes:
        orig_iter = cls._iter_next_step
        orig_aiter = cls._aiter_next_step
        cls._iter_next_step = _make_iter_next_step_wrapper(orig_iter)
        cls._aiter_next_step = _make_aiter_next_step_wrapper(orig_aiter)
        _patched_agent_executors[cls] = (orig_iter, orig_aiter)

    logger.debug(
        "Patched AgentExecutor._iter_next_step and _aiter_next_step (%d class(es))",
        len(classes),
    )
    return True


def _uninstrument_agent_executor() -> None:
    """Restore original AgentExecutor methods."""
    if not _patched_agent_executors:
        return
    for cls, (orig_iter, orig_aiter) in list(_patched_agent_executors.items()):
        try:
            cls._iter_next_step = orig_iter
            cls._aiter_next_step = orig_aiter
        except Exception:  # noqa: S110
            logger.debug(
                "Failed to restore AgentExecutor methods for %s",
                cls,
                exc_info=True,
            )
    logger.debug(
        "Restored AgentExecutor._iter_next_step and _aiter_next_step (%d class(es))",
        len(_patched_agent_executors),
    )
    _patched_agent_executors.clear()


# ------------------------------------------------------------------
# create_agent patch (langchain >= 1.x)
# ------------------------------------------------------------------

_REACT_AGENT_GRAPH_ATTR = "_loongsuite_react_agent"


def _create_agent_wrapper(
    wrapped: Any, _instance: Any, args: Any, kwargs: Any
) -> Any:
    """``wrapt`` wrapper for ``create_agent``.

    Calls the original factory, then marks the returned graph with
    ``_loongsuite_react_agent = True`` so that the langgraph Pregel
    stream/astream wrapper can inject metadata into ``RunnableConfig``.
    """
    graph = wrapped(*args, **kwargs)
    setattr(graph, _REACT_AGENT_GRAPH_ATTR, True)
    return graph


def _get_create_agent_locations() -> list[tuple[str, str]]:
    """Return (module, attribute) pairs for ``create_agent``.

    langchain >= 1.x exposes ``create_agent`` in ``langchain.agents``.
    """
    locations: list[tuple[str, str]] = []
    try:
        from langchain.agents import create_agent  # noqa: PLC0415, F401

        locations.append(("langchain.agents", "create_agent"))
    except ImportError as exc:
        logger.debug("langchain.agents.create_agent not available: %s", exc)
    return locations


def _instrument_create_agent() -> None:
    """Wrap ``create_agent`` so that the returned graph is marked as a
    ReAct agent, enabling downstream metadata injection by langgraph
    instrumentation.
    """
    locations = _get_create_agent_locations()
    if not locations:
        logger.debug(
            "create_agent not found in langchain.agents; "
            "create_agent instrumentation skipped."
        )
        return

    for module_path, attr_name in locations:
        wrap_function_wrapper(module_path, attr_name, _create_agent_wrapper)
        logger.debug("Patched %s.%s", module_path, attr_name)

    _patched_create_agent_locations.clear()
    _patched_create_agent_locations.extend(locations)


def _uninstrument_create_agent() -> None:
    """Restore original ``create_agent`` functions."""
    for module_path, attr_name in _patched_create_agent_locations:
        try:
            mod = importlib.import_module(module_path)
            unwrap(mod, attr_name)
            logger.debug("Restored %s.%s", module_path, attr_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Failed to restore %s.%s: %s", module_path, attr_name, exc
            )

    _patched_create_agent_locations.clear()


# ------------------------------------------------------------------
# BaseDocumentCompressor patch (rerank / compression)
# ------------------------------------------------------------------


def _instrument_document_compressor(handler: Any) -> None:
    """Wrap all current and future ``BaseDocumentCompressor`` subclasses."""
    from opentelemetry.instrumentation.langchain.internal.patch_rerank import (  # noqa: PLC0415
        instrument_document_compressor,
    )

    instrument_document_compressor(handler)


def _uninstrument_document_compressor() -> None:
    """Restore original ``BaseDocumentCompressor`` methods."""
    from opentelemetry.instrumentation.langchain.internal.patch_rerank import (  # noqa: PLC0415
        uninstrument_document_compressor,
    )

    uninstrument_document_compressor()


class LangChainInstrumentor(BaseInstrumentor):
    """An instrumentor for LangChain."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from opentelemetry.instrumentation.langchain.internal._tracer import (  # noqa: PLC0415
            LoongsuiteTracer,
        )
        from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
            ExtendedTelemetryHandler,
        )

        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(
                cls=LoongsuiteTracer,
                handler=handler,
                tracer_provider=tracer_provider,
            ),
        )

        _instrument_agent_executor()
        _instrument_create_agent()
        _instrument_document_compressor(handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import langchain_core.callbacks  # noqa: PLC0415

            unwrap(langchain_core.callbacks.BaseCallbackManager, "__init__")
            logger.debug("Uninstrumented BaseCallbackManager.__init__")
        except Exception as e:
            logger.warning("Failed to uninstrument BaseCallbackManager: %s", e)

        _uninstrument_agent_executor()
        _uninstrument_create_agent()
        _uninstrument_document_compressor()


class _BaseCallbackManagerInit:
    __slots__ = ("_tracer_instance",)

    def __init__(
        self,
        cls: Type["LoongsuiteTracer"],
        handler: Any,
        tracer_provider: Any = None,
    ):
        self._tracer_instance = cls(
            handler=handler,
            tracer_provider=tracer_provider,
        )

    def __call__(
        self,
        wrapped: Callable[..., None],
        instance: "BaseCallbackManager",
        args: Any,
        kwargs: Any,
    ) -> None:
        wrapped(*args, **kwargs)

        for h in instance.inheritable_handlers:
            if isinstance(h, type(self._tracer_instance)):
                break
        else:
            instance.add_handler(self._tracer_instance, True)
