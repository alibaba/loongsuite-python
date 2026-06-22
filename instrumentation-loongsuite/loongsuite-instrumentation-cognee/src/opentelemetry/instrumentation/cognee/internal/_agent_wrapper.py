"""AGENT wrapper for Cognee ``AgenticRetriever``.

``AgenticRetriever._run_tool_loop`` is the only path that drives a multi-round
ReAct loop in Cognee v1.2.1. We wrap it to produce an AGENT span and to set a
``contextvars.ContextVar`` that the STEP wrapper reads to know it is inside
the ReAct loop.

``_run_tool_loop`` is a private method (prefixed with ``_``), so we guard the
wrap with ``hasattr`` and fall back to wrapping the public method
``get_retrieved_objects`` (which is what callers actually invoke externally).
The public fallback still creates an AGENT span but loses STEP granularity.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.cognee.internal._react_context import (
    reset_react_round,
    set_react_round,
)
from opentelemetry.instrumentation.cognee.internal._util import (
    maybe_capture,
    normalize_kwargs,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


_AGENT_MODULE = "cognee.modules.retrieval.agentic_retriever"
_AGENT_CLASS = "AgenticRetriever"
_AGENT_PRIMARY_METHOD = "_run_tool_loop"
_AGENT_FALLBACK_METHOD = "get_retrieved_objects"


def _resolve_agent_id(instance: Any) -> str:
    try:
        dataset_id = getattr(instance, "dataset_id", None)
        if dataset_id is None:
            return ""
        return str(dataset_id)
    except Exception:
        return ""


def _resolve_tool_definitions(instance: Any, kwargs: dict[str, Any]) -> list:
    tool_names = kwargs.get("tool_names")
    if not tool_names:
        return []
    return [
        {
            "name": str(name),
            "type": "function",
        }
        for name in tool_names
    ]


def _make_agent_wrapper(handler: ExtendedTelemetryHandler) -> Callable:
    async def _agent_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        try:
            merged = normalize_kwargs(wrapped, args, kwargs)
        except Exception:
            merged = dict(kwargs)

        agent_name = instance.__class__.__name__ if instance is not None else "AgenticRetriever"
        agent_id = _resolve_agent_id(instance)
        tool_definitions = _resolve_tool_definitions(instance, merged)

        invocation = InvokeAgentInvocation(
            provider="cognee",
            agent_name=agent_name,
            agent_id=agent_id or None,
            data_source_id=agent_id or None,
            tool_definitions=tool_definitions,
        )
        # Capture input payload if enabled (the query string).
        query = merged.get("query")
        if query is not None:
            captured = maybe_capture(query)
            if captured:
                invocation.input_messages = [
                    {"role": "user", "content": captured}
                ]

        handler.start_invoke_agent(invocation)
        token = set_react_round(0)
        try:
            result = await wrapped(*args, **kwargs)
            captured_out = maybe_capture(result)
            if captured_out:
                invocation.output_messages = [
                    {"role": "assistant", "content": captured_out}
                ]
            handler.stop_invoke_agent(invocation)
            return result
        except Exception as e:
            handler.fail_invoke_agent(
                invocation, Error(message=str(e), type=type(e))
            )
            raise
        finally:
            reset_react_round(token)

    return _agent_wrapper


def install_agent_wrapper(handler: ExtendedTelemetryHandler) -> None:
    """Wrap ``AgenticRetriever._run_tool_loop`` with hasattr detection and fallback.

    The primary target ``_run_tool_loop`` is private; if it is not present (Cognee
    refactor renamed it), we fall back to the public ``get_retrieved_objects``
    so the AGENT span is still produced but without STEP granularity.
    """
    try:
        primary_module = __import__(
            _AGENT_MODULE, fromlist=[_AGENT_CLASS]
        )
        primary_cls = getattr(primary_module, _AGENT_CLASS, None)
    except Exception as e:
        logger.debug("Cognee AgenticRetriever module not importable: %s", e)
        primary_cls = None

    target_method = None
    if primary_cls is not None:
        if hasattr(primary_cls, _AGENT_PRIMARY_METHOD):
            target_method = _AGENT_PRIMARY_METHOD
        elif hasattr(primary_cls, _AGENT_FALLBACK_METHOD):
            target_method = _AGENT_FALLBACK_METHOD
            logger.debug(
                "AgenticRetriever.%s missing — falling back to %s",
                _AGENT_PRIMARY_METHOD,
                _AGENT_FALLBACK_METHOD,
            )

    if target_method is None:
        logger.debug(
            "AgenticRetriever has neither %s nor %s; AGENT span disabled",
            _AGENT_PRIMARY_METHOD,
            _AGENT_FALLBACK_METHOD,
        )
        return

    try:
        wrap_function_wrapper(
            _AGENT_MODULE,
            f"{_AGENT_CLASS}.{target_method}",
            _make_agent_wrapper(handler),
        )
    except Exception as e:
        logger.debug(
            "Failed to wrap AgenticRetriever.%s: %s", target_method, e
        )


def uninstall_agent_wrapper() -> None:
    from opentelemetry.instrumentation.utils import unwrap

    for method in (_AGENT_PRIMARY_METHOD, _AGENT_FALLBACK_METHOD):
        try:
            unwrap(_AGENT_MODULE, f"{_AGENT_CLASS}.{method}")
        except Exception as e:
            logger.debug(
                "Failed to unwrap AgenticRetriever.%s: %s", method, e
            )
