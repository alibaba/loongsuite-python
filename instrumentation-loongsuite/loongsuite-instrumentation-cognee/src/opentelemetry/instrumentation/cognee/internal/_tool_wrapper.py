"""TOOL wrapper for Cognee ``cognee.modules.tools.execute_tool.execute_tool``.

``execute_tool(user, dataset_id, tool_name, args=None, allowed_tools=None)``
is async and the single entry point for executing any tool call dispatched
by the AgenticRetriever ReAct loop. We wrap it to produce a TOOL span.
"""

from __future__ import annotations

import logging
from typing import Callable
from uuid import uuid4

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.cognee.internal._util import (
    maybe_capture,
    normalize_kwargs,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


_TOOL_MODULE = "cognee.modules.tools.execute_tool"
_TOOL_FUNC = "execute_tool"


def _make_tool_wrapper(handler: ExtendedTelemetryHandler) -> Callable:
    async def _tool_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        try:
            merged = normalize_kwargs(wrapped, args, kwargs)
        except Exception:
            merged = dict(kwargs)

        tool_name = merged.get("tool_name") or (args[2] if len(args) > 2 else None)
        tool_args = merged.get("args") or (args[3] if len(args) > 3 else None) or {}

        invocation = ExecuteToolInvocation(
            tool_name=str(tool_name) if tool_name is not None else "unknown",
            tool_call_id=uuid4().hex,
            tool_type="function",
        )
        captured_args = maybe_capture(tool_args)
        if captured_args is not None:
            invocation.tool_call_arguments = captured_args

        handler.start_execute_tool(invocation)
        try:
            result = await wrapped(*args, **kwargs)
            captured_result = maybe_capture(result)
            if captured_result is not None:
                invocation.tool_call_result = captured_result
            handler.stop_execute_tool(invocation)
            return result
        except Exception as e:
            handler.fail_execute_tool(
                invocation, Error(message=str(e), type=type(e))
            )
            raise

    return _tool_wrapper


def install_tool_wrapper(handler: ExtendedTelemetryHandler) -> None:
    try:
        wrap_function_wrapper(
            _TOOL_MODULE, _TOOL_FUNC, _make_tool_wrapper(handler)
        )
    except Exception as e:
        logger.debug("Failed to wrap %s.%s: %s", _TOOL_MODULE, _TOOL_FUNC, e)


def uninstall_tool_wrapper() -> None:
    from opentelemetry.instrumentation.utils import unwrap

    try:
        unwrap(_TOOL_MODULE, _TOOL_FUNC)
    except Exception as e:
        logger.debug("Failed to unwrap %s.%s: %s", _TOOL_MODULE, _TOOL_FUNC, e)
