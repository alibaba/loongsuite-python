"""ENTRY wrapper for Cognee V1 API.

Wraps ``cognee.add`` / ``cognify`` / ``search`` / ``recall`` / ``remember`` with
``ExtendedTelemetryHandler.start_entry`` / ``stop_entry`` / ``fail_entry`` to
produce an ENTRY span named ``enter_ai_application_system``.

All five functions are async in Cognee v1.2.1.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.cognee.internal._util import (
    maybe_capture,
    normalize_kwargs,
)
from opentelemetry.instrumentation.cognee.semconv import COGNEE_SEARCH_QUERY
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


# module_path, attr_name, operation_name (used for input extraction)
_ENTRY_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("cognee.api.v1.add", "add", "add"),
    ("cognee.api.v1.cognify", "cognify", "cognify"),
    ("cognee.api.v1.search", "search", "search"),
    ("cognee.api.v1.recall", "recall", "recall"),
    ("cognee.api.v1.remember", "remember", "remember"),
)


def _extract_session_id(operation: str, kwargs: dict[str, Any]) -> str | None:
    sid = kwargs.get("session_id")
    if sid is None and operation == "remember":
        sids = kwargs.get("session_ids")
        if isinstance(sids, list) and sids:
            sid = sids[0]
    if sid is None:
        return None
    return str(sid)


def _extract_user_id(operation: str, kwargs: dict[str, Any]) -> str | None:
    user = kwargs.get("user")
    if user is None:
        return None
    try:
        if hasattr(user, "id") and user.id is not None:
            return str(user.id)
    except Exception:
        pass
    try:
        return str(user)
    except Exception:
        return None


def _build_input(operation: str, kwargs: dict[str, Any]) -> list:
    captured = maybe_capture(_select_input_payload(operation, kwargs))
    if captured is None:
        return []
    return [{"role": "user", "content": captured}]


def _select_input_payload(operation: str, kwargs: dict[str, Any]) -> Any:
    if operation in ("add", "remember"):
        return kwargs.get("data")
    if operation == "cognify":
        return kwargs.get("datasets")
    if operation in ("search", "recall"):
        return kwargs.get("query_text")
    return None


def _build_output(operation: str, result: Any) -> list:
    captured = maybe_capture(result)
    if captured is None:
        return []
    return [{"role": "assistant", "content": captured}]


def _make_entry_wrapper(
    handler: ExtendedTelemetryHandler,
    operation: str,
) -> Callable:
    async def _entry_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        try:
            merged = normalize_kwargs(wrapped, args, kwargs)
        except Exception:
            merged = dict(kwargs)
        session_id = _extract_session_id(operation, merged)
        user_id = _extract_user_id(operation, merged)
        input_messages = _build_input(operation, merged)

        invocation = EntryInvocation(
            session_id=session_id,
            user_id=user_id,
            input_messages=input_messages,
        )
        # Propagate Cognee's own search query attribute as baggage-like metadata
        # so downstream spans can be associated (best-effort).
        query_text = merged.get("query_text")
        if query_text:
            invocation.attributes[COGNEE_SEARCH_QUERY] = str(query_text)[:500]

        handler.start_entry(invocation)
        try:
            result = await wrapped(*args, **kwargs)
            invocation.output_messages = _build_output(operation, result)
            handler.stop_entry(invocation)
            return result
        except Exception as e:
            handler.fail_entry(invocation, Error(message=str(e), type=type(e)))
            raise

    return _entry_wrapper


def install_entry_wrappers(handler: ExtendedTelemetryHandler) -> None:
    for module, name, op in _ENTRY_TARGETS:
        try:
            wrap_function_wrapper(module, name, _make_entry_wrapper(handler, op))
        except Exception as e:
            logger.debug("Failed to wrap %s.%s: %s", module, name, e)


def uninstall_entry_wrappers() -> None:
    from opentelemetry.instrumentation.utils import unwrap

    for module, name, _ in _ENTRY_TARGETS:
        try:
            unwrap(module, name)
        except Exception as e:
            logger.debug("Failed to unwrap %s.%s: %s", module, name, e)
