"""Utility helpers for Cognee instrumentation."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Optional

from opentelemetry.instrumentation.cognee.config import (
    MAX_PAYLOAD_BYTES,
    is_capture_message_content_enabled,
)

logger = logging.getLogger(__name__)


def safe_str(value: Any, *, max_len: Optional[int] = MAX_PAYLOAD_BYTES) -> str:
    try:
        if value is None:
            return ""
        text = str(value)
    except Exception:
        return ""
    if max_len is not None and len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def _default_json_serializer(value: Any) -> str:
    def _fallback(o: Any) -> Any:
        if isinstance(o, (set, frozenset)):
            return list(o)
        if hasattr(o, "model_dump_json"):
            try:
                return o.model_dump_json()
            except Exception:
                pass
        if hasattr(o, "dict") and callable(o.dict):
            try:
                return o.dict()
            except Exception:
                pass
        if hasattr(o, "__dict__"):
            return {k: v for k, v in vars(o).items() if not k.startswith("_")}
        return repr(o)

    return json.dumps(value, default=_fallback, ensure_ascii=False)


def gen_ai_json_dumps(value: Any) -> str:
    try:
        text = _default_json_serializer(value)
    except Exception as e:
        logger.debug("gen_ai_json_dumps failed: %s", e)
        try:
            text = str(value)
        except Exception:
            return ""
    if len(text) > MAX_PAYLOAD_BYTES:
        text = text[:MAX_PAYLOAD_BYTES] + "..."
    return text


def maybe_capture(value: Any) -> Optional[str]:
    if not is_capture_message_content_enabled():
        return None
    return gen_ai_json_dumps(value)


def get_exception_type(exc: BaseException) -> str:
    return type(exc).__qualname__


def normalize_kwargs(func: Any, args: tuple, kwargs: dict) -> dict:
    """Merge positional args into kwargs using the function signature."""
    merged = dict(kwargs)
    if not args:
        return merged
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return merged
    params = list(sig.parameters.values())
    for value, param in zip(args, params):
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            break
        merged.setdefault(param.name, value)
    return merged
