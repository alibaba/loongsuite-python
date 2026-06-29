"""
Utility helpers for MemPalace instrumentation: redaction, truncation,
parameter normalization, provider/model inference.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Mirror of mempalace.wal._WAL_REDACT_KEYS — used as a second-pass redaction
# whitelist when capture-message-content is enabled.
WAL_REDACT_KEYS: frozenset[str] = frozenset(
    {
        "content",
        "content_preview",
        "document",
        "entry",
        "entry_preview",
        "query",
        "text",
    }
)


def safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def truncate_bytes(value: str, max_bytes: int) -> str:
    if not isinstance(value, str) or max_bytes <= 0:
        return value
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + "…"


def redact_mapping(payload: Any) -> Any:
    """Replace values of WAL redact keys with length-only placeholders."""
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            if isinstance(k, str) and k in WAL_REDACT_KEYS:
                out[k] = f"<redacted:len={len(str(v))}>"
            else:
                out[k] = redact_mapping(v)
        return out
    if isinstance(payload, list):
        return [redact_mapping(v) for v in payload]
    return payload


def safe_json_dumps(value: Any, max_bytes: int, redact: bool = False) -> Optional[str]:
    try:
        if redact:
            value = redact_mapping(value)
        text = json.dumps(value, ensure_ascii=False, default=str)
        return truncate_bytes(text, max_bytes)
    except Exception as e:
        logger.debug("Failed to serialize value: %s", e)
        return None


def sha8(value: str) -> Optional[str]:
    try:
        if not value:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return None


def palace_path_hash(palace_path: Optional[str]) -> Optional[str]:
    if not palace_path:
        return None
    return sha8(palace_path)


def normalize_call_parameters(func: Callable, args: tuple, kwargs: dict) -> dict:
    """Merge positional + keyword args into a kwargs dict via inspect.signature."""
    normalized = dict(kwargs)
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        start_index = 0
        if params and params[0].name in ("self", "cls"):
            start_index = 1
        for idx, arg_value in enumerate(args):
            param_idx = start_index + idx
            if param_idx >= len(params):
                break
            param = params[param_idx]
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param.name not in normalized:
                normalized[param.name] = arg_value
    except Exception as e:
        logger.debug("Failed to normalize call parameters: %s", e)
    return normalized


def get_exception_type(exception: Exception) -> str:
    return type(exception).__name__


def extract_filter_keys(where: Any) -> list[str]:
    if isinstance(where, dict) and where:
        return list(where.keys())
    return []


def extract_filter_operators(where: Any) -> list[str]:
    ops: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.startswith("$"):
                    ops.append(k)
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(where)
    return ops


def distance_to_similarity(distance: Any, metric: str = "cosine") -> Optional[float]:
    d = safe_float(distance)
    if d is None:
        return None
    if metric == "cosine":
        try:
            return 1.0 - float(d)
        except Exception:
            return None
    return None


def infer_provider_from_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    try:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        host = parsed.hostname or endpoint
        host = host.lower()
        for name in (
            "openai.com",
            "anthropic.com",
            "dashscope.aliyuncs.com",
            "ollama",
            "vllm",
            "deepseek",
            "moonshot",
            "qwen",
        ):
            if name in host:
                return name.split(".")[0]
        return host.split(".")[0] or None
    except Exception:
        return None


def monotonic_now() -> float:
    return time.monotonic()
