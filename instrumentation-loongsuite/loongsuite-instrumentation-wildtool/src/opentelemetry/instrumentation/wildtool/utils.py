"""Utility functions for WildToolBench instrumentation."""

import json
from typing import Any, Optional


def safe_json_dumps(obj: Any, max_length: int = 10000) -> Optional[str]:
    """Safely serialize object to JSON string with length limit."""
    if obj is None:
        return None
    try:
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) > max_length:
            return s[:max_length] + "...(truncated)"
        return s
    except (TypeError, ValueError):
        return str(obj)[:max_length]
