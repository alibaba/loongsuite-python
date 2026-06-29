"""
Public hook types for MemPalace instrumentation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

HookContext = dict[str, Any]

MemoryBeforeHook = Optional[Callable[..., Any]]
MemoryAfterHook = Optional[Callable[..., Any]]
InnerBeforeHook = Optional[Callable[..., Any]]
InnerAfterHook = Optional[Callable[..., Any]]


def safe_call_hook(hook: Optional[Callable[..., Any]], *args: Any) -> None:
    if not callable(hook):
        return
    try:
        hook(*args)
    except Exception as e:
        logger.debug("mempalace hook raised and was swallowed: %s", e)
