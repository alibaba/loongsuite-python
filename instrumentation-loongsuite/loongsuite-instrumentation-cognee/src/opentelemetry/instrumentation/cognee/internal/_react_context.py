"""ContextVar for tracking Cognee ReAct round across coroutines.

ContextVar copies are propagated automatically into child coroutines created by
asyncio.create_task / asyncio.gather, which is what makes this approach safe for
``AgenticRetriever._run_tool_loop`` (async for-loop). Each AGENT invocation sets
a fresh value; STEP wrapper increments the value while inside that context only.
"""

from __future__ import annotations

import contextvars
from typing import Optional

_COGNEE_REACT_ROUND: "contextvars.ContextVar[Optional[int]]" = contextvars.ContextVar(
    "cognee_react_round", default=None
)


def get_react_round() -> Optional[int]:
    return _COGNEE_REACT_ROUND.get()


def set_react_round(value: Optional[int]):
    return _COGNEE_REACT_ROUND.set(value)


def reset_react_round(token) -> None:
    try:
        _COGNEE_REACT_ROUND.reset(token)
    except (LookupError, ValueError):
        # token mismatch can occur if a sibling context already reset; ignore.
        pass
