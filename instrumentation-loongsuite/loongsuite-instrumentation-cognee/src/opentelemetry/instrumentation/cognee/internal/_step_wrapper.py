"""STEP wrapper for Cognee ``generate_completion``.

``cognee.modules.retrieval.utils.completion.generate_completion`` is called
both inside the AgenticRetriever ReAct loop (with
``user_prompt_path="agentic_user.txt"``) and outside it (graph completion,
summarization, forced final answer). We only create a STEP span when the
caller is inside an AGENT context (detected via the ContextVar set by the
AGENT wrapper) AND the ``user_prompt_path`` matches the agentic prompt.

Round counting semantics:
    The AGENT wrapper sets the ContextVar to 0 on entry and resets it to its
    prior value on exit. Inside the AGENT context, each STEP wrapper call
    increments the value by 1 and writes it back WITHOUT resetting — this is
    how sequential ReAct iterations see 1, 2, 3 (rather than always 1). The
    AGENT wrapper's outer reset wipes all STEP increments at exit, so the
    value does not leak past the agent invocation.

    Because ``contextvars.ContextVar`` writes are local to the current
    ``asyncio.Task`` context, two agent loops running concurrently under
    ``asyncio.gather`` each see their own independent counter — concurrent
    rounds do not cross-talk. This is verified by the STEP wrapper unit test.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.cognee.config import is_react_step_enabled
from opentelemetry.instrumentation.cognee.internal._react_context import (
    get_react_round,
    set_react_round,
)
from opentelemetry.instrumentation.cognee.internal._util import (
    normalize_kwargs,
)
from opentelemetry.instrumentation.cognee.semconv import (
    AGENTIC_USER_PROMPT_FILENAME,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import ReactStepInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


_STEP_MODULE = "cognee.modules.retrieval.utils.completion"
_STEP_FUNC = "generate_completion"


def _is_agentic_prompt(user_prompt_path: Any) -> bool:
    if not user_prompt_path or not isinstance(user_prompt_path, str):
        return False
    if user_prompt_path == AGENTIC_USER_PROMPT_FILENAME:
        return True
    return os.path.basename(user_prompt_path) == AGENTIC_USER_PROMPT_FILENAME


def _infer_finish_reason(result: Any) -> str:
    if result is None:
        return "unknown"
    final_answer = getattr(result, "final_answer", None)
    tool_call = getattr(result, "tool_call", None)
    if final_answer and isinstance(final_answer, str) and final_answer.strip():
        return "stop"
    if tool_call is not None:
        return "tool_use"
    # Forced final answer path: AgenticRetriever calls generate_completion with
    # response_model=str when budget is exhausted, returning a plain string.
    if isinstance(result, str) and result.strip():
        return "max_iterations"
    return "unknown"


def _make_step_wrapper(handler: ExtendedTelemetryHandler) -> Callable:
    async def _step_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        current_round = get_react_round()
        if current_round is None:
            # Outside AGENT context — pass through without creating a STEP span.
            return await wrapped(*args, **kwargs)

        if not is_react_step_enabled():
            return await wrapped(*args, **kwargs)

        try:
            merged = normalize_kwargs(wrapped, args, kwargs)
        except Exception:
            merged = dict(kwargs)

        user_prompt_path = merged.get("user_prompt_path")
        # If the caller explicitly passed a non-agentic prompt path, do not
        # create a STEP span (e.g. forced final answer with user_prompt_path=
        # the parent retriever's user_prompt_path). When the prompt path is
        # missing we treat it as agentic to avoid dropping STEP spans.
        if user_prompt_path is not None and not _is_agentic_prompt(user_prompt_path):
            return await wrapped(*args, **kwargs)

        new_round = current_round + 1
        # NOTE: we do NOT capture a token for reset here. The increment is
        # meant to persist across sequential STEP calls inside the same
        # AGENT context so iterations see 1, 2, 3, …. The AGENT wrapper's
        # outer reset(token) wipes these increments when the agent exits.
        set_react_round(new_round)

        invocation = ReactStepInvocation(round=new_round)
        handler.start_react_step(invocation)
        try:
            result = await wrapped(*args, **kwargs)
            invocation.finish_reason = _infer_finish_reason(result)
            handler.stop_react_step(invocation)
            return result
        except Exception as e:
            invocation.finish_reason = "error"
            handler.fail_react_step(invocation, Error(message=str(e), type=type(e)))
            raise

    return _step_wrapper


def install_step_wrapper(handler: ExtendedTelemetryHandler) -> None:
    try:
        wrap_function_wrapper(
            _STEP_MODULE, _STEP_FUNC, _make_step_wrapper(handler)
        )
    except Exception as e:
        logger.debug("Failed to wrap %s.%s: %s", _STEP_MODULE, _STEP_FUNC, e)


def uninstall_step_wrapper() -> None:
    from opentelemetry.instrumentation.utils import unwrap

    try:
        unwrap(_STEP_MODULE, _STEP_FUNC)
    except Exception as e:
        logger.debug("Failed to unwrap %s.%s: %s", _STEP_MODULE, _STEP_FUNC, e)
