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
ReAct STEP instrumentation patch.

``ReAct.forward`` runs its Reasoning-Acting loop inline and is not decorated
with ``@with_callbacks``, so the per-round boundary is invisible to DSPy's
callback system. The reasoning call of every round goes through
``ReAct._call_with_potential_trajectory_truncation`` (and its ``async``
counterpart), which is therefore the only observable round marker.

A STEP span opens when a round's reasoning starts and stays open — and
attached to the OpenTelemetry context — until the *next* round starts or the
final ``extract`` runs, so the tool call that the round decided on nests
inside the STEP instead of becoming a sibling of it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import dspy

from opentelemetry.instrumentation.dspy.internal.callback import (
    OTelDSPyCallback,
    _CallData,
)
from opentelemetry.instrumentation.dspy.internal.config import (
    react_step_enabled,
)

logger = logging.getLogger(__name__)

_TRUNCATION_METHODS = (
    "_call_with_potential_trajectory_truncation",
    "_async_call_with_potential_trajectory_truncation",
)


def _active_callback(instance: Any) -> OTelDSPyCallback | None:
    """Find the LoongSuite callback the same way DSPy resolves callbacks."""
    try:
        callbacks = list(dspy.settings.get("callbacks") or []) + list(
            getattr(instance, "callbacks", None) or []
        )
    except Exception:
        logger.debug("Failed to read DSPy callbacks", exc_info=True)
        return None
    for callback in callbacks:
        if isinstance(callback, OTelDSPyCallback):
            return callback
    return None


def _begin_round(
    instance: Any, module: Any
) -> tuple[OTelDSPyCallback, _CallData] | None:
    """Open (or close) a STEP span for this reasoning / extract call.

    ``_call_with_potential_trajectory_truncation`` serves both the per-round
    reasoning predictor and the final ``extract``; only the former is a ReAct
    round, so ``extract`` merely closes the last open STEP.
    """
    if not react_step_enabled():
        return None

    callback = _active_callback(instance)
    if callback is None:
        return None

    data = callback.current_agent_data()
    if data is None:
        return None

    if module is not getattr(instance, "react", None):
        callback.exit_react_step(data)
        return None

    callback.enter_react_step(data)
    return callback, data


def _record_reasoning(state: Any, result: Any) -> None:
    """Remember the tool the round selected, for the STEP finish reason.

    The STEP is not closed here even when the round chose ``finish``: DSPy
    still executes the ``finish`` tool before leaving the loop, and that tool
    call belongs to this round.
    """
    if state is None:
        return
    _, data = state
    tool_name = getattr(result, "next_tool_name", None)
    if isinstance(tool_name, str):
        data.last_tool_name = tool_name


def _fail_round(state: Any, exception: BaseException) -> None:
    if state is None:
        return
    callback, data = state
    callback.fail_react_step(data, exception)


def _called_module(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return args[0] if args else kwargs.get("module")


def _make_sync_wrapper(original_fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        state = None
        try:
            state = _begin_round(self, _called_module(args, kwargs))
        except Exception:
            logger.debug("Failed to start ReAct step span", exc_info=True)

        try:
            result = original_fn(self, *args, **kwargs)
        except BaseException as exception:
            try:
                _fail_round(state, exception)
            except Exception:
                logger.debug("Failed to fail ReAct step span", exc_info=True)
            raise

        try:
            _record_reasoning(state, result)
        except Exception:
            logger.debug("Failed to finalize ReAct step span", exc_info=True)
        return result

    return wrapper


def _make_async_wrapper(original_fn: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        state = None
        try:
            state = _begin_round(self, _called_module(args, kwargs))
        except Exception:
            logger.debug("Failed to start ReAct step span", exc_info=True)

        try:
            result = await original_fn(self, *args, **kwargs)
        except BaseException as exception:
            try:
                _fail_round(state, exception)
            except Exception:
                logger.debug("Failed to fail ReAct step span", exc_info=True)
            raise

        try:
            _record_reasoning(state, result)
        except Exception:
            logger.debug("Failed to finalize ReAct step span", exc_info=True)
        return result

    return wrapper


_originals: dict[str, Callable[..., Any]] = {}


def instrument_react_step() -> bool:
    """Patch the ReAct round marker. Returns True when the patch applied."""
    react_cls = getattr(dspy, "ReAct", None)
    if react_cls is None:
        logger.debug("dspy.ReAct not available, skipping ReAct STEP patch")
        return False

    for name in _TRUNCATION_METHODS:
        original = getattr(react_cls, name, None)
        if original is None:
            logger.debug("dspy.ReAct.%s not found, skipping that patch", name)
            continue
        _originals[name] = original
        wrapper = (
            _make_async_wrapper(original)
            if name.startswith("_async")
            else _make_sync_wrapper(original)
        )
        setattr(react_cls, name, wrapper)

    return bool(_originals)


def uninstrument_react_step() -> None:
    """Restore the original ReAct methods."""
    react_cls = getattr(dspy, "ReAct", None)
    if react_cls is None:
        _originals.clear()
        return

    for name, original in list(_originals.items()):
        try:
            setattr(react_cls, name, original)
        except Exception:
            logger.debug(
                "Failed to restore dspy.ReAct.%s", name, exc_info=True
            )
    _originals.clear()
