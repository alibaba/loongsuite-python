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

"""The ReAct STEP layer of the LoongSuite agentUniverse instrumentation.

A tool-calling agentUniverse agent is a Reasoning-Acting loop: every round calls
the LLM, and a round whose answer carries tool calls is followed by the tools.
LoongSuite models one such iteration as its own ``STEP`` span
(``gen_ai.span.kind=STEP``, ``gen_ai.operation.name=react``,
``gen_ai.react.round`` and ``gen_ai.react.finish_reason``), which is what this
module creates.

The interception point is the framework's own round hooks,
``Agent.on_before_llm_round``/``Agent.on_after_llm_round``: the tool-calling loop
calls them around every round, the async loop delegates to them, and no
framework class overrides them. They are patched on the class for the duration of
an install and restored verbatim on uninstrument -- the same transactional rule
the layer wrappers follow.

Each step span is a child of the agent span, and its token usage bucket is
promoted to the agent when the step closes, so a round's LLM usage still reaches
the agent exactly once.

No agentUniverse OTel class is imported or called here.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Any, Dict, Optional

from agentuniverse.agent.agent import Agent

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import ReactStepInvocation
from opentelemetry.util.genai.types import Error

from ._common import (
    AgentAttributes,
    agent_name_attributes,
    detach_context_token,
    error_status_text,
    promote_token_usage_to_parent,
    reset_token_usage,
    run_safely,
    token_usage_for,
)

logger = logging.getLogger(__name__)

#: ``au.span.kind`` carrier value for a ReAct step, like the other layers.
STEP_SPAN_KIND_VALUE = "step"

#: Used when the round answered without naming a reason of its own.
DEFAULT_FINISH_REASON = "stop"

#: The reason a round that answered with tool calls gives, per the framework's
#: own ``finish_reason`` vocabulary.
TOOL_CALL_FINISH_REASON = "tool_calls"


class ReactStep:
    """One Reasoning-Acting iteration, as a LoongSuite STEP span."""

    def __init__(
        self, layer: "ReactStepLayer", round_index: Optional[int]
    ) -> None:
        self._layer = layer
        number: Optional[int] = None
        if isinstance(round_index, int) and round_index >= 0:
            # The framework counts rounds from zero, the convention from one.
            number = round_index + 1
        self.invocation = ReactStepInvocation(
            round=number, attributes=agent_name_attributes()
        )
        self.span = None
        self.finished = False

    @property
    def started(self) -> bool:
        return self.span is not None

    def start(self) -> None:
        """Start the step span and attach it as the round's parent context."""
        run_safely(self._layer.handler.start_react_step, self.invocation)
        self.span = self.invocation.span
        if self.span is None:
            # ``start_react_step`` attaches the context token and the span
            # together; a token without a span would stay attached to this
            # context for the rest of the process, so it is released here.
            token, self.invocation.context_token = (
                self.invocation.context_token,
                None,
            )
            run_safely(detach_context_token, token)
            return
        run_safely(
            self.span.set_attribute,
            AgentAttributes.SPAN_KIND,
            STEP_SPAN_KIND_VALUE,
        )
        # The framework's own span bookkeeping seeds a fresh usage bucket for
        # every span it starts; the shared handler does not, so the step gets its
        # own bucket before its children report into it.
        run_safely(reset_token_usage, self.span.context.span_id)

    def finish(
        self,
        error: Optional[BaseException] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        """Close the step span exactly once, whatever ends the round."""
        if self.finished:
            return
        self.finished = True
        span = self.span
        if span is None:
            return

        if finish_reason and not self.invocation.finish_reason:
            self.invocation.finish_reason = finish_reason

        # What the round's children accumulated belongs to the agent span: the
        # step is a span of its own, not a second place usage is counted.
        parent = span.parent
        run_safely(
            promote_token_usage_to_parent,
            token_usage_for(span.context.span_id),
            parent.span_id if parent is not None else None,
        )

        if error is None:
            run_safely(self._layer.handler.stop_react_step, self.invocation)
        else:
            run_safely(
                self._layer.handler.fail_react_step,
                self.invocation,
                Error(message=error_status_text(error), type=type(error)),
            )
        # Both handlers detach the context token and end the span. Ending is
        # checked first because a span that is no longer recording is exactly the
        # one they already closed; the token is theirs to release, and releasing
        # it twice raises out of ``contextvars``.
        if span.is_recording():
            run_safely(span.end)


class ReactStepTracker:
    """Opens one STEP span per round of the agent call it belongs to."""

    def __init__(self, layer: "ReactStepLayer") -> None:
        self._layer = layer
        self._open: Optional[ReactStep] = None

    def begin(self, round_index: Optional[int]) -> None:
        if self._open is not None:
            return
        step = ReactStep(self._layer, round_index)
        step.start()
        if step.started:
            self._open = step

    def end(
        self,
        llm_output: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        step, self._open = self._open, None
        if step is None:
            return
        step.finish(error, self._finish_reason(llm_output))

    @staticmethod
    def _finish_reason(llm_output: Any) -> Optional[str]:
        if llm_output is None:
            return None
        reason = getattr(llm_output, "finish_reason", None)
        if isinstance(reason, str) and reason:
            return reason
        has_tool_calls = getattr(llm_output, "has_tool_calls", None)
        if callable(has_tool_calls) and run_safely(has_tool_calls):
            return TOOL_CALL_FINISH_REASON
        return DEFAULT_FINISH_REASON

    def close(self, error: Optional[BaseException] = None) -> None:
        """Close a step the loop left open, as an aborted round does."""
        self.end(None, error)


class ReactStepLayer:
    """The step spans an agent call's rounds are recorded as."""

    def __init__(self, handler: ExtendedTelemetryHandler) -> None:
        self.handler = handler

    def tracker(self) -> ReactStepTracker:
        return ReactStepTracker(self)


_current_tracker: ContextVar[Optional[ReactStepTracker]] = ContextVar(
    "agentuniverse_react_step_tracker", default=None
)


def current_react_step_tracker() -> Optional[ReactStepTracker]:
    """The tracker of the agent call that is running in this context."""
    return _current_tracker.get()


def push_react_step_tracker(
    tracker: Optional[ReactStepTracker],
) -> Optional[Token]:
    if tracker is None:
        return None
    try:
        return _current_tracker.set(tracker)
    except Exception:  # pragma: no cover - defensive
        return None


def pop_react_step_tracker(token: Optional[Token]) -> None:
    if token is None:
        return
    try:
        _current_tracker.reset(token)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Failed to reset the ReAct step tracker", exc_info=True)


#: The class attributes this package claims for the step spans.
REACT_STEP_HOOKS: tuple[str, ...] = (
    "on_before_llm_round",
    "on_after_llm_round",
)


def install_react_step_hooks(layer: ReactStepLayer) -> Dict[str, Any]:
    """Claim the framework's round hooks; return the originals it replaced.

    The hooks stay no-ops for the application: whatever the framework (or
    anything that claimed them before) does around a round still happens, in the
    same order. Only the step span is added.
    """
    originals: Dict[str, Any] = {
        name: getattr(Agent, name) for name in REACT_STEP_HOOKS
    }

    def before(
        self: Any,
        context: Any,
        round_index: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        tracker = _current_tracker.get()
        if tracker is not None:
            run_safely(tracker.begin, round_index)
        return originals["on_before_llm_round"](
            self, context, round_index=round_index, *args, **kwargs
        )

    def after(
        self: Any,
        context: Any,
        llm_output: Any = None,
        round_index: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            return originals["on_after_llm_round"](
                self,
                context,
                llm_output,
                round_index=round_index,
                *args,
                **kwargs,
            )
        finally:
            tracker = _current_tracker.get()
            if tracker is not None:
                run_safely(tracker.end, llm_output)

    try:
        Agent.on_before_llm_round = before
        Agent.on_after_llm_round = after
    except Exception:
        logger.warning(
            "Could not claim the agentUniverse round hooks; no ReAct STEP "
            "spans will be recorded",
            exc_info=True,
        )
        remove_react_step_hooks(originals)
        return {}
    return originals


def remove_react_step_hooks(originals: Dict[str, Any]) -> None:
    """Restore the round hooks this package replaced."""
    for name in REACT_STEP_HOOKS:
        original = originals.get(name)
        if original is None:
            continue
        try:
            setattr(Agent, name, original)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "Could not restore the agentUniverse round hook %s",
                name,
                exc_info=True,
            )
