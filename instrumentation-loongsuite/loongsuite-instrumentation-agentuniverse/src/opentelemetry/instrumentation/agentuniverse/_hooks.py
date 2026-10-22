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

"""Reversible hooks at provider conversion and native tool dispatch boundaries."""

import inspect
from contextvars import ContextVar
from functools import wraps

from agentuniverse.agent.agent import Agent
from agentuniverse.llm.llm_output import TokenUsage

from ._common import (
    pop_current_agent_name,
    push_current_agent_name,
    run_safely,
)
from ._usage import capture_provider_usage

_tool_call_id: ContextVar[str | None] = ContextVar(
    "agentuniverse_tool_call_id", default=None
)


def current_tool_call_id():
    return _tool_call_id.get()


def _usage_hook(original, provider):
    @wraps(original.__func__)
    def converted(cls, payload, *args, **kwargs):
        # The SDK conversion stays outside telemetry error handling and its
        # exact result object is returned, even when our collector fails.
        result = original.__get__(None, cls)(payload, *args, **kwargs)
        run_safely(capture_provider_usage, payload, provider)
        return result

    return classmethod(converted)


def _dispatch_hook(original, asynchronous):
    if asynchronous:

        @wraps(original)
        async def dispatch(self, tool_call, *args, **kwargs):
            identifier = run_safely(getattr, tool_call, "id", None)
            token = _tool_call_id.set(
                identifier if isinstance(identifier, str) else None
            )
            name, session = run_safely(
                _dispatch_metadata, self, args, kwargs
            ) or (None, None)
            agent_token = push_current_agent_name(name, session)
            try:
                return await original(self, tool_call, *args, **kwargs)
            finally:
                pop_current_agent_name(agent_token)
                _tool_call_id.reset(token)
    else:

        @wraps(original)
        def dispatch(self, tool_call, *args, **kwargs):
            identifier = run_safely(getattr, tool_call, "id", None)
            token = _tool_call_id.set(
                identifier if isinstance(identifier, str) else None
            )
            name, session = run_safely(
                _dispatch_metadata, self, args, kwargs
            ) or (None, None)
            agent_token = push_current_agent_name(name, session)
            try:
                return original(self, tool_call, *args, **kwargs)
            finally:
                pop_current_agent_name(agent_token)
                _tool_call_id.reset(token)

    return dispatch


def _dispatch_metadata(agent, args, kwargs):
    # The framework's thread pool propagates OTel context, rather than every
    # ContextVar. Its explicit AgentContext is authoritative in each worker.
    current = kwargs.get("agent_context") or (args[0] if args else None)
    session = getattr(current, "session_id", None)
    info = getattr(getattr(agent, "agent_model", None), "info", None)
    name = info.get("name") if isinstance(info, dict) else None
    return (
        name if isinstance(name, str) else None,
        session if isinstance(session, str) else None,
    )


def restore_auxiliary_hooks(hooks):
    for cls, name, original, wrapper in reversed(hooks):
        if inspect.getattr_static(cls, name, None) is wrapper:
            run_safely(setattr, cls, name, original)


def install_auxiliary_hooks():
    hooks = []
    try:
        for name, provider in (
            ("from_openai", "openai"),
            ("from_anthropic", "anthropic"),
        ):
            original = inspect.getattr_static(TokenUsage, name, None)
            if original is None:
                continue
            if not isinstance(original, classmethod):
                raise TypeError("Unsupported TokenUsage converter")
            wrapper = _usage_hook(original, provider)
            hooks.append((TokenUsage, name, original, wrapper))
            setattr(TokenUsage, name, wrapper)
        for name, asynchronous in (
            ("execute_tool_call", False),
            ("async_execute_tool_call", True),
        ):
            original = inspect.getattr_static(Agent, name, None)
            if original is None:
                continue
            wrapper = _dispatch_hook(original, asynchronous)
            hooks.append((Agent, name, original, wrapper))
            setattr(Agent, name, wrapper)
    except BaseException:
        restore_auxiliary_hooks(hooks)
        raise
    return hooks
