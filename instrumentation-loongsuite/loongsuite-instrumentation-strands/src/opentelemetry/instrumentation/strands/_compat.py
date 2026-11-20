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

"""Compat shims for Strands Agents behaviour that spans OTel contexts."""

import inspect
import logging
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# Private probe used to tell whether a context token can still be reset in the
# current context. A dedicated ContextVar keeps that check off the
# ``opentelemetry.context`` logging path.
_CONTEXT_PROBE: ContextVar = ContextVar("loongsuite_strands_context_probe")


class _TolerantContextApi:
    """Delegate to the OTel context API and keep one framework detach safe.

    ``strands.agent.agent`` keeps a module level ``context_api`` binding and,
    from Strands 1.60.0 on, wraps every invocation in an OpenTelemetry baggage
    attach/detach pair. ``context_api.attach`` returns a token that only the
    context that created it can reset, and ``opentelemetry.context.detach``
    logs ``Failed to detach context`` *before* it raises ``ValueError`` when the
    ambient context changed. An application that closes the stream from another
    context - ``asyncio.create_task(stream.aclose())`` copies the current
    context - resumes the framework's ``finally`` somewhere else, so tolerating
    the exception would still leave that error record behind.

    This proxy pairs every framework attach with a private context probe. When
    the probe proves the token can still be reset, the proxy resets it as
    before; when it cannot, the impossible reset is skipped, so nothing is
    logged and no exception escapes. Attach and detach of tokens the proxy did
    not hand out are delegated unchanged.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        # Context tokens are unhashable, so pair them by identity and keep the
        # token alive for as long as the pairing exists.
        self._probes: dict[int, tuple[Any, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def attach(self, context: Any) -> Any:
        probe_token = _CONTEXT_PROBE.set(object())
        token = self._delegate.attach(context)
        self._probes[id(token)] = (token, probe_token)
        return token

    def detach(self, token: Any) -> None:
        paired = self._probes.pop(id(token), None)
        if paired is None or paired[0] is not token:
            self._delegate.detach(token)
            return
        if _resets_in_current_context(paired[1]):
            self._delegate.detach(token)
            return
        logger.debug(
            "Skipped a Strands baggage detach: the token belongs to the "
            "context that started the stream",
        )


def _resets_in_current_context(probe_token: Any) -> bool:
    """Whether the ambient context is the one that created ``probe_token``."""
    try:
        _CONTEXT_PROBE.reset(probe_token)
    except ValueError:
        return False
    return True


class BaggageContextCompat:
    """Keep the framework owned baggage detach from failing on stream close.

    Strands 1.60.0 attaches an OpenTelemetry baggage context inside
    ``Agent.stream_async`` and detaches it in the async generator ``finally``.
    The baggage travels with the context that starts the stream, so closing a
    stream from another context (for example
    ``asyncio.create_task(stream.aclose())``) cannot restore it. This shim makes
    only that framework owned detach tolerant, keeps the instrumentation's own
    ContextVar hygiene intact, and leaves frameworks without the baggage attach
    (Strands < 1.60.0) untouched.
    """

    def __init__(self) -> None:
        self._module: Any = None
        self._original: Any = None
        self._proxy: Any = None

    def install(self, module: Any = None) -> None:
        """Patch ``module.context_api`` when the framework attaches baggage.

        ``module`` defaults to ``strands.agent.agent`` and is injectable so the
        shim can be tested against a stub module.
        """
        agent_module = module if module is not None else _agent_module()
        original = getattr(agent_module, "context_api", None)
        if original is None or not _attaches_invocation_baggage(agent_module):
            return
        proxy = _TolerantContextApi(original)
        agent_module.context_api = proxy
        self._module = agent_module
        self._original = original
        self._proxy = proxy

    def restore(self) -> None:
        if (
            self._module is not None
            and self._module.context_api is self._proxy
        ):
            self._module.context_api = self._original
        self._module = None
        self._original = None
        self._proxy = None


def _agent_module() -> Any:
    from strands.agent import agent as agent_module  # noqa: PLC0415

    return agent_module


def _attaches_invocation_baggage(agent_module: Any) -> bool:
    """Whether ``Agent`` accepts the 1.60.0 ``baggage_attributes`` argument."""
    agent = getattr(agent_module, "Agent", None)
    try:
        parameters = inspect.signature(agent.__init__).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "baggage_attributes" in parameters
