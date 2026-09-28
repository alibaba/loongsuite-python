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

"""OpenTelemetry instrumentation for agentUniverse (LoongSuite).

This package instruments the agentUniverse Agent, LLM and Tool layers by
claiming the six module-level wrapper extension points of
``agentuniverse.base.annotation.trace``. The ``@trace_agent``, ``@trace_llm``
and ``@trace_tool`` decorators resolve those globals on every call, so replacing
them intercepts every call without touching the decorated functions.

It is an independent implementation: it never imports, calls, patches or
delegates to agentUniverse's own OTel instrumentors (``AgentInstrumentor``,
``LLMInstrumentor``, ``ToolInstrumentor``, their attribute setters, span and
metric managers or queue wrappers). Everything an instrumented call needs --
spans, the ``au.*`` compatibility attributes, metrics, token aggregation,
streaming finalization and error recording -- is produced here, together with
the LoongSuite ``gen_ai.*`` semantic conventions written onto the same span by
the shared ``ExtendedTelemetryHandler``.

Instrumentation is transactional: the six extension points are snapshotted
first, replaced only after every layer is ready, rolled back as a whole if any
step fails, and restored to their original objects on uninstrument. If
agentUniverse's own instrumentors were active before this package was
instrumented, their wrappers are suppressed (never called) while LoongSuite owns
the extension points, and are restored verbatim afterwards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from agentuniverse.base.annotation import trace as trace_module

from opentelemetry import metrics, trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from ._agent import AgentLayer
from ._common import (
    INSTRUMENTOR_NAME,
    LAYER_WRAPPER_GLOBALS,
    WRAPPER_GLOBAL_NAMES,
    describe_wrapper_owner,
)
from ._llm import LLMLayer
from ._tool import ToolLayer
from .version import __version__

logger = logging.getLogger(__name__)

__all__ = ["AgentUniverseInstrumentor"]


class _InstallState:
    """Which instrumentor instance currently owns the extension points."""

    owner: Optional["AgentUniverseInstrumentor"] = None


_install_state = _InstallState()


class AgentUniverseInstrumentor(BaseInstrumentor):
    """LoongSuite instrumentation for the agentUniverse Agent, LLM and Tool."""

    _snapshot: Dict[str, Any] = {}
    _wrappers: Dict[str, Any] = {}
    _owns_install = False
    #: Set when ``_instrument`` rolled back, so ``instrument`` can undo the
    #: instrumented mark ``BaseInstrumentor`` applies after it returns.
    _install_failed = False

    def instrumentation_dependencies(self) -> Sequence[str]:
        return ("agentUniverse >= 0.0.19",)

    # -- install ------------------------------------------------------

    def instrument(self, **kwargs: Any) -> None:
        """Instrument the framework, keeping a rolled-back install retryable.

        ``BaseInstrumentor.instrument`` marks the instrumentor as instrumented
        as soon as ``_instrument`` returns, including when it returned because
        it had rolled the install back again. Clearing that mark keeps the
        bookkeeping honest: nothing of ours is installed, and a later attempt
        can run.
        """
        super().instrument(**kwargs)
        if self._install_failed:
            self._install_failed = False
            self._is_instrumented_by_opentelemetry = False

    def _instrument(self, **kwargs: Any) -> None:
        owner = self._installed_owner()
        if owner is not None:
            logger.warning(
                "The agentUniverse wrapper extension points are already owned "
                "by %s; skipping this install so calls are not wrapped twice",
                owner,
            )
            self._install_failed = True
            return

        snapshot = {
            name: getattr(trace_module, name, None)
            for name in WRAPPER_GLOBAL_NAMES
        }
        self._log_suppressed_wrappers(snapshot)

        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")

        try:
            tracer = trace.get_tracer(
                INSTRUMENTOR_NAME, __version__, tracer_provider
            )
            meter = metrics.get_meter(
                INSTRUMENTOR_NAME, __version__, meter_provider
            )
            handler = ExtendedTelemetryHandler(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )
            agent_layer = AgentLayer(tracer, handler, meter)
            llm_layer = LLMLayer(tracer, handler, meter)
            tool_layer = ToolLayer(tracer, handler, meter)
            wrappers: Dict[str, Any] = {
                "_agent_wrapper_sync": agent_layer.wrap_sync,
                "_agent_wrapper_async": agent_layer.wrap_async,
                "_llm_wrapper_sync": llm_layer.wrap_sync,
                "_llm_wrapper_async": llm_layer.wrap_async,
                "_tool_wrapper_sync": tool_layer.wrap_sync,
                "_tool_wrapper_async": tool_layer.wrap_async,
            }
        except Exception:
            logger.warning(
                "Could not build the LoongSuite agentUniverse instrumentation; "
                "the application keeps using the original wrappers",
                exc_info=True,
            )
            self._install_failed = True
            return

        installed: list[str] = []
        try:
            for name, wrapper in wrappers.items():
                self._install_wrapper(name, wrapper)
                installed.append(name)
        except Exception:
            self._rollback(snapshot, wrappers, installed)
            logger.warning(
                "Installing the LoongSuite agentUniverse wrappers failed; "
                "rolled back to the original wrappers",
                exc_info=True,
            )
            self._install_failed = True
            return

        self._snapshot = snapshot
        self._wrappers = wrappers
        self._owns_install = True
        _install_state.owner = self

    def _install_wrapper(self, name: str, wrapper: Any) -> None:
        """Replace one extension point (a seam, so install can be tested)."""
        setattr(trace_module, name, wrapper)

    def _rollback(
        self,
        snapshot: Dict[str, Any],
        wrappers: Dict[str, Any],
        installed: list[str],
    ) -> None:
        """Put the snapshotted wrappers back after a failed install.

        Only extension points this install still holds are restored; anything
        another instrumentation has taken over in the meantime is left alone.
        """
        for name in installed:
            if getattr(trace_module, name, None) is not wrappers.get(name):
                continue
            try:
                setattr(trace_module, name, snapshot[name])
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "Could not restore the agentUniverse wrapper %s",
                    name,
                    exc_info=True,
                )

    def _log_suppressed_wrappers(self, snapshot: Dict[str, Any]) -> None:
        """Report third-party wrappers that this install takes over."""
        for name in WRAPPER_GLOBAL_NAMES:
            owner = describe_wrapper_owner(snapshot.get(name))
            if owner is None:
                continue
            logger.info(
                "Suppressing the pre-existing %s wrapper for %s; it will be "
                "restored when this instrumentation is removed",
                owner,
                name,
            )

    @staticmethod
    def _installed_owner() -> Optional["AgentUniverseInstrumentor"]:
        """The instance that currently owns the extension points, if any.

        The sentinel can go stale: an application (or a test harness) may put the
        wrappers back without going through this class. It therefore only counts
        while the extension points still hold that instance's wrappers.
        """
        owner = _install_state.owner
        if owner is None:
            return None
        if getattr(
            trace_module, "_agent_wrapper_sync", None
        ) is owner._wrappers.get("_agent_wrapper_sync"):
            return owner
        _install_state.owner = None
        owner._owns_install = False
        return None

    # -- uninstall ----------------------------------------------------

    def _uninstrument(self, **kwargs: Any) -> None:
        if not self._owns_install:
            logger.debug(
                "Nothing to remove: this instance does not own the "
                "agentUniverse wrapper extension points"
            )
            return

        failures = []
        replaced = []
        for name, original in self._snapshot.items():
            if getattr(trace_module, name, None) is not self._wrappers.get(
                name
            ):
                replaced.append(name)
                continue
            try:
                setattr(trace_module, name, original)
            except Exception:  # pragma: no cover - defensive
                failures.append(name)
        if replaced:
            logger.warning(
                "Left %s untouched: another instrumentation replaced the "
                "LoongSuite wrappers after they were installed",
                ", ".join(replaced),
            )
        if failures:
            logger.warning(
                "Could not restore the agentUniverse wrapper extension "
                "points: %s",
                ", ".join(failures),
            )

        if _install_state.owner is self:
            _install_state.owner = None
        self._owns_install = False
        self._snapshot = {}
        self._wrappers = {}

    # -- introspection (used by tests and diagnostics) -----------------

    @property
    def wrapper_snapshot(self) -> Dict[str, Any]:
        """The wrapper objects captured before this install."""
        return dict(self._snapshot)

    @property
    def installed_wrappers(self) -> Dict[str, Any]:
        """The wrapper objects this install put in place."""
        return dict(self._wrappers)

    @property
    def layer_globals(self) -> tuple[tuple[str, str], ...]:
        return LAYER_WRAPPER_GLOBALS
