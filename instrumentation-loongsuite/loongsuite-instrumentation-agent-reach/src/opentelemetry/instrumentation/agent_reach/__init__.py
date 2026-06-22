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

"""OpenTelemetry instrumentation for the Agent-Reach framework.

Validated against ``agent-reach >= 1.5.0``.

Coverage (per ``execute.md``):

* ENTRY span — ``agent_reach.cli.main``
* TOOL span — ``agent_reach.doctor.check_all``
* TOOL span — every ``Channel.check`` subclass discovered dynamically via
  ``register_post_import_hook("agent_reach.channels")``
* TOOL span — ``agent_reach.probe.probe_command``
* TOOL span — ``agent_reach.transcribe.transcribe``
* TOOL span — ``agent_reach.transcribe.transcribe_chunk``
* TOOL span — ``agent_reach.integrations.mcp_server.create_server`` (only
  when ``loongsuite-instrumentation-mcp`` is NOT active; detected via
  ``BaseInstrumentor._is_instrumented_by_opentelemetry``)

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.agent_reach import AgentReachInstrumentor

    AgentReachInstrumentor().instrument()
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Collection

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.instrumentation.agent_reach._wrappers import (
    ChannelCheckWrapper,
    DoctorWrapper,
    EntryWrapper,
    MCPServerWrapper,
    ProbeWrapper,
    TranscribeChunkWrapper,
    TranscribeWrapper,
)
from opentelemetry.instrumentation.agent_reach.metrics import AgentReachMetrics
from opentelemetry.instrumentation.agent_reach.package import _instruments
from opentelemetry.instrumentation.agent_reach.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

__all__ = ["AgentReachInstrumentor", "__version__"]

_CLI_MODULE = "agent_reach.cli"
_DOCTOR_MODULE = "agent_reach.doctor"
_CHANNELS_MODULE = "agent_reach.channels"
_PROBE_MODULE = "agent_reach.probe"
_TRANSCRIBE_MODULE = "agent_reach.transcribe"
_MCP_SERVER_MODULE = "agent_reach.integrations.mcp_server"


def _is_instrumentor_active(instrumentor_class_name: str) -> bool:
    """Detect whether a given BaseInstrumentor subclass is active.

    Looks up ``_is_instrumented_by_opentelemetry`` (set by
    ``BaseInstrumentor.instrument()``) on the given class across a list of
    known module paths. Returns ``False`` if the class is missing or not
    yet instrumented.
    """
    candidate_modules: tuple[str, ...]
    if instrumentor_class_name == "MCPInstrumentor":
        candidate_modules = (
            "opentelemetry.instrumentation.mcp",
            "loongsuite_instrumentation_mcp",
        )
    elif instrumentor_class_name == "OpenAIInstrumentor":
        candidate_modules = (
            "opentelemetry.instrumentation.openai",
            "opentelemetry.instrumentation.openai.v2",
        )
    elif instrumentor_class_name == "RequestsInstrumentor":
        candidate_modules = (
            "opentelemetry.instrumentation.requests",
        )
    else:
        return False

    for module_path in candidate_modules:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        cls = getattr(mod, instrumentor_class_name, None)
        if cls is None:
            continue
        # BaseInstrumentor sets this class attr True after instrument().
        if getattr(cls, "_is_instrumented_by_opentelemetry", False):
            return True
        # Some instrumentors track active instances; if any exist, treat as
        # active even when the class flag is missing.
        active_instances = getattr(cls, "_instance", None)
        if active_instances is not None:
            return True
    return False


class AgentReachInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for Agent-Reach."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: ExtendedTelemetryHandler | None = None
        self._metrics: AgentReachMetrics | None = None
        self._patched_channel_classes: set[type] = set()
        self._mcp_patched = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    # ── helpers ────────────────────────────────────────────────

    def _patch_all_channels(self, _module=None) -> None:
        """Wrap ``check()`` on every ``Channel`` subclass currently loaded.

        Idempotent: keeps a set of classes we've already patched so a repeated
        ``instrument()`` call (or a re-triggered post-import hook) does not
        double-wrap. Subclasses that appear later via dynamic imports are
        picked up on the next call, but since Agent-Reach registers all 13
        channels at import time, one pass is sufficient in practice.
        """
        try:
            from agent_reach.channels.base import Channel  # noqa: PLC0415
        except ImportError:
            return

        wrapper = ChannelCheckWrapper(self._handler)
        for subclass in self._iter_channel_subclasses(Channel):
            if subclass in self._patched_channel_classes:
                continue
            try:
                wrap_function_wrapper(
                    module=subclass.__module__,
                    name=f"{subclass.__name__}.check",
                    wrapper=wrapper,
                )
                self._patched_channel_classes.add(subclass)
                logger.debug(
                    "Patched channel %s.%s.check",
                    subclass.__module__,
                    subclass.__name__,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Failed to patch %s.check: %s",
                    subclass.__name__,
                    e,
                )

    @staticmethod
    def _iter_channel_subclasses(base_cls: type):
        """Yield every concrete subclass of ``Channel`` (recursive)."""
        seen: set[type] = set()
        stack = list(base_cls.__subclasses__())
        while stack:
            cls = stack.pop()
            if cls in seen:
                continue
            seen.add(cls)
            yield cls
            stack.extend(cls.__subclasses__())

    # ── instrument / uninstrument ───────────────────────────────

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        meter = get_meter(__name__, __version__, meter_provider=meter_provider)
        self._metrics = AgentReachMetrics(meter)
        # Stash metrics on the handler so wrappers can reach it without an
        # extra constructor argument (ExtendedTelemetryHandler is shared
        # with other instrumentation code; we use a private attribute name
        # to avoid colliding with its internals).
        try:
            self._handler.__dict__["_agent_reach_metrics"] = self._metrics
        except Exception:  # noqa: BLE001
            pass

        entry_wrapper = EntryWrapper(self._handler)
        doctor_wrapper = DoctorWrapper(self._handler)
        probe_wrapper = ProbeWrapper(self._handler)
        transcribe_wrapper = TranscribeWrapper(self._handler)
        transcribe_chunk_wrapper = TranscribeChunkWrapper(self._handler)

        try:
            wrap_function_wrapper(_CLI_MODULE, "main", entry_wrapper)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to instrument agent_reach.cli.main: %s", e)

        try:
            wrap_function_wrapper(
                _DOCTOR_MODULE, "check_all", doctor_wrapper
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to instrument agent_reach.doctor.check_all: %s", e
            )

        try:
            wrap_function_wrapper(
                _PROBE_MODULE, "probe_command", probe_wrapper
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to instrument agent_reach.probe.probe_command: %s", e
            )

        try:
            wrap_function_wrapper(
                _TRANSCRIBE_MODULE, "transcribe", transcribe_wrapper
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to instrument agent_reach.transcribe.transcribe: %s",
                e,
            )

        try:
            wrap_function_wrapper(
                _TRANSCRIBE_MODULE,
                "transcribe_chunk",
                transcribe_chunk_wrapper,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to instrument agent_reach.transcribe.transcribe_chunk: %s",
                e,
            )

        try:
            register_post_import_hook(
                self._patch_all_channels, _CHANNELS_MODULE
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to register post-import hook for %s: %s",
                _CHANNELS_MODULE,
                e,
            )
        # If the channels module is already imported (common case when
        # instrument() runs late), the hook will not fire; patch now.
        try:
            already_loaded = importlib.util.find_spec(_CHANNELS_MODULE)
            if already_loaded is not None:
                mod = importlib.import_module(_CHANNELS_MODULE)
                if mod is not None:
                    self._patch_all_channels(mod)
        except Exception as e:  # noqa: BLE001
            logger.debug("Late channel patch skipped: %s", e)

        # MCP Server — only when standalone MCP instrumentation is OFF.
        try:
            mcp_active = _is_instrumentor_active("MCPInstrumentor")
        except Exception:  # noqa: BLE001
            mcp_active = False
        if not mcp_active:
            try:
                wrap_function_wrapper(
                    _MCP_SERVER_MODULE,
                    "create_server",
                    MCPServerWrapper(self._handler),
                )
                self._mcp_patched = True
                logger.info(
                    "Agent-Reach MCP Server instrumentation enabled "
                    "(no standalone MCP instrumentor detected)"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to instrument agent_reach.integrations.mcp_server.create_server: %s",
                    e,
                )
        else:
            logger.info(
                "Standalone MCP instrumentor active — skipping "
                "Agent-Reach MCP Server instrumentation to avoid duplicate spans"
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        for module_path, attr in (
            (_CLI_MODULE, "main"),
            (_DOCTOR_MODULE, "check_all"),
            (_PROBE_MODULE, "probe_command"),
            (_TRANSCRIBE_MODULE, "transcribe"),
            (_TRANSCRIBE_MODULE, "transcribe_chunk"),
            (_MCP_SERVER_MODULE, "create_server"),
        ):
            try:
                mod = importlib.import_module(module_path)
                unwrap(mod, attr)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Failed to uninstrument %s.%s: %s", module_path, attr, e
                )

        for subclass in list(self._patched_channel_classes):
            try:
                unwrap(subclass, "check")
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Failed to uninstrument %s.check: %s",
                    subclass.__name__,
                    e,
                )
        self._patched_channel_classes.clear()
        self._mcp_patched = False
