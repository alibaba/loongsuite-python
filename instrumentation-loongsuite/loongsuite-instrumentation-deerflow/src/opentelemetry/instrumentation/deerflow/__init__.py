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

"""LoongSuite instrumentation for ByteDance DeerFlow 2.x."""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Collection

from packaging.version import InvalidVersion, Version

from opentelemetry.instrumentation.deerflow.internal.patch import (
    instrument_deerflow,
    remove_owned_langchain_alias_wrappers,
    uninstrument_deerflow,
)
from opentelemetry.instrumentation.deerflow.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

__all__ = ["DeerFlowInstrumentor"]

logger = logging.getLogger(__name__)

_DEERFLOW_DISTRIBUTION = "deerflow-harness"
_MINIMUM_VERSION = Version("2")
_MAXIMUM_VERSION = Version("3")


def _is_langchain_instrumentor(instrumentor: BaseInstrumentor) -> bool:
    return (
        instrumentor.__class__.__module__
        == "opentelemetry.instrumentation.langchain"
    )


def _deerflow_runtime_supported() -> bool:
    """Validate the source-installed DeerFlow distribution at runtime."""
    try:
        importlib.import_module("deerflow")
    except Exception:  # noqa: BLE001
        logger.debug(
            "DeerFlow could not be imported; instrumentation skipped.",
            exc_info=True,
        )
        return False

    try:
        installed_version = Version(version(_DEERFLOW_DISTRIBUTION))
    except PackageNotFoundError:
        logger.debug(
            "The deerflow module has no deerflow-harness distribution "
            "metadata; DeerFlow instrumentation skipped. Install DeerFlow "
            "2.x from the official source repository."
        )
        return False
    except InvalidVersion as exc:
        logger.debug(
            "DeerFlow has an invalid distribution version (%s); "
            "instrumentation skipped.",
            exc,
        )
        return False
    except Exception:  # noqa: BLE001
        logger.debug(
            "DeerFlow distribution metadata could not be read; "
            "instrumentation skipped.",
            exc_info=True,
        )
        return False

    if not _MINIMUM_VERSION <= installed_version < _MAXIMUM_VERSION:
        logger.debug(
            "DeerFlow instrumentation supports deerflow-harness >=2,<3; "
            "found %s. Instrumentation skipped.",
            installed_version,
        )
        return False
    return True


def _instrument_dependency(
    module_name: str,
    class_name: str,
    **kwargs: Any,
) -> BaseInstrumentor | None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or (
            exc.name is not None and module_name.startswith(f"{exc.name}.")
        ):
            logger.warning(
                "DeerFlow instrumentation requires %s; continuing without it.",
                module_name,
            )
            return None
        raise

    instrumentor_type = getattr(module, class_name, None)
    if instrumentor_type is None:
        logger.warning(
            "DeerFlow instrumentation could not find %s.%s",
            module_name,
            class_name,
        )
        return None

    instrumentor = instrumentor_type()
    if instrumentor.is_instrumented_by_opentelemetry:
        return None
    instrumentor.instrument(**kwargs)
    if instrumentor.is_instrumented_by_opentelemetry:
        return instrumentor
    return None


class DeerFlowInstrumentor(BaseInstrumentor):
    """Instrument DeerFlow graphs, Gateway runs, and embedded streams."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        self._dependency_instrumentors: list[BaseInstrumentor] = []
        self._deerflow_patched = False
        if not _deerflow_runtime_supported():
            return

        try:
            for module_name, class_name in (
                (
                    "opentelemetry.instrumentation.langchain",
                    "LangChainInstrumentor",
                ),
                (
                    "opentelemetry.instrumentation.langgraph",
                    "LangGraphInstrumentor",
                ),
            ):
                instrumentor = _instrument_dependency(
                    module_name,
                    class_name,
                    **kwargs,
                )
                if instrumentor is not None:
                    self._dependency_instrumentors.append(instrumentor)

            from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
                ExtendedTelemetryHandler,
            )

            handler = ExtendedTelemetryHandler(
                tracer_provider=kwargs.get("tracer_provider"),
                meter_provider=kwargs.get("meter_provider"),
                logger_provider=kwargs.get("logger_provider"),
            )
            self._deerflow_patched = instrument_deerflow(handler)
        except BaseException:
            uninstrument_deerflow()
            owned_langchain = any(
                _is_langchain_instrumentor(instrumentor)
                for instrumentor in self._dependency_instrumentors
            )
            for instrumentor in reversed(self._dependency_instrumentors):
                instrumentor.uninstrument()
            if owned_langchain:
                remove_owned_langchain_alias_wrappers()
            self._dependency_instrumentors = []
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        if getattr(self, "_deerflow_patched", False):
            uninstrument_deerflow()
        self._deerflow_patched = False

        dependency_instrumentors = getattr(
            self, "_dependency_instrumentors", []
        )
        owned_langchain = any(
            _is_langchain_instrumentor(instrumentor)
            for instrumentor in dependency_instrumentors
        )
        for instrumentor in reversed(dependency_instrumentors):
            try:
                instrumentor.uninstrument()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Failed to uninstrument DeerFlow dependency %s",
                    instrumentor.__class__.__name__,
                    exc_info=True,
                )
        if owned_langchain:
            remove_owned_langchain_alias_wrappers()
        self._dependency_instrumentors = []
