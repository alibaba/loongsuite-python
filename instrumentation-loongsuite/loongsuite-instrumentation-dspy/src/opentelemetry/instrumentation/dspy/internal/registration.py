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
Global registration of the LoongSuite DSPy callback.

DSPy resolves callbacks per call from ``dspy.settings.callbacks`` plus the
component's own ``callbacks`` list, so registering once globally is enough to
observe every module, tool and evaluation.

``dspy.settings.configure`` claims ownership of the settings for the calling
thread and refuses later calls from other threads, so instrumentation must not
call it — the callback list is updated in place instead. ``configure`` is
wrapped so that a user call passing ``callbacks=[...]`` (which replaces the
list wholesale) does not silently drop the LoongSuite callback.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import dspy
from dspy.dsp.utils.settings import main_thread_config
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.dspy.internal.callback import (
    OTelDSPyCallback,
)
from opentelemetry.instrumentation.utils import unwrap

logger = logging.getLogger(__name__)

_CONFIGURE_LOCATIONS = (
    ("dspy", "configure"),
    ("dspy.dsp.utils.settings", "Settings.configure"),
)


def _current_callbacks() -> list[Any]:
    return list(main_thread_config.get("callbacks") or [])


def _install(callback: OTelDSPyCallback) -> None:
    callbacks = _current_callbacks()
    if any(isinstance(item, OTelDSPyCallback) for item in callbacks):
        return
    callbacks.append(callback)
    main_thread_config["callbacks"] = callbacks


def _make_configure_wrapper(callback: OTelDSPyCallback) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        result = wrapped(*args, **kwargs)
        if "callbacks" in kwargs:
            try:
                _install(callback)
            except Exception:
                logger.debug(
                    "Failed to re-register DSPy callback after configure",
                    exc_info=True,
                )
        return result

    return wrapper


def register_callback(callback: OTelDSPyCallback) -> None:
    """Register *callback* globally and keep it registered across ``configure``."""
    _install(callback)

    wrapper = _make_configure_wrapper(callback)
    for module_path, name in _CONFIGURE_LOCATIONS:
        try:
            wrap_function_wrapper(module_path, name, wrapper)
        except Exception:
            logger.debug(
                "Failed to wrap %s.%s", module_path, name, exc_info=True
            )


def unregister_callback() -> None:
    """Remove the LoongSuite callback and restore ``configure``."""
    try:
        main_thread_config["callbacks"] = [
            item
            for item in _current_callbacks()
            if not isinstance(item, OTelDSPyCallback)
        ]
    except Exception:
        logger.debug("Failed to unregister DSPy callback", exc_info=True)

    try:
        unwrap(dspy, "configure")
    except Exception:
        logger.debug("Failed to restore dspy.configure", exc_info=True)
    try:
        from dspy.dsp.utils.settings import Settings  # noqa: PLC0415

        unwrap(Settings, "configure")
    except Exception:
        logger.debug("Failed to restore Settings.configure", exc_info=True)
