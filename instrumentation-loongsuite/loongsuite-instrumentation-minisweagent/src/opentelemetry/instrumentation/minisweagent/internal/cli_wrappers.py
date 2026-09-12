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

"""CLI ENTRY: ``mini`` is exposed as Typer ``app``, not Typer-decorated ``main``."""

from __future__ import annotations

import logging
import sys
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.instrumentation.minisweagent.config import ENTRY_SPAN_ACTIVE
from opentelemetry.instrumentation.minisweagent.internal.conversation import (
    apply_payload_to_entry_invocation,
    try_fill_entry_payload_from_mini_trajectory,
)

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_otel_loongsuite_mini_app_patched"
_ORIG_APP_ATTR = "_otel_loongsuite_orig_mini_app"


class _MiniTyperAppProxy:
    """Delegates to real Typer/Click ``app``; ``__call__`` wraps ENTRY span."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)

    def _hydrate_entry(self, entry_inv: Any) -> None:
        try:
            payload = try_fill_entry_payload_from_mini_trajectory()
            if payload:
                apply_payload_to_entry_invocation(entry_inv, payload)
        except Exception:
            logger.debug("ENTRY traj hydrate failed", exc_info=True)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from opentelemetry.util.genai.extended_handler import (
            get_extended_telemetry_handler,  # noqa: PLC0415
        )
        from opentelemetry.util.genai.extended_types import (
            EntryInvocation,  # noqa: PLC0415
        )
        from opentelemetry.util.genai.types import (
            Error as GenAIError,  # noqa: PLC0415
        )

        han = get_extended_telemetry_handler()
        entry_inv = EntryInvocation()
        token = ENTRY_SPAN_ACTIVE.set(True)
        han.start_entry(entry_inv, context=context_api.get_current())
        try:
            result = self._inner(*args, **kwargs)
        except Exception as exc:
            self._hydrate_entry(entry_inv)
            han.fail_entry(
                entry_inv,
                GenAIError(message=str(exc), type=type(exc)),
            )
            raise
        except BaseException:
            self._hydrate_entry(entry_inv)
            han.stop_entry(entry_inv)
            raise
        finally:
            ENTRY_SPAN_ACTIVE.reset(token)

        self._hydrate_entry(entry_inv)
        han.stop_entry(entry_inv)
        return result

    # Typer exposes click commands via attribute access — forward everything.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def patch_mini_cli_app_module() -> None:
    """Replace ``minisweagent.run.mini.app`` once the module is loaded."""
    try:
        import minisweagent.environments as envs_mod
        import minisweagent.run.mini as mini_mod
    except Exception as exc:
        logger.debug(
            "minisweagent.run.mini not available for ENTRY patch: %s", exc
        )
        return
    if hasattr(mini_mod, "get_environment"):
        mini_mod.get_environment = envs_mod.get_environment
    if getattr(mini_mod, _PATCH_FLAG, False):
        return
    inner = getattr(mini_mod, "app", None)
    if inner is None or isinstance(inner, _MiniTyperAppProxy):
        return
    setattr(mini_mod, _ORIG_APP_ATTR, inner)
    setattr(mini_mod, "app", _MiniTyperAppProxy(inner))
    setattr(mini_mod, _PATCH_FLAG, True)


def unpatch_mini_cli_app_module() -> None:
    try:
        mini_mod = sys.modules.get("minisweagent.run.mini")
        if mini_mod is None or not getattr(mini_mod, _PATCH_FLAG, False):
            return
        orig = getattr(mini_mod, _ORIG_APP_ATTR, None)
        if orig is not None:
            mini_mod.app = orig  # type: ignore[assignment]
        delattr(mini_mod, _PATCH_FLAG)
        delattr(mini_mod, _ORIG_APP_ATTR)
    except Exception as exc:
        logger.debug("unpatch mini app failed: %s", exc)
