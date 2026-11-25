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

"""Propagate the active trace into the framework shell tool's child process."""

from __future__ import annotations

import logging
import os
from importlib import import_module

from wrapt import ObjectProxy

from opentelemetry import propagate, trace

from ._constants import COPAW_OTEL_CHILD_AGENT, COPAW_OTEL_INJECT_SHELL_TRACE
from ._env_carrier import EnvironmentSetter

logger = logging.getLogger(__name__)


def should_inject_trace_for_shell_command(command: str) -> bool:
    if os.environ.get(COPAW_OTEL_INJECT_SHELL_TRACE, "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    command = command.lower()
    return (
        ("copaw" in command or "qwenpaw" in command)
        and "agents" in command
        and "chat" in command
    )


def _build_subprocess_env(command, env):
    if not should_inject_trace_for_shell_command(command):
        return env
    if not trace.get_current_span().get_span_context().is_valid:
        return env
    delta = {}
    try:
        propagate.inject(delta, setter=EnvironmentSetter())
    except Exception:
        logger.debug(
            "Failed to inject trace into shell environment", exc_info=True
        )
        return env
    if not delta:
        return env
    return {
        **(os.environ if env is None else env),
        **delta,
        COPAW_OTEL_CHILD_AGENT: "1",
    }


class _ShellAsyncio(ObjectProxy):
    """Intercept only process creation in the framework's own shell module."""

    def __init__(self, module):
        super().__init__(module.asyncio)
        self._self_windows_runner = getattr(
            module, "_execute_subprocess_sync", None
        )

    async def create_subprocess_shell(self, cmd, *args, **kwargs):
        kwargs["env"] = _build_subprocess_env(cmd, kwargs.get("env"))
        return await self.__wrapped__.create_subprocess_shell(
            cmd, *args, **kwargs
        )

    async def to_thread(self, func, *args, **kwargs):
        if func is self._self_windows_runner and len(args) >= 4:
            cmd, cwd, timeout, env, *rest = args
            args = (cmd, cwd, timeout, _build_subprocess_env(cmd, env), *rest)
        return await self.__wrapped__.to_thread(func, *args, **kwargs)


def instrument_shell(distribution):
    try:
        module = import_module(f"{distribution}.agents.tools.shell")
    except ImportError:
        logger.debug(
            "Shell tool is unavailable in %s", distribution, exc_info=True
        )
        return None
    if not hasattr(module, "asyncio"):
        return None
    proxy = _ShellAsyncio(module)
    module.asyncio = proxy
    return module, proxy


def uninstrument_shell(binding):
    if binding is not None:
        module, proxy = binding
        if module.asyncio is proxy:
            module.asyncio = proxy.__wrapped__
