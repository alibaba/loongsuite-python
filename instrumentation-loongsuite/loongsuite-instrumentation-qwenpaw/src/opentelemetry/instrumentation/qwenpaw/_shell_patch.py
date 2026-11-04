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

"""Inject trace context into AgentScope shell-command subprocesses."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

from opentelemetry import propagate

from ._constants import (
    COPAW_OTEL_CHILD_AGENT,
    COPAW_OTEL_INJECT_SHELL_TRACE,
)
from ._env_carrier import EnvironmentSetter

logger = logging.getLogger(__name__)

_MODULE_SHELL = "agentscope.tool._coding._shell"
_PATCH_TARGET = "execute_shell_command"


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def should_inject_trace_for_shell_command(command: str) -> bool:
    if _truthy_env(COPAW_OTEL_INJECT_SHELL_TRACE):
        return True
    command = command.lower()
    return (
        ("copaw" in command or "qwenpaw" in command)
        and "agents" in command
        and "chat" in command
    )


async def _run_shell_command_with_env(
    command: str,
    timeout: int,
    env: dict[str, str],
) -> Any:
    from agentscope.message import TextBlock  # noqa: PLC0415
    from agentscope.tool._response import ToolResponse  # noqa: PLC0415

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        bufsize=0,
        env=env,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        stdout, stderr = await proc.communicate()
        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")
        returncode = proc.returncode
    except asyncio.TimeoutError:
        suffix = (
            "TimeoutError: The command execution exceeded "
            f"the timeout of {timeout} seconds."
        )
        returncode = -1
        try:
            proc.terminate()
            stdout, stderr = await proc.communicate()
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")
            stderr_str = f"{stderr_str}\n{suffix}" if stderr_str else suffix
        except ProcessLookupError:
            stdout_str = ""
            stderr_str = suffix

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"<returncode>{returncode}</returncode>"
                    f"<stdout>{stdout_str}</stdout>"
                    f"<stderr>{stderr_str}</stderr>"
                ),
            )
        ]
    )


def _build_subprocess_env() -> dict[str, str]:
    merged = os.environ.copy()
    delta: dict[str, str] = {}
    try:
        propagate.get_global_textmap().inject(
            delta, setter=EnvironmentSetter()
        )
    except Exception:
        logger.debug("Failed to inject trace into env", exc_info=True)
        return merged
    merged.update(delta)
    merged[COPAW_OTEL_CHILD_AGENT] = "1"
    return merged


def make_execute_shell_command_wrapper() -> Callable[..., Any]:
    async def execute_shell_command_wrapper(
        wrapped: Any,
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        del instance
        command = str(args[0]) if args else str(kwargs.get("command", ""))
        timeout = (
            int(args[1]) if len(args) >= 2 else int(kwargs.get("timeout", 300))
        )
        if not should_inject_trace_for_shell_command(command):
            return await wrapped(*args, **kwargs)

        try:
            return await _run_shell_command_with_env(
                command, timeout, _build_subprocess_env()
            )
        except Exception:
            logger.debug(
                "%s.%s inject path failed; falling back to original",
                _MODULE_SHELL,
                _PATCH_TARGET,
                exc_info=True,
            )
            return await wrapped(*args, **kwargs)

    return execute_shell_command_wrapper
