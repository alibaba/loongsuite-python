# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opentelemetry import trace
from opentelemetry.instrumentation.qwenpaw._constants import (
    COPAW_OTEL_CHILD_AGENT,
    COPAW_OTEL_INJECT_SHELL_TRACE,
)
from opentelemetry.instrumentation.qwenpaw._shell_patch import (
    _build_subprocess_env,
    _ShellAsyncio,
    instrument_shell,
    should_inject_trace_for_shell_command,
    uninstrument_shell,
)


def test_should_inject_for_supported_agent_chat_commands():
    assert should_inject_trace_for_shell_command("copaw agents chat -m hello")
    assert should_inject_trace_for_shell_command(
        "qwenpaw agents chat -m hello"
    )
    assert not should_inject_trace_for_shell_command("ls -la")
    assert not should_inject_trace_for_shell_command("copaw app")


def test_should_inject_when_env_forces_all_shell(monkeypatch):
    monkeypatch.setenv(COPAW_OTEL_INJECT_SHELL_TRACE, "1")
    assert should_inject_trace_for_shell_command("/bin/true")


def test_build_subprocess_env_sets_child_marker_and_traceparent(
    tracer_provider,
):
    tracer = tracer_provider.get_tracer(__name__)
    with trace.use_span(tracer.start_span("parent_shell"), end_on_exit=True):
        env = _build_subprocess_env("qwenpaw agents chat", {"PATH": "/bin"})

    assert env["PATH"] == "/bin"
    assert env[COPAW_OTEL_CHILD_AGENT] == "1"
    assert env["TRACEPARENT"].startswith("00-")


def test_no_parent_context_preserves_environment():
    env = {"PATH": "/bin"}
    assert _build_subprocess_env("qwenpaw agents chat", env) is env


@pytest.mark.asyncio
async def test_native_shell_remains_owner_of_process_and_errors(
    tracer_provider,
):
    error = OSError("native creation failed")
    create = AsyncMock(side_effect=error)
    shell = _ShellAsyncio(
        SimpleNamespace(
            asyncio=SimpleNamespace(create_subprocess_shell=create)
        )
    )
    with tracer_provider.get_tracer(__name__).start_as_current_span("parent"):
        with pytest.raises(OSError) as caught:
            await shell.create_subprocess_shell(
                "copaw agents chat", cwd="/workspace", env={"EXISTING": "yes"}
            )
    assert caught.value is error
    create.assert_awaited_once()
    assert create.await_args.kwargs["cwd"] == "/workspace"
    assert create.await_args.kwargs["env"]["EXISTING"] == "yes"
    assert create.await_args.kwargs["env"][COPAW_OTEL_CHILD_AGENT] == "1"


@pytest.mark.asyncio
async def test_real_shell_retains_output_and_propagates_parent(
    tracer_provider, monkeypatch, tmp_path
):
    distribution = "qwenpaw" if find_spec("qwenpaw") else "copaw"
    module = import_module(f"{distribution}.agents.tools.shell")
    original_asyncio = module.asyncio
    monkeypatch.setenv(COPAW_OTEL_INJECT_SHELL_TRACE, "1")
    binding = instrument_shell(distribution)
    try:
        with tracer_provider.get_tracer(__name__).start_as_current_span(
            "parent"
        ) as parent:
            response = await module.execute_shell_command(
                """python -c "import os; print(os.environ['TRACEPARENT']); print('x'*262144)" """,
                timeout=10,
                cwd=tmp_path,
            )
        block = response.content[0]
        text = block["text"] if isinstance(block, dict) else block.text
        assert f"{parent.get_span_context().trace_id:032x}" in text
        assert "x" * 262144 in text
    finally:
        uninstrument_shell(binding)
    assert module.asyncio is original_asyncio
