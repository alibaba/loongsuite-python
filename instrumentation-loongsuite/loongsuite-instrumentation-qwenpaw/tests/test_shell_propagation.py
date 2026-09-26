# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from opentelemetry import trace
from opentelemetry.instrumentation.qwenpaw._constants import (
    COPAW_OTEL_CHILD_AGENT,
    COPAW_OTEL_INJECT_SHELL_TRACE,
)
from opentelemetry.instrumentation.qwenpaw._shell_patch import (
    _build_subprocess_env,
    should_inject_trace_for_shell_command,
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
        env = _build_subprocess_env()

    assert env[COPAW_OTEL_CHILD_AGENT] == "1"
    assert env["TRACEPARENT"].startswith("00-")
