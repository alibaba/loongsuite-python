# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("copaw")
pytest.importorskip("agentscope.message")

from agentscope.message import Msg, TextBlock  # noqa: E402


@pytest.mark.asyncio
async def test_child_agent_process_does_not_emit_entry_span(
    instrument,
    span_exporter,
    monkeypatch,
):
    monkeypatch.setenv("COPAW_OTEL_CHILD_AGENT", "1")
    monkeypatch.setenv(
        "TRACEPARENT",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )

    from copaw.app.runner.runner import AgentRunner  # noqa: PLC0415

    async def fake_resolve(self, session_id, query):
        del self, session_id, query
        denial = Msg(
            name="Friday",
            role="assistant",
            content=[TextBlock(type="text", text="child-no-entry")],
        )
        return (denial, True, None)

    monkeypatch.setattr(AgentRunner, "_resolve_pending_approval", fake_resolve)
    runner = AgentRunner(agent_id="child-agent")
    request = SimpleNamespace(
        session_id="sess-child", user_id="user-child", channel="console"
    )

    async for _ in runner.query_handler([], request):
        pass

    assert not span_exporter.get_finished_spans()
