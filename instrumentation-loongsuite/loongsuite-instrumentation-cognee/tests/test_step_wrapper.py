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

"""Tests for the STEP wrapper and its ``ContextVar`` round counter.

The critical test is ``test_concurrent_agent_loops_do_not_cross_talk`` — it
verifies that two AGENT loops running concurrently under ``asyncio.gather``
each see their own independent round sequence (1, 2, 3) without bleed.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from opentelemetry.instrumentation.cognee.internal._react_context import (
    get_react_round,
    reset_react_round,
    set_react_round,
)
from opentelemetry.instrumentation.cognee.internal._step_wrapper import (
    _infer_finish_reason,
    _is_agentic_prompt,
    _make_step_wrapper,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


@pytest.fixture
def handler():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    h = ExtendedTelemetryHandler(tracer_provider=provider)
    yield h, exporter
    provider.shutdown()


class _FakeAgentStep:
    def __init__(self, final_answer=None, tool_call=None):
        self.final_answer = final_answer
        self.tool_call = tool_call


async def _fake_generate_completion(
    query="", context="", user_prompt_path="agentic_user.txt",
    system_prompt_path="", system_prompt=None,
    conversation_history=None, response_model=str,
):
    return _FakeAgentStep(final_answer=f"answer to {query}")


async def _run_react_loop(
    step_wrapper, n_iterations: int, label: str
) -> List[int]:
    """Simulate one AGENT loop: set round=0 at entry, run N STEP iterations
    sequentially, return the round values each STEP saw, then reset the
    ContextVar at exit (mirroring the AGENT wrapper)."""
    rounds_seen: List[int] = []
    token = set_react_round(0)
    try:
        for i in range(n_iterations):
            captured_rounds: List[int] = []

            async def _probe(*args, **kwargs):
                # _step_wrapper has already incremented the ContextVar before
                # calling wrapped(*args, **kwargs). We capture the current value
                # here to verify the round the wrapper assigned to this STEP.
                captured_rounds.append(get_react_round())
                return _FakeAgentStep(final_answer=f"{label}-{i}")

            await step_wrapper(
                _probe,
                None,
                (),
                {
                    "query": f"{label}-{i}",
                    "context": "",
                    "user_prompt_path": "agentic_user.txt",
                    "system_prompt_path": "agentic_system.txt",
                },
            )
            rounds_seen.append(captured_rounds[0])
    finally:
        reset_react_round(token)
    return rounds_seen


@pytest.mark.asyncio
async def test_sequential_rounds_increment(handler):
    hdl, exporter = handler
    step_wrapper = _make_step_wrapper(hdl)

    rounds = await _run_react_loop(step_wrapper, n_iterations=3, label="A")
    assert rounds == [1, 2, 3]


@pytest.mark.asyncio
async def test_concurrent_agent_loops_do_not_cross_talk(handler):
    """Two AGENT loops running concurrently must each see 1, 2, 3."""
    hdl, exporter = handler
    step_wrapper = _make_step_wrapper(hdl)

    rounds_a, rounds_b = await asyncio.gather(
        _run_react_loop(step_wrapper, n_iterations=3, label="A"),
        _run_react_loop(step_wrapper, n_iterations=3, label="B"),
    )
    assert rounds_a == [1, 2, 3]
    assert rounds_b == [1, 2, 3]

    # ContextVar must be None outside the agent loops.
    assert get_react_round() is None


@pytest.mark.asyncio
async def test_many_concurrent_agent_loops_isolated(handler):
    hdl, _ = handler
    step_wrapper = _make_step_wrapper(hdl)

    results = await asyncio.gather(
        *(_run_react_loop(step_wrapper, n_iterations=3, label=str(i)) for i in range(10))
    )
    for rounds in results:
        assert rounds == [1, 2, 3]


@pytest.mark.asyncio
async def test_step_wrapper_passthrough_outside_agent(handler):
    """Outside AGENT context, the wrapper is a transparent passthrough."""
    hdl, exporter = handler
    step_wrapper = _make_step_wrapper(hdl)

    assert get_react_round() is None

    async def _fake(wrapped_=None, *args, **kwargs):
        return "raw"

    # _step_wrapper expects (wrapped, instance, args, kwargs)
    result = await step_wrapper(_fake, None, (), {"user_prompt_path": "agentic_user.txt"})
    assert result == "raw"
    # No span should have been created.
    assert len(exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_step_wrapper_skips_non_agentic_prompt(handler):
    """Inside AGENT context but with a non-agentic prompt path (forced final
    answer), the wrapper passes through without creating a STEP span."""
    hdl, exporter = handler
    step_wrapper = _make_step_wrapper(hdl)

    token = set_react_round(0)
    try:
        async def _fake(wrapped_=None, *args, **kwargs):
            return "forced"

        result = await step_wrapper(
            _fake, None, (), {"user_prompt_path": "answer_simple_question.txt"}
        )
        assert result == "forced"
        # No STEP span should have been emitted.
        step_spans = [
            s for s in exporter.get_finished_spans()
            if s.name == "react step"
        ]
        assert step_spans == []
    finally:
        reset_react_round(token)


@pytest.mark.asyncio
async def test_step_wrapper_emits_span_with_round_attribute(handler):
    hdl, exporter = handler
    step_wrapper = _make_step_wrapper(hdl)

    async def _real_completion(*args, **kwargs):
        return _FakeAgentStep(final_answer="ok")

    token = set_react_round(0)
    try:
        await step_wrapper(
            _real_completion,
            None,
            (),
            {
                "query": "q",
                "context": "ctx",
                "user_prompt_path": "agentic_user.txt",
                "system_prompt_path": "agentic_system.txt",
            },
        )
    finally:
        reset_react_round(token)

    spans = exporter.get_finished_spans()
    react_spans = [s for s in spans if s.name == "react step"]
    assert len(react_spans) == 1
    assert react_spans[0].attributes.get("gen_ai.react.round") == 1
    assert react_spans[0].attributes.get("gen_ai.span.kind") == "STEP"
    assert react_spans[0].attributes.get("gen_ai.react.finish_reason") == "stop"


def test_finish_reason_inference():
    assert _infer_finish_reason(_FakeAgentStep(final_answer="answer")) == "stop"
    assert _infer_finish_reason(_FakeAgentStep(tool_call={"name": "x"})) == "tool_use"
    assert _infer_finish_reason("forced string") == "max_iterations"
    assert _infer_finish_reason(None) == "unknown"


def test_is_agentic_prompt():
    assert _is_agentic_prompt("agentic_user.txt") is True
    assert _is_agentic_prompt("/some/path/agentic_user.txt") is True
    assert _is_agentic_prompt("answer_simple_question.txt") is False
    assert _is_agentic_prompt(None) is False
    assert _is_agentic_prompt("") is False
