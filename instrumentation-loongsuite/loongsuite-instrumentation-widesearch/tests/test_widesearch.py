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

"""Tests for WideSearch instrumentation.

Covers:
- Instrumentor lifecycle (instrument/uninstrument idempotency)
- 5 span types: ENTRY, AGENT, STEP, TOOL, TASK
- Parent-child relationships
- Key attributes
- Error paths
"""

from __future__ import annotations

import asyncio
import json

import pytest

from opentelemetry.trace import StatusCode

from .conftest import (
    ActionStep,
    ActionStepError,
    Agent,
    InternalResponse,
    LLMOutputItem,
    ModelResponse,
    StepStatus,
    ToolCall,
)


def _run_async(coro):
    """Helper to run async coroutines in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _run_async_gen(async_gen):
    """Helper to consume an async generator."""

    async def _consume():
        results = []
        async for item in async_gen:
            results.append(item)
        return results

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_consume())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Instrumentor Lifecycle Tests
# ---------------------------------------------------------------------------


class TestInstrumentorLifecycle:
    def test_instrument_and_uninstrument(
        self, tracer_provider, meter_provider
    ):
        from opentelemetry.instrumentation.widesearch import (
            WideSearchInstrumentor,
        )

        instrumentor = WideSearchInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )
        assert instrumentor._handler is not None
        instrumentor.uninstrument()
        assert instrumentor._handler is None

    def test_double_instrument_uninstrument(
        self, tracer_provider, meter_provider
    ):
        from opentelemetry.instrumentation.widesearch import (
            WideSearchInstrumentor,
        )

        instrumentor = WideSearchInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )
        instrumentor.uninstrument()

        instrumentor2 = WideSearchInstrumentor()
        instrumentor2.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )
        assert instrumentor2._handler is not None
        instrumentor2.uninstrument()

    def test_instrumentation_dependencies(self):
        from opentelemetry.instrumentation.widesearch import (
            WideSearchInstrumentor,
        )

        instrumentor = WideSearchInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert ("widesearch >= 0.1.0",) == deps


# ---------------------------------------------------------------------------
# ENTRY Span Tests (H1: run_single_query)
# ---------------------------------------------------------------------------


class TestEntrySpan:
    def test_entry_span_created(self, span_exporter, instrument):
        """run_single_query should produce an ENTRY span."""
        from src.agent.run import run_single_query

        _run_async(run_single_query("What is AI?", agent_name="searcher"))

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) == 1

        entry = entry_spans[0]
        attrs = dict(entry.attributes)
        assert attrs.get("gen_ai.span.kind") == "ENTRY"
        assert attrs.get("gen_ai.operation.name") == "enter"
        assert attrs.get("gen_ai.framework") == "widesearch"

    def test_entry_span_records_gen_ai_arms_semantic_attrs(
        self, span_exporter, instrument
    ):
        """ENTRY should record input/output messages, but not agent-only metadata.

        Controlled by OTEL_SEMCONV_STABILITY_OPT_IN + SPAN_ONLY capture mode (see conftest).
        """
        from src.agent.run import run_single_query

        tools_desc = [
            {
                "type": "function",
                "function": {
                    "name": "search_global",
                    "description": "Search the web",
                    "properties": {},
                },
            }
        ]

        _run_async(
            run_single_query(
                "What is AI?",
                agent_name="searcher",
                system_prompt="You are an expert researcher.",
                tools_desc=tools_desc,
            )
        )

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) == 1
        attrs = dict(entry_spans[0].attributes)
        assert "gen_ai.input.messages" in attrs
        assert '"role":"user"' in attrs["gen_ai.input.messages"]
        assert "gen_ai.output.messages" in attrs
        assert "gen_ai.system_instructions" not in attrs
        assert "gen_ai.tool.definitions" not in attrs

    def test_entry_span_error(self, span_exporter, instrument):
        """ENTRY span should record ERROR on exception."""
        from src.agent.run import Runner, run_single_query

        async def failing_step(*, agent, memory):
            raise RuntimeError("LLM connection failed")

        Runner._step_override = failing_step

        try:
            with pytest.raises(RuntimeError, match="LLM connection failed"):
                _run_async(run_single_query("test"))
        finally:
            Runner._step_override = None

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) == 1
        assert entry_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# AGENT Span Tests (H2: Runner.run)
# ---------------------------------------------------------------------------


class TestAgentSpan:
    def test_agent_span_created(self, span_exporter, instrument):
        """Runner.run should produce an AGENT span."""
        from src.agent.run import Runner

        agent = Agent(name="search-agent", model_config_name="gpt-4o")

        async def _run():
            results = []
            async for step in Runner.run(agent, "Hello"):
                results.append(step)
            return results

        _run_async(_run())

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1

        span = agent_spans[0]
        attrs = dict(span.attributes)
        assert attrs.get("gen_ai.span.kind") == "AGENT"
        assert attrs.get("gen_ai.operation.name") == "invoke_agent"
        assert attrs.get("gen_ai.agent.name") == "search-agent"
        assert attrs.get("gen_ai.framework") == "widesearch"

    def test_agent_span_records_gen_ai_arms_semantic_attrs(
        self, span_exporter, instrument
    ):
        """AGENT invoke_agent should expose ARMS-aligned message/tool attributes."""
        from src.agent.run import Runner

        tools_desc = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add numbers",
                    "parameters": {},
                },
            }
        ]

        agent = Agent(
            name="search-agent",
            model_config_name="gpt-4o",
            tools_desc=tools_desc,
            instructions="Solve tasks with tools.",
        )

        async def _run():
            results = []
            async for step in Runner.run(agent, "Hello"):
                results.append(step)
            return results

        _run_async(_run())

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1
        attrs = dict(agent_spans[0].attributes)
        assert "gen_ai.input.messages" in attrs
        assert '"role":"user"' in attrs["gen_ai.input.messages"]
        assert "gen_ai.output.messages" in attrs
        assert "gen_ai.system_instructions" in attrs
        assert "gen_ai.tool.definitions" in attrs
        assert "add" in attrs["gen_ai.tool.definitions"]

    def test_agent_span_is_child_of_entry(self, span_exporter, instrument):
        """AGENT span should be a child of ENTRY span."""
        from src.agent.run import run_single_query

        _run_async(run_single_query("test query", agent_name="test"))

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        agent_spans = [s for s in spans if "invoke_agent" in s.name]

        assert len(entry_spans) == 1
        assert len(agent_spans) == 1

        entry = entry_spans[0]
        agent = agent_spans[0]
        assert agent.parent.span_id == entry.context.span_id

    def test_agent_span_error(self, span_exporter, instrument):
        """AGENT span should record ERROR when _step raises."""
        from src.agent.run import Runner

        async def failing_step(*, agent, memory):
            raise ValueError("Step failure")

        Runner._step_override = failing_step
        agent = Agent(name="fail-agent")

        async def _run():
            async for _ in Runner.run(agent, "Hello"):
                pass

        try:
            with pytest.raises(ValueError):
                _run_async(_run())
        finally:
            Runner._step_override = None

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1
        assert agent_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# STEP Span Tests (H3: Runner._step)
# ---------------------------------------------------------------------------


class TestStepSpan:
    def test_step_span_created(self, span_exporter, instrument):
        """Runner._step should produce a STEP span."""
        from src.agent.run import Runner

        agent = Agent(name="stepper")

        async def _run():
            async for _ in Runner.run(agent, "test"):
                pass

        _run_async(_run())

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) >= 1

        step = step_spans[0]
        attrs = dict(step.attributes)
        assert attrs.get("gen_ai.span.kind") == "STEP"
        assert attrs.get("gen_ai.operation.name") == "react"
        assert attrs.get("gen_ai.react.round") == 1

    def test_step_span_is_child_of_agent(self, span_exporter, instrument):
        """STEP span should be child of AGENT span."""
        from src.agent.run import Runner

        agent = Agent(name="stepper")

        async def _run():
            async for _ in Runner.run(agent, "test"):
                pass

        _run_async(_run())

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        step_spans = [s for s in spans if s.name == "react step"]

        assert len(agent_spans) == 1
        assert len(step_spans) >= 1

        agent_span = agent_spans[0]
        step_span = step_spans[0]
        assert step_span.parent.span_id == agent_span.context.span_id

    def test_step_span_finish_reason_finished(self, span_exporter, instrument):
        """STEP span should have finish_reason='finished' when step finishes."""
        from src.agent.run import Runner

        agent = Agent(name="stepper")

        async def _run():
            async for _ in Runner.run(agent, "test"):
                pass

        _run_async(_run())

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) >= 1
        attrs = dict(step_spans[0].attributes)
        assert attrs.get("gen_ai.react.finish_reason") == "finished"

    def test_step_span_error_on_action_step_error(
        self, span_exporter, instrument
    ):
        """STEP span should record ERROR when _step returns ActionStepError."""
        from src.agent.run import Runner

        async def error_step(*, agent, memory):
            return ActionStepError(message="LLM timeout")

        Runner._step_override = error_step
        agent = Agent(name="error-agent")

        try:

            async def _run():
                async for _ in Runner.run(agent, "test"):
                    pass

            _run_async(_run())
        finally:
            Runner._step_override = None

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) >= 1
        assert step_spans[0].status.status_code == StatusCode.ERROR
        attrs = dict(step_spans[0].attributes)
        assert attrs.get("gen_ai.react.finish_reason") == "error"


# ---------------------------------------------------------------------------
# TOOL Span Tests (H4: Runner._invoke_tool_call)
# ---------------------------------------------------------------------------


class TestToolSpan:
    def test_tool_span_created(self, span_exporter, instrument):
        """_invoke_tool_call should produce TOOL spans."""
        from src.agent.run import Runner

        async def mock_tool(**kwargs):
            return InternalResponse(data="search results")

        agent = Agent(
            name="tool-agent",
            tools={"search_global": mock_tool},
            tools_desc=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_global",
                        "description": "Search the web",
                        "parameters": {},
                    },
                }
            ],
        )

        tc = ToolCall(
            tool_name="search_global",
            arguments='{"q": "AI"}',
            tool_call_id="call_123",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        _run_async(Runner._invoke_tool_call(agent, model_resp))

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1

        span = tool_spans[0]
        attrs = dict(span.attributes)
        assert attrs.get("gen_ai.span.kind") == "TOOL"
        assert attrs.get("gen_ai.operation.name") == "execute_tool"
        assert attrs.get("gen_ai.tool.name") == "search_global"
        assert attrs.get("gen_ai.tool.call.id") == "call_123"
        assert attrs.get("gen_ai.framework") == "widesearch"

    def test_tool_span_records_arguments_and_result(
        self, span_exporter, instrument
    ):
        """TOOL span should record arguments and result."""
        from src.agent.run import Runner

        async def mock_tool(q=""):
            return InternalResponse(data=f"results for: {q}")

        agent = Agent(
            name="tool-agent",
            tools={"search_global": mock_tool},
        )

        tc = ToolCall(
            tool_name="search_global",
            arguments=json.dumps({"q": "OpenTelemetry"}),
            tool_call_id="call_456",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        results = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert len(results) == 1
        assert results[0].content == "results for: OpenTelemetry"

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        attrs = dict(tool_spans[0].attributes)
        assert "gen_ai.tool.call.arguments" in attrs
        assert "gen_ai.tool.call.result" in attrs

    def test_tool_span_error_on_missing_tool(self, span_exporter, instrument):
        """TOOL span should record ERROR when tool not found."""
        from src.agent.run import Runner

        agent = Agent(name="tool-agent", tools={})

        tc = ToolCall(
            tool_name="nonexistent_tool",
            arguments="{}",
            tool_call_id="call_789",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        results = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert len(results) == 1
        assert results[0].error_marker is not None

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code == StatusCode.ERROR

    def test_tool_span_error_on_exception(self, span_exporter, instrument):
        """TOOL span should record ERROR when tool raises exception."""
        from src.agent.run import Runner

        async def failing_tool(**kwargs):
            raise ConnectionError("Network error")

        agent = Agent(
            name="tool-agent",
            tools={"flaky_tool": failing_tool},
        )

        tc = ToolCall(
            tool_name="flaky_tool",
            arguments="{}",
            tool_call_id="call_err",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        results = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert len(results) == 1
        assert results[0].error_marker is not None
        assert "Network error" in results[0].error_marker.message

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code == StatusCode.ERROR

    def test_multiple_tool_spans(self, span_exporter, instrument):
        """Multiple tool_calls should produce multiple TOOL spans."""
        from src.agent.run import Runner

        async def mock_search(**kwargs):
            return InternalResponse(data="search result")

        async def mock_browse(**kwargs):
            return InternalResponse(data="page content")

        agent = Agent(
            name="multi-tool",
            tools={
                "search_global": mock_search,
                "text_browser_view": mock_browse,
            },
        )

        tc1 = ToolCall(
            tool_name="search_global",
            arguments='{"q": "test"}',
            tool_call_id="call_1",
        )
        tc2 = ToolCall(
            tool_name="text_browser_view",
            arguments='{"url": "http://example.com"}',
            tool_call_id="call_2",
        )
        model_resp = ModelResponse(
            outputs=[LLMOutputItem(tool_calls=[tc1, tc2])]
        )

        results = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert len(results) == 2

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 2


# ---------------------------------------------------------------------------
# TASK Span Tests (H5: create_sub_agents_wrap)
# ---------------------------------------------------------------------------


class TestTaskSpan:
    def test_task_span_created(self, span_exporter, instrument):
        """create_sub_agents closure should produce a TASK span."""
        from src.agent.multi_agent_tools import create_sub_agents_wrap

        closure = create_sub_agents_wrap(
            "main-agent", "gpt-4o", {}, [], "system prompt"
        )

        sub_agents = [
            {"index": 0, "prompt": "Search for X"},
            {"index": 1, "prompt": "Search for Y"},
        ]

        result = _run_async(closure(sub_agents))
        assert result is not None

        spans = span_exporter.get_finished_spans()
        task_spans = [
            s for s in spans if s.name == "run_task create_sub_agents"
        ]
        assert len(task_spans) == 1

        span = task_spans[0]
        attrs = dict(span.attributes)
        assert attrs.get("gen_ai.span.kind") == "TASK"
        assert attrs.get("gen_ai.operation.name") == "run_task"
        assert attrs.get("gen_ai.framework") == "widesearch"
        assert "input.value" in attrs

    def test_task_span_records_output(self, span_exporter, instrument):
        """TASK span should record output.value."""
        from src.agent.multi_agent_tools import create_sub_agents_wrap

        closure = create_sub_agents_wrap("agent", "gpt-4o", {}, [], "prompt")

        sub_agents = [{"index": 0, "prompt": "find info"}]
        _run_async(closure(sub_agents))

        spans = span_exporter.get_finished_spans()
        task_spans = [
            s for s in spans if s.name == "run_task create_sub_agents"
        ]
        assert len(task_spans) == 1
        attrs = dict(task_spans[0].attributes)
        assert "output.value" in attrs

    def test_task_span_error(self, span_exporter, instrument):
        """TASK span should record ERROR when closure raises."""

        # Temporarily replace create_sub_agents_wrap's inner closure behavior
        import src.agent.multi_agent_tools as mat

        original = mat.create_sub_agents_wrap

        def error_factory(*args, **kwargs):
            original(*args, **kwargs)

            async def error_closure(sub_agents):
                raise RuntimeError("Sub-agent execution failed")

            return error_closure

        mat.create_sub_agents_wrap = error_factory

        # Re-instrument to pick up the new function

        instrument.uninstrument()
        instrument.instrument(
            tracer_provider=span_exporter._tracer_provider
            if hasattr(span_exporter, "_tracer_provider")
            else None,
            skip_dep_check=True,
        )

        # Since re-instrumentation is complex, let's just test the wrapper directly
        # by calling the instrumented version
        instrument.uninstrument()

        # Simpler approach: directly test the wrap function
        from opentelemetry.instrumentation.widesearch.patch import (
            wrap_create_sub_agents_factory,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        from opentelemetry.util.genai.extended_handler import (
            ExtendedTelemetryHandler,
        )

        exporter = InMemorySpanExporter()
        tp = TracerProvider()
        tp.add_span_processor(SimpleSpanProcessor(exporter))
        handler = ExtendedTelemetryHandler(tracer_provider=tp)

        def failing_factory(*args, **kwargs):
            async def failing_closure(sub_agents):
                raise RuntimeError("Boom")

            return failing_closure

        wrapped_factory = wrap_create_sub_agents_factory(
            failing_factory, None, (), {}, handler=handler
        )

        with pytest.raises(RuntimeError, match="Boom"):
            _run_async(wrapped_factory([{"index": 0, "prompt": "x"}]))

        spans = exporter.get_finished_spans()
        task_spans = [
            s for s in spans if s.name == "run_task create_sub_agents"
        ]
        assert len(task_spans) == 1
        assert task_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Parent-Child Relationship Tests
# ---------------------------------------------------------------------------


class TestParentChildRelationships:
    def test_full_hierarchy_entry_agent_step(self, span_exporter, instrument):
        """Full call through run_single_query should produce ENTRY > AGENT > STEP."""
        from src.agent.run import run_single_query

        _run_async(run_single_query("hierarchy test", agent_name="root"))

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        step_spans = [s for s in spans if s.name == "react step"]

        assert len(entry_spans) == 1
        assert len(agent_spans) == 1
        assert len(step_spans) >= 1

        entry = entry_spans[0]
        agent = agent_spans[0]
        step = step_spans[0]

        # AGENT is child of ENTRY
        assert agent.parent.span_id == entry.context.span_id
        # STEP is child of AGENT
        assert step.parent.span_id == agent.context.span_id

    def test_tool_span_is_child_of_step(self, span_exporter, instrument):
        """TOOL span should be child of the STEP span when invoked during a step."""
        from src.agent.run import Runner

        async def mock_tool(**kwargs):
            return InternalResponse(data="result")

        agent = Agent(
            name="hierarchy-agent",
            tools={"my_tool": mock_tool},
        )

        async def custom_step(*, agent, memory):
            tc = ToolCall(
                tool_name="my_tool",
                arguments="{}",
                tool_call_id="tc_hier",
            )
            model_resp = ModelResponse(
                outputs=[LLMOutputItem(tool_calls=[tc])]
            )
            await Runner._invoke_tool_call(agent, model_resp)
            return ActionStep(step_status=StepStatus.FINISHED, content="done")

        Runner._step_override = custom_step

        try:

            async def _run():
                async for _ in Runner.run(agent, "test"):
                    pass

            _run_async(_run())
        finally:
            Runner._step_override = None

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        tool_spans = [s for s in spans if "execute_tool" in s.name]

        assert len(step_spans) >= 1
        assert len(tool_spans) >= 1

        step_span = step_spans[0]
        tool_span = tool_spans[0]
        assert tool_span.parent.span_id == step_span.context.span_id
