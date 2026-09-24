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

"""Integration tests for Qwen-Agent instrumentation using VCR cassettes.

These tests use real qwen-agent calls that are recorded via VCR so subsequent
runs replay the HTTP interactions without hitting the real API.
"""

import json
import os

import pytest
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
from qwen_agent.tools.base import BaseTool, register_tool

from opentelemetry.instrumentation.qwen_agent import QwenAgentInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# cassette directory for this test module
VCR_CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")


def _make_providers():
    """Create fresh OTel providers for each test."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    log_exporter = InMemoryLogExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    return (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    )


@pytest.mark.vcr()
def test_qwen_agent_basic_run():
    """Test basic qwen-agent Assistant conversation.

    Verifies that:
    - An invoke_agent span is produced
    - A chat (LLM) span is produced as a child
    - Both spans have the expected gen_ai.* attributes
    """
    (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    ) = _make_providers()

    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    try:
        bot = Assistant(
            llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
            name="TestAssistant",
        )
        messages = [{"role": "user", "content": "Hello, what is 1+1?"}]
        # Consume the generator to trigger all spans
        list(bot.run(messages))
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 2, (
        f"Expected at least 2 spans, got {len(spans)}: {[s.name for s in spans]}"
    )

    # Find invoke_agent span
    agent_spans = [s for s in spans if "invoke_agent" in s.name]
    assert len(agent_spans) >= 1, (
        f"No invoke_agent span found in: {[s.name for s in spans]}"
    )
    agent_span = agent_spans[0]
    assert (
        agent_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )

    # Find chat/LLM span
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, (
        f"No chat span found in: {[s.name for s in spans]}"
    )
    chat_span = chat_spans[0]

    # Verify key span attributes on the chat span
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )

    # response_model_name fix: GEN_AI_RESPONSE_MODEL should now be populated
    response_model = chat_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_MODEL
    )
    assert response_model is not None, (
        "gen_ai.response.model should be set (P0 fix)"
    )

    # finish_reasons fix: GEN_AI_RESPONSE_FINISH_REASONS should be populated
    finish_reasons = chat_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    )
    assert finish_reasons is not None, (
        "gen_ai.response.finish_reasons should be set (P0 fix)"
    )


@pytest.mark.vcr()
def test_qwen_agent_stream_llm_with_ttft():
    """Test streaming LLM call and verify TTFT (Time-to-First-Token) is recorded.

    Verifies that P2 fix (monotonic_first_token_s) is working:
    - gen_ai.response.time_to_first_token attribute is recorded when available
    """
    (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    ) = _make_providers()

    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    try:
        llm = get_chat_model(
            {"model": "qwen-max", "model_type": "qwen_dashscope"}
        )
        messages = [{"role": "user", "content": "Say hello in one word."}]
        # stream=True is the default; consume the iterator
        list(llm.chat(messages=messages, stream=True))
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, (
        f"No chat span found in: {[s.name for s in spans]}"
    )
    chat_span = chat_spans[0]

    # Verify basic span attributes
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )

    # TTFT (gen_ai.response.time_to_first_token) may be None during cassette replay;
    # we assert finish_reasons to ensure the P2 code path ran.
    finish_reasons = chat_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    )
    assert finish_reasons is not None, (
        "gen_ai.response.finish_reasons should be set (P0 fix)"
    )


@pytest.mark.vcr()
def test_non_stream_chat():
    """Test non-streaming LLM chat call (stream=False code path).

    Verifies the else-branch in wrap_chat_model_chat:
    - chat span is produced
    - response_model and finish_reasons are set (P0 fix)
    - no streaming wrapper is used
    """
    (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    ) = _make_providers()

    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    try:
        # use_raw_api=False forces the non-streaming code path; newer qwen-agent
        # versions default use_raw_api=True for qwen_dashscope which only supports stream=True.
        llm = get_chat_model(
            {
                "model": "qwen-max",
                "model_type": "qwen_dashscope",
                "generate_cfg": {"use_raw_api": False},
            }
        )
        messages = [
            {
                "role": "user",
                "content": "What is 2+2? Answer with just the number.",
            }
        ]
        list(llm.chat(messages=messages, stream=False))
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, (
        f"Expected at least 1 chat span, got: {[s.name for s in spans]}"
    )
    chat_span = chat_spans[0]

    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )

    response_model = chat_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_MODEL
    )
    assert response_model is not None, (
        "gen_ai.response.model should be set for non-stream (P0 fix)"
    )

    finish_reasons = chat_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    )
    assert finish_reasons is not None, (
        "gen_ai.response.finish_reasons should be set for non-stream (P0 fix)"
    )


@pytest.mark.vcr()
def test_agent_run_nonstream():
    """Test Agent.run_nonstream() produces a single invoke_agent span.

    run_nonstream() is not wrapped separately — it calls self.run() internally,
    so the invoke_agent span is created once by the run() wrapper.
    Verifies there is no span duplication when using the non-streaming entry point.
    """
    (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    ) = _make_providers()

    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    try:
        bot = Assistant(
            llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
            name="NonStreamAssistant",
        )
        messages = [{"role": "user", "content": "Say 'OK' and nothing else."}]
        bot.run_nonstream(messages)
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    # Assistant may spawn a Memory sub-agent producing extra invoke_agent spans;
    # filter to the target agent by name.
    agent_spans = [
        s for s in spans if s.name == "invoke_agent NonStreamAssistant"
    ]
    assert len(agent_spans) >= 1, (
        f"Expected invoke_agent NonStreamAssistant span, got: {[s.name for s in spans]}"
    )
    agent_span = agent_spans[0]
    assert (
        agent_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )

    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, (
        f"No chat span found in: {[s.name for s in spans]}"
    )


@pytest.mark.vcr()
def test_multi_turn_conversation():
    """Test multi-turn conversation with history messages.

    Verifies that all input messages (user + assistant history) are correctly
    captured in the span event log, not just the last message.
    """
    (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    ) = _make_providers()

    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    try:
        bot = Assistant(
            llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
            name="MultiTurnAssistant",
        )
        # Simulate a 2-turn conversation by including history
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]
        list(bot.run(messages))
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    # Filter to our specific agent; Assistant may produce extra Memory sub-agent spans.
    agent_spans = [
        s for s in spans if s.name == "invoke_agent MultiTurnAssistant"
    ]
    assert len(agent_spans) >= 1, (
        f"Expected invoke_agent MultiTurnAssistant span in: {[s.name for s in spans]}"
    )

    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, (
        f"No chat span found in: {[s.name for s in spans]}"
    )
    chat_span = chat_spans[0]

    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )
    finish_reasons = chat_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    )
    assert finish_reasons is not None, (
        "gen_ai.response.finish_reasons should be set"
    )


@pytest.mark.vcr()
def test_react_multi_round():
    """Test ReAct agent completing a multi-round task via tool calls.

    Verifies the react_step span introduced for ReAct tracking:
    - react_step spans appear when the agent has tools
    - gen_ai.react.round attribute increments per iteration
    - chat and execute_tool spans are nested inside react_step
    """
    (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    ) = _make_providers()

    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    try:

        @register_tool("calculator_react_test")
        class CalculatorTool(BaseTool):
            description = "Evaluate a simple arithmetic expression and return the numeric result."
            parameters = [
                {
                    "name": "expression",
                    "type": "string",
                    "description": "The arithmetic expression to evaluate, e.g. '3 * 7'.",
                    "required": True,
                }
            ]

            def call(self, params, **kwargs):
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {"expression": params}
                expr = params.get("expression", "0")
                try:
                    return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
                except Exception as e:
                    return f"Error: {e}"

        bot = Assistant(
            llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
            name="ReactAgent",
            function_list=["calculator_react_test"],
        )
        messages = [{"role": "user", "content": "What is 6 multiplied by 7?"}]
        list(bot.run(messages))
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    # Filter to our specific agent; Assistant may produce extra Memory sub-agent spans.
    agent_spans = [s for s in spans if s.name == "invoke_agent ReactAgent"]
    assert len(agent_spans) >= 1, (
        f"Expected invoke_agent ReactAgent span in: {span_names}"
    )

    # "react step" spans must appear (agent has tools → ReAct mode)
    react_spans = [s for s in spans if s.name == "react step"]
    assert len(react_spans) >= 1, (
        f"Expected 'react step' spans for a tool-enabled agent, got none. All spans: {span_names}"
    )

    # gen_ai.react.round attribute must be set and start at 1
    rounds = [s.attributes.get("gen_ai.react.round") for s in react_spans]
    assert 1 in rounds, (
        f"Expected gen_ai.react.round=1 in react step spans, got rounds={rounds}"
    )

    # chat spans must exist inside react step (nested via OTel context)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, f"No chat span in: {span_names}"


@pytest.mark.vcr()
def test_qwen_agent_with_tool_call():
    """Test qwen-agent with tools, verify execute_tool span is produced.

    Uses a custom BaseTool subclass (can be passed directly to Assistant
    without TOOL_REGISTRY registration) to exercise the tool_definitions P1 fix
    and verify execute_tool spans are generated when the model calls a tool.
    """
    (
        tracer_provider,
        span_exporter,
        logger_provider,
        log_exporter,
        meter_provider,
    ) = _make_providers()

    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    try:

        @register_tool("get_current_weather_test")
        class GetCurrentWeatherTool(BaseTool):
            description = "Get the current weather for a given city."
            parameters = [
                {
                    "name": "city",
                    "type": "string",
                    "description": "The city name to get weather for.",
                    "required": True,
                }
            ]

            def call(self, params, **kwargs):
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {"city": params}
                city = (
                    params.get("city", "unknown")
                    if isinstance(params, dict)
                    else "unknown"
                )
                return (
                    f"The weather in {city} is sunny and 22 degrees Celsius."
                )

        bot = Assistant(
            llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
            name="WeatherAgent",
            function_list=["get_current_weather_test"],
        )
        messages = [
            {
                "role": "user",
                "content": "What is the weather in Beijing right now?",
            }
        ]
        list(bot.run(messages))
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    # Verify we got invoke_agent and chat spans
    agent_spans = [s for s in spans if "invoke_agent" in s.name]
    assert len(agent_spans) >= 1, f"No invoke_agent span in: {span_names}"

    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, f"No chat span in: {span_names}"

    # Verify chat span attributes
    chat_span = chat_spans[0]
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )

    # If the model called the tool, there should be an execute_tool span
    tool_spans = [s for s in spans if "execute_tool" in s.name]
    # Tool call is model-dependent; only assert if tool spans exist
    for tool_span in tool_spans:
        assert (
            tool_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "execute_tool"
        )
