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
Test cases for TOOL spans of tools that CrewAI runs by native function calling.

When the LLM supports function calling, as CrewAI's native OpenAI provider
does, CrewAI sends the tools to the model and runs the returned tool calls
through ``BaseTool.run`` instead of ``ToolUsage._use``. The OpenAI-compatible
chat completions are mocked with respx, so these tests need no cassette.
"""

import json
import os
from types import SimpleNamespace

import httpx
import respx
from crewai import LLM, Agent, Crew, Task
from crewai.tools.base_tool import BaseTool, Tool

from opentelemetry.instrumentation.crewai import (
    CrewAIInstrumentor,
    _ToolUseWrapper,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

try:
    from opentelemetry.instrumentation._semconv import (
        _OpenTelemetrySemanticConventionStability,
        _OpenTelemetryStabilitySignalType,
        _StabilityMode,
    )
except ImportError:
    _OpenTelemetrySemanticConventionStability = None
    _OpenTelemetryStabilitySignalType = None
    _StabilityMode = None

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class WeatherTool(BaseTool):
    """Sample tool for weather lookup."""

    name: str = "get_weather"
    description: str = "Get current weather for a location"

    def _run(self, location: str) -> str:
        """Execute the tool."""
        return f"Weather in {location}: Sunny, 72°F"


class FailingTool(BaseTool):
    """Sample tool that always fails."""

    name: str = "failing_tool"
    description: str = "A tool that always fails"

    def _run(self, input_str: str) -> str:
        """Execute the tool."""
        raise RuntimeError("Tool execution failed")


def chat_completion(content=None, tool_calls=None):
    """Return an OpenAI-compatible chat completion body."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-native-tools",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen-turbo",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def tool_call(call_id, name, arguments):
    """Return an OpenAI-compatible tool call."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class TestNativeToolCalls(TestBase):
    """Test TOOL spans of natively called tools."""

    def setUp(self):
        """Setup test resources."""
        super().setUp()
        os.environ["CREWAI_TRACING_ENABLED"] = "false"
        # Enable experimental mode and content capture for testing
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = (
            "gen_ai_latest_experimental"
        )
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "SPAN_ONLY"
        )

        if _OpenTelemetrySemanticConventionStability:
            try:
                _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING[
                    _OpenTelemetryStabilitySignalType.GEN_AI
                ] = _StabilityMode.GEN_AI_LATEST_EXPERIMENTAL
            except (AttributeError, KeyError):
                pass

        self.instrumentor = CrewAIInstrumentor()
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # A model that is not a known OpenAI model, on an explicit endpoint,
        # gets CrewAI's native OpenAI provider, which calls tools natively.
        self.llm = LLM(
            model="openai/qwen-turbo", base_url=BASE_URL, api_key="fake-key"
        )

    def tearDown(self):
        """Cleanup test resources."""
        with self.disable_logging():
            self.instrumentor.uninstrument()
        super().tearDown()

    def kickoff(self, tools, *tool_call_responses):
        """Run a one-task crew; the LLM answers with the given tool calls."""
        responses = iter(tool_call_responses)

        def respond(request):
            body = next(
                responses, chat_completion(content="All tools were used.")
            )
            return httpx.Response(200, json=body)

        agent = Agent(
            role="Tool User",
            goal="Use the tools",
            backstory="Agent that uses tools",
            verbose=False,
            llm=self.llm,
            tools=tools,
        )
        task = Task(
            description="Use the tools",
            expected_output="The tool results",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        with respx.mock(assert_all_called=False) as router:
            router.post(f"{BASE_URL}/chat/completions").mock(
                side_effect=respond
            )
            crew.kickoff()

        return self.memory_exporter.get_finished_spans()

    @staticmethod
    def spans_of_kind(spans, kind):
        return [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == kind
        ]

    def test_native_tool_call_creates_tool_span(self):
        """A native tool call gets one TOOL span under the AGENT span."""
        self.assertTrue(self.llm.supports_function_calling())

        spans = self.kickoff(
            [WeatherTool()],
            chat_completion(
                tool_calls=[
                    tool_call(
                        "call_1", "get_weather", {"location": "San Francisco"}
                    )
                ]
            ),
        )

        tool_spans = self.spans_of_kind(spans, "TOOL")
        self.assertEqual(len(tool_spans), 1)
        tool_span = tool_spans[0]
        self.assertEqual(tool_span.name, "execute_tool get_weather")
        self.assertEqual(
            tool_span.attributes.get("gen_ai.operation.name"), "execute_tool"
        )
        self.assertEqual(
            tool_span.attributes.get("gen_ai.crewai.operation"),
            "tool.execute",
        )
        self.assertEqual(
            tool_span.attributes.get("gen_ai.tool.name"), "get_weather"
        )
        self.assertIn(
            "San Francisco",
            tool_span.attributes.get("gen_ai.tool.call.arguments"),
        )
        self.assertIn(
            "Sunny", tool_span.attributes.get("gen_ai.tool.call.result")
        )
        self.assertEqual(tool_span.status.status_code, StatusCode.OK)

        (agent_span,) = self.spans_of_kind(spans, "AGENT")
        self.assertEqual(tool_span.parent.span_id, agent_span.context.span_id)

    def test_failing_native_tool_call_creates_error_span(self):
        """A native tool call that raises gets a TOOL span with ERROR status."""
        spans = self.kickoff(
            [FailingTool()],
            chat_completion(
                tool_calls=[
                    tool_call("call_1", "failing_tool", {"input_str": "x"})
                ]
            ),
        )

        tool_spans = self.spans_of_kind(spans, "TOOL")
        self.assertEqual(len(tool_spans), 1)
        tool_span = tool_spans[0]
        self.assertEqual(tool_span.name, "execute_tool failing_tool")
        self.assertEqual(tool_span.status.status_code, StatusCode.ERROR)
        self.assertIn(
            "Tool execution failed",
            tool_span.events[0].attributes["exception.message"],
        )

        (agent_span,) = self.spans_of_kind(spans, "AGENT")
        self.assertEqual(tool_span.parent.span_id, agent_span.context.span_id)

    def test_parallel_native_tool_calls_create_one_span_each(self):
        """Tool calls that CrewAI runs in parallel threads keep the parent."""
        spans = self.kickoff(
            [WeatherTool()],
            chat_completion(
                tool_calls=[
                    tool_call("call_1", "get_weather", {"location": "Paris"}),
                    tool_call("call_2", "get_weather", {"location": "Tokyo"}),
                ]
            ),
        )

        tool_spans = self.spans_of_kind(spans, "TOOL")
        self.assertEqual(len(tool_spans), 2)
        results = sorted(
            s.attributes.get("gen_ai.tool.call.result") for s in tool_spans
        )
        self.assertIn("Paris", results[0])
        self.assertIn("Tokyo", results[1])

        (agent_span,) = self.spans_of_kind(spans, "AGENT")
        for tool_span in tool_spans:
            self.assertEqual(
                tool_span.parent.span_id, agent_span.context.span_id
            )

    def test_tool_run_inside_traced_tool_use_is_not_traced_again(self):
        """``run`` inside a traced ``ToolUsage._use`` of that tool: one span."""
        tool = WeatherTool()
        wrapper = _ToolUseWrapper(
            ExtendedTelemetryHandler(tracer_provider=self.tracer_provider)
        )

        result = wrapper(
            lambda *args, **kwargs: tool.run(location="Paris"),
            SimpleNamespace(),
            (
                "get_weather",
                tool.to_structured_tool(),
                SimpleNamespace(
                    tool_name="get_weather", arguments={"location": "Paris"}
                ),
            ),
            {},
        )

        self.assertIn("Paris", result)
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(self.spans_of_kind(spans, "TOOL")), 1)

    def test_uninstrument_unwraps_tool_run(self):
        """Uninstrumenting restores ``BaseTool.run`` and ``Tool.run``."""
        self.assertTrue(hasattr(BaseTool.run, "__wrapped__"))

        self.instrumentor.uninstrument()

        self.assertFalse(hasattr(BaseTool.run, "__wrapped__"))
        self.assertFalse(hasattr(Tool.run, "__wrapped__"))
        WeatherTool().run(location="Paris")
        self.assertEqual(len(self.memory_exporter.get_finished_spans()), 0)
