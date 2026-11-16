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

import contextvars
import json
import os
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import respx
from crewai import LLM, Agent, Crew, Task
from crewai.agents import step_executor
from crewai.agents.agent_executor import AgentExecutor
from crewai.tools import tool
from crewai.tools.base_tool import BaseTool, Tool
from crewai.tools.tool_calling import ToolCalling
from crewai.tools.tool_failure import ToolFailure
from crewai.tools.tool_usage import ToolUsage
from pydantic import BaseModel

from opentelemetry import trace
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


class ForecastTool(BaseTool):
    """Sample tool that reports a failure instead of raising."""

    name: str = "get_forecast"
    description: str = "Get the weather forecast for a location"

    def _run(self, location: str) -> ToolFailure:
        """Execute the tool."""
        return ToolFailure(
            message=f"No forecast for {location}", code="unknown_location"
        )


class Forecast(BaseModel):
    """Structured tool result."""

    location: str
    summary: str


class StructuredForecastTool(BaseTool):
    """Sample tool that returns a pydantic model."""

    name: str = "get_structured_forecast"
    description: str = "Get a structured weather forecast for a location"

    def _run(self, location: str) -> Forecast:
        """Execute the tool."""
        return Forecast(location=location, summary="Sunny")


class AccountTool(BaseTool):
    """Sample tool whose output is rewritten before the agent gets it."""

    name: str = "get_account"
    description: str = "Get the account of a customer"

    def _run(self, customer: str) -> str:
        """Execute the tool."""
        return f"{customer} pays with card 4111-1111-1111-1111"

    def format_output_for_agent(self, raw_result: Any) -> str:
        """Mask the card number."""
        masked = str(raw_result).replace("4111-1111-1111-", "****-****-****-")
        return f"Masked: {masked}"


class ReportTool(BaseTool):
    """Sample tool whose formatter returns data, which crewAI makes text."""

    name: str = "get_report"
    description: str = "Get the sales report of a quarter"

    def _run(self, quarter: str) -> dict:
        """Execute the tool."""
        return {"quarter": quarter, "deals": 12}

    def format_output_for_agent(self, raw_result: Any) -> Any:
        """Add the report date."""
        return {**raw_result, "as_of": date(2024, 3, 31)}


class MeetingTool(BaseTool):
    """Sample tool that takes values JSON can't encode as they are."""

    name: str = "book_meeting"
    description: str = "Book a meeting on a day"

    def _run(self, day: date, attendees: Any = None) -> str:
        """Execute the tool."""
        return f"Booked on {day.isoformat()}"


class FullyBookedTool(BaseTool):
    """Sample tool that fails for a day it is given."""

    name: str = "book_room"
    description: str = "Book a room on a day"

    def _run(self, day: date) -> str:
        """Execute the tool."""
        raise RuntimeError(f"No room left on {day.isoformat()}")


@tool("lookup_capital")
def lookup_capital(country: str) -> str:
    """Return the capital city of a country."""
    return {"France": "Paris"}.get(country, "unknown")


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

    def kickoff(self, tools, *responses, llm=None):
        """Run a one-task crew; the LLM answers with ``responses``."""
        replies = iter(responses)

        def respond(request):
            body = next(replies, chat_completion(content="All tools used."))
            return httpx.Response(200, json=body)

        agent = Agent(
            role="Tool User",
            goal="Use the tools",
            backstory="Agent that uses tools",
            verbose=False,
            llm=llm or self.llm,
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

    def react_llm(self):
        """An LLM without function calling: tools go through ``ToolUsage``."""
        llm = LLM(
            model="openai/qwen-turbo",
            base_url=BASE_URL,
            api_key="fake-key",
            is_litellm=True,
        )
        self.assertFalse(llm.supports_function_calling())
        return llm

    @staticmethod
    def spans_of_kind(spans, kind):
        return [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == kind
        ]

    def finished_tool_spans(self):
        return self.spans_of_kind(
            self.memory_exporter.get_finished_spans(), "TOOL"
        )

    @staticmethod
    def call_in_new_context(function, *args, **kwargs):
        """Call ``function`` in a copy of the current context.

        Returns its result and the span current there after the call, so a
        span left current cannot leak into other tests.
        """

        def call():
            return function(*args, **kwargs), trace.get_current_span()

        return contextvars.copy_context().run(call)

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
        self.assertEqual(
            tool_span.attributes.get("gen_ai.tool.call.id"), "call_1"
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
        """Parallel tool calls keep the parent and the model's call ids."""
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
        call_ids = {
            s.attributes.get("gen_ai.tool.call.id"): s.attributes.get(
                "gen_ai.tool.call.result"
            )
            for s in tool_spans
        }
        self.assertEqual(set(call_ids), {"call_1", "call_2"})
        self.assertIn("Paris", call_ids["call_1"])
        self.assertIn("Tokyo", call_ids["call_2"])

        (agent_span,) = self.spans_of_kind(spans, "AGENT")
        for tool_span in tool_spans:
            self.assertEqual(
                tool_span.parent.span_id, agent_span.context.span_id
            )

    def test_native_call_of_decorated_tool_creates_one_span(self):
        """A ``@tool`` function, whose ``Tool`` has its own ``run``."""
        spans = self.kickoff(
            [lookup_capital],
            chat_completion(
                tool_calls=[
                    tool_call(
                        "call_1", "lookup_capital", {"country": "France"}
                    )
                ]
            ),
        )

        tool_spans = self.spans_of_kind(spans, "TOOL")
        self.assertEqual(len(tool_spans), 1)
        tool_span = tool_spans[0]
        self.assertEqual(tool_span.name, "execute_tool lookup_capital")
        self.assertEqual(
            tool_span.attributes.get("gen_ai.tool.call.id"), "call_1"
        )
        self.assertEqual(
            tool_span.attributes.get("gen_ai.tool.call.result"), "Paris"
        )

    def test_native_tool_returning_tool_failure_creates_error_span(self):
        """A ``ToolFailure`` returned instead of raised marks the span ERROR."""
        spans = self.kickoff(
            [ForecastTool()],
            chat_completion(
                tool_calls=[
                    tool_call(
                        "call_1", "get_forecast", {"location": "Atlantis"}
                    )
                ]
            ),
        )

        (tool_span,) = self.spans_of_kind(spans, "TOOL")
        self.assert_tool_failure_span(tool_span)

    def test_react_tool_returning_tool_failure_creates_same_error_span(self):
        """The ReAct path (``ToolUsage._use``) reports it the same way."""
        spans = self.kickoff(
            [ForecastTool()],
            chat_completion(
                content=(
                    "Thought: I need the forecast.\n"
                    "Action: get_forecast\n"
                    'Action Input: {"location": "Atlantis"}'
                )
            ),
            chat_completion(
                content=(
                    "Thought: I now know the final answer\n"
                    "Final Answer: There is no forecast for Atlantis."
                )
            ),
            llm=self.react_llm(),
        )

        (tool_span,) = self.spans_of_kind(spans, "TOOL")
        self.assert_tool_failure_span(tool_span)

    def assert_tool_failure_span(self, tool_span):
        message = "No forecast for Atlantis (code: unknown_location)"
        self.assertEqual(tool_span.name, "execute_tool get_forecast")
        self.assertEqual(tool_span.status.status_code, StatusCode.ERROR)
        self.assertEqual(tool_span.status.description, message)
        self.assertEqual(
            tool_span.attributes.get("error.type"), "RuntimeError"
        )
        self.assertEqual(
            tool_span.attributes.get("gen_ai.tool.call.result"), message
        )
        self.assertEqual(len(tool_span.events), 0)

    def test_native_tool_string_result_is_recorded_as_the_agent_gets_it(self):
        """A string result goes through the tool's formatter too."""
        spans = self.kickoff(
            [AccountTool()],
            chat_completion(
                tool_calls=[
                    tool_call("call_1", "get_account", {"customer": "Ann"})
                ]
            ),
        )

        (tool_span,) = self.spans_of_kind(spans, "TOOL")
        self.assert_masked_account_span(tool_span)

    def test_react_tool_result_is_not_formatted_again(self):
        """``ToolUsage._use`` returns the text the agent gets, as is."""
        spans = self.kickoff(
            [AccountTool()],
            chat_completion(
                content=(
                    "Thought: I need the account.\n"
                    "Action: get_account\n"
                    'Action Input: {"customer": "Ann"}'
                )
            ),
            chat_completion(
                content=(
                    "Thought: I now know the final answer\n"
                    "Final Answer: Ann pays by card."
                )
            ),
            llm=self.react_llm(),
        )

        (tool_span,) = self.spans_of_kind(spans, "TOOL")
        self.assert_masked_account_span(tool_span)

    def assert_masked_account_span(self, tool_span):
        self.assertEqual(tool_span.name, "execute_tool get_account")
        self.assertEqual(tool_span.status.status_code, StatusCode.OK)
        self.assertEqual(
            tool_span.attributes.get("gen_ai.tool.call.result"),
            "Masked: Ann pays with card ****-****-****-1111",
        )

    def test_native_tool_returning_model_records_text_result(self):
        """A non-JSON result (a pydantic model) still ends the span."""
        spans = self.kickoff(
            [StructuredForecastTool()],
            chat_completion(
                tool_calls=[
                    tool_call(
                        "call_1",
                        "get_structured_forecast",
                        {"location": "Paris"},
                    )
                ]
            ),
        )

        (tool_span,) = self.spans_of_kind(spans, "TOOL")
        self.assertEqual(tool_span.status.status_code, StatusCode.OK)
        # The text the agent gets, as CrewAI formats it.
        self.assertIn(
            "Sunny", tool_span.attributes.get("gen_ai.tool.call.result")
        )

    def test_direct_tool_run_uses_tool_name_as_call_id(self):
        """Outside a native tool call there is no model call id."""
        WeatherTool().run(location="Paris")

        (tool_span,) = self.spans_of_kind(
            self.memory_exporter.get_finished_spans(), "TOOL"
        )
        self.assertEqual(
            tool_span.attributes.get("gen_ai.tool.call.id"), "get_weather"
        )

    def test_tool_run_records_arguments_json_cannot_encode_as_text(self):
        """Positional and keyword arguments keep their structure."""
        day = date(2024, 1, 1)
        for args, kwargs, recorded in (
            ((day,), {}, ["2024-01-01"]),
            ((), {"day": day}, {"day": "2024-01-01"}),
            (
                (day,),
                {"attendees": ("Ann", Decimal("1.5"))},
                {
                    "args": ["2024-01-01"],
                    "kwargs": {"attendees": ["Ann", "1.5"]},
                },
            ),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                self.memory_exporter.clear()

                result, current_span = self.call_in_new_context(
                    MeetingTool().run, *args, **kwargs
                )

                self.assertEqual(result, "Booked on 2024-01-01")
                (tool_span,) = self.finished_tool_spans()
                self.assertEqual(tool_span.status.status_code, StatusCode.OK)
                self.assertEqual(
                    json.loads(
                        tool_span.attributes["gen_ai.tool.call.arguments"]
                    ),
                    recorded,
                )
                self.assertIs(current_span, trace.INVALID_SPAN)

    def test_failing_tool_run_with_a_date_argument_ends_error_span(self):
        """Its ERROR span ends, and the tool's exception reaches the caller."""

        def book_room():
            with self.assertRaises(RuntimeError) as raised:
                FullyBookedTool().run(day=date(2024, 1, 1))
            return raised.exception

        error, current_span = self.call_in_new_context(book_room)

        self.assertEqual(str(error), "No room left on 2024-01-01")
        (tool_span,) = self.finished_tool_spans()
        self.assertEqual(tool_span.status.status_code, StatusCode.ERROR)
        self.assertEqual(
            json.loads(tool_span.attributes["gen_ai.tool.call.arguments"]),
            {"day": "2024-01-01"},
        )
        self.assertIs(current_span, trace.INVALID_SPAN)

    def test_tool_usage_records_arguments_json_cannot_encode_as_text(self):
        """The ReAct path (``ToolUsage._use``) records them the same way."""
        usage = ToolUsage(
            tools_handler=None,
            tools=[MeetingTool().to_structured_tool()],
            task=None,
            function_calling_llm=self.llm,
        )
        calling = ToolCalling(
            tool_name="book_meeting", arguments={"day": date(2024, 1, 1)}
        )

        result, current_span = self.call_in_new_context(
            usage.use, calling, "book_meeting"
        )

        self.assertEqual(result, "Booked on 2024-01-01")
        (tool_span,) = self.finished_tool_spans()
        self.assertEqual(tool_span.status.status_code, StatusCode.OK)
        self.assertEqual(
            json.loads(tool_span.attributes["gen_ai.tool.call.arguments"]),
            {"day": "2024-01-01"},
        )
        self.assertIs(current_span, trace.INVALID_SPAN)

    def test_tool_result_formatted_as_data_is_recorded_as_text(self):
        """A formatter may return data: crewAI gives the agent its text."""
        result, current_span = self.call_in_new_context(
            ReportTool().run, quarter="Q1"
        )

        self.assertEqual(result, {"quarter": "Q1", "deals": 12})
        (tool_span,) = self.finished_tool_spans()
        self.assertEqual(tool_span.status.status_code, StatusCode.OK)
        self.assertEqual(
            tool_span.attributes["gen_ai.tool.call.result"],
            "{'quarter': 'Q1', 'deals': 12, "
            "'as_of': datetime.date(2024, 3, 31)}",
        )
        self.assertIs(current_span, trace.INVALID_SPAN)

    def test_tool_result_is_not_formatted_without_content_capture(self):
        """The span records no result, so the formatter is not run for it."""
        no_content = {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT"
        }
        with (
            patch.dict(os.environ, no_content),
            patch.object(
                StructuredForecastTool, "format_output_for_agent"
            ) as format_output,
        ):
            StructuredForecastTool().run(location="Paris")

        format_output.assert_not_called()
        (tool_span,) = self.finished_tool_spans()
        self.assertNotIn("gen_ai.tool.call.result", tool_span.attributes)

    def test_tool_run_inside_traced_tool_use_is_not_traced_again(self):
        """``run`` inside a traced ``ToolUsage._use`` of that tool: one span."""
        weather_tool = WeatherTool()
        wrapper = _ToolUseWrapper(
            ExtendedTelemetryHandler(tracer_provider=self.tracer_provider)
        )

        result = wrapper(
            lambda *args, **kwargs: weather_tool.run(location="Paris"),
            SimpleNamespace(),
            (
                "get_weather",
                weather_tool.to_structured_tool(),
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
        """Uninstrumenting restores the tools' ``run`` and native dispatch."""
        wrapped = (
            BaseTool.run,
            Tool.run,
            AgentExecutor._execute_single_native_tool_call,
            step_executor.execute_single_native_tool_call,
        )
        for function in wrapped:
            self.assertTrue(hasattr(function, "__wrapped__"))

        self.instrumentor.uninstrument()

        unwrapped = (
            BaseTool.run,
            Tool.run,
            AgentExecutor._execute_single_native_tool_call,
            step_executor.execute_single_native_tool_call,
        )
        for function in unwrapped:
            self.assertFalse(hasattr(function, "__wrapped__"))
        WeatherTool().run(location="Paris")
        self.assertEqual(len(self.memory_exporter.get_finished_spans()), 0)
