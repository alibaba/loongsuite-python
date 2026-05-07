"""Test configuration for WildToolBench instrumentation tests."""

import json
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test_key_not_real")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:9999/v1")

from opentelemetry.instrumentation.wildtool import WildToolInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrument(tracer_provider):
    instrumentor = WildToolInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()


# ==================== Minimal test data fixtures ====================


def _make_chat_completion_response(
    content=None,
    tool_calls=None,
    input_tokens=10,
    output_tokens=5,
    model="gpt-4o",
):
    """Build a minimal ChatCompletion-like dict that can be JSON-serialized."""
    message = {"role": "assistant", "content": content or ""}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


class FakeChatCompletion:
    """Mimics openai.types.chat.ChatCompletion enough for _parse_api_response."""

    def __init__(self, data: dict):
        self._data = data

    def json(self):
        return json.dumps(self._data)

    def __getattr__(self, name):
        return self._data[name]


@pytest.fixture()
def make_completion():
    """Factory fixture to build FakeChatCompletion objects."""

    def _factory(**kwargs):
        return FakeChatCompletion(_make_chat_completion_response(**kwargs))

    return _factory


@pytest.fixture()
def simple_test_entry():
    """A minimal WildToolBench test_entry with 1 task, 1 step (prepare_to_answer)."""
    return {
        "id": "wild_tool_bench_test_001",
        "english_env_info": "2025-01-01",
        "english_tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "english_tasks": ["What is the weather in Beijing?"],
        "english_answer_list": [
            [
                {
                    "action": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                    "observation": "Sunny, 25°C",
                    "dependency_list": [],
                },
                {
                    "action": {
                        "name": "prepare_to_answer",
                        "arguments": {},
                    },
                    "observation": "The weather in Beijing is Sunny, 25°C",
                    "dependency_list": [0],
                },
            ]
        ],
    }


@pytest.fixture()
def tool_call_response_factory():
    """Factory to make tool_call ChatCompletion responses."""

    def _factory(tool_name, arguments, tool_call_id="call_001"):
        tc = [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": (
                        json.dumps(arguments)
                        if isinstance(arguments, dict)
                        else arguments
                    ),
                },
            }
        ]
        return FakeChatCompletion(
            _make_chat_completion_response(tool_calls=tc)
        )

    return _factory


@pytest.fixture()
def text_response_factory():
    """Factory to make text-only ChatCompletion responses."""

    def _factory(content, input_tokens=10, output_tokens=5):
        return FakeChatCompletion(
            _make_chat_completion_response(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    return _factory
