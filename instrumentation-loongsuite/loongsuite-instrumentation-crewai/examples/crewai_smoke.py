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

"""Real CrewAI smoke scenarios for LoongSuite GenAI telemetry."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from typing import Any

import crewai
from crewai import Agent, Crew, Task
from crewai.tools.base_tool import BaseTool

from opentelemetry import trace
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

DEFAULT_DASHSCOPE_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
_TRACER_PROVIDER = None


class WordCountTool(BaseTool):
    """Small deterministic tool to make TOOL spans easy to verify."""

    name: str = "word_count"
    description: str = "Count words in a short text string."

    def _run(self, text: str) -> str:
        words = [part for part in str(text).split() if part]
        return f"word_count={len(words)}"


def _api_key() -> str:
    value = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not value:
        raise RuntimeError(
            "Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running "
            "the CrewAI smoke example."
        )
    return value


def _llm(streaming: bool) -> Any:
    api_key = _api_key()
    base_url = os.getenv("DASHSCOPE_API_BASE") or os.getenv(
        "OPENAI_API_BASE", DEFAULT_DASHSCOPE_BASE_URL
    )
    model = os.getenv("CREWAI_SMOKE_MODEL", "dashscope/qwen-turbo")

    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_API_BASE", base_url)
    os.environ.setdefault("DASHSCOPE_API_BASE", base_url)
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    llm_cls = getattr(crewai, "LLM", None)
    if llm_cls is None:
        return model

    try:
        return llm_cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            stream=streaming,
            temperature=0,
        )
    except TypeError:
        return llm_cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )


def _consume_result(result: Any) -> str:
    if not isinstance(result, str) and hasattr(result, "__iter__"):
        chunks = []
        for chunk in result:
            content = getattr(chunk, "content", chunk)
            if content is not None:
                chunks.append(str(content))
        final_result = getattr(result, "result", None)
        if final_result is not None:
            return str(getattr(final_result, "raw", final_result))
        return "".join(chunks)

    return str(getattr(result, "raw", result))


def run_crew(run_id: int, *, streaming: bool) -> str:
    word_count = WordCountTool()
    agent = Agent(
        role=f"CrewAI Telemetry Analyst {run_id}",
        goal="Use tools and write concise telemetry validation summaries.",
        backstory="You are validating LoongSuite CrewAI OpenTelemetry spans.",
        verbose=False,
        llm=_llm(streaming),
        tools=[word_count],
    )
    task = Task(
        description=(
            "Use the word_count tool exactly once on the text "
            f"'CrewAI telemetry smoke run {run_id}', then answer with the "
            "tool result and one short sentence."
        ),
        expected_output="The tool result plus one short sentence.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False, stream=streaming)
    result = crew.kickoff(
        inputs={
            "session_id": f"crewai-smoke-{run_id}",
            "user_id": "loongsuite-smoke",
            "streaming": streaming,
        }
    )
    return _consume_result(result)


def _maybe_manual_instrument() -> None:
    if os.getenv("CREWAI_SMOKE_MANUAL_INSTRUMENT", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    tracer_provider = _maybe_configure_otlp()
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)


def _maybe_configure_otlp() -> Any:
    global _TRACER_PROVIDER  # pylint: disable=global-statement

    if os.getenv("CREWAI_SMOKE_CONFIGURE_OTLP", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None

    if _TRACER_PROVIDER is not None:
        return _TRACER_PROVIDER

    try:
        exporter_module = import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install opentelemetry-exporter-otlp-proto-http to use "
            "CREWAI_SMOKE_CONFIGURE_OTLP=true, or unset "
            "CREWAI_SMOKE_CONFIGURE_OTLP for the default smoke run."
        ) from exc

    resource = Resource.create(
        {"service.name": os.getenv("OTEL_SERVICE_NAME", "crewai-smoke")}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(exporter_module.OTLPSpanExporter())
    )
    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER = provider
    return provider


def _flush_traces() -> None:
    if _TRACER_PROVIDER is not None:
        _TRACER_PROVIDER.force_flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("sync", "stream", "concurrent"),
        default="sync",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    _maybe_manual_instrument()

    if args.mode == "concurrent":
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(run_crew, index, streaming=index % 2 == 0)
                for index in range(args.concurrency)
            ]
            for future in as_completed(futures):
                print(future.result())
        _flush_traces()
        return

    print(run_crew(0, streaming=args.mode == "stream"))
    _flush_traces()


if __name__ == "__main__":
    main()
