#!/usr/bin/env python3

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

"""Google ADK smoke scenarios for local otel-gui verification.

Run this script through loongsuite-instrument so spans are exported to otel-gui:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:5173 \
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
    OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental \
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY \
    OTEL_SERVICE_NAME=loongsuite-google-adk-smoke \
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:5173/v1/traces \
    GOOGLE_ADK_SMOKE_CONFIGURE_OTLP=1 python examples/otelgui_smoke.py --scenario all
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from collections.abc import Iterable

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)
from google.adk.tools import FunctionTool
from google.genai import types

from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor


def _configure_otlp_exporter_from_env():
    """Configure an OTLP HTTP exporter for standalone smoke runs."""
    enabled = os.getenv("GOOGLE_ADK_SMOKE_CONFIGURE_OTLP", "").lower()
    if enabled not in ("1", "true", "yes"):
        return None

    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            BatchSpanProcessor,
        )
    except ImportError as exc:
        raise SystemExit(
            "GOOGLE_ADK_SMOKE_CONFIGURE_OTLP=1 requires "
            "opentelemetry-exporter-otlp-proto-http"
        ) from exc

    resource = Resource.create(
        {
            "service.name": os.getenv(
                "OTEL_SERVICE_NAME", "loongsuite-google-adk-smoke"
            )
        }
    )
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    exporter = (
        OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def _disable_native_agent_span_for_smoke() -> None:
    """Optionally suppress ADK's native agent wrapper span during smoke tests.

    This private ADK monkey-patch is only for otel-gui smoke validation: it
    removes ADK's native wrapper span so the LoongSuite GenAI span tree is
    easier to inspect. It may need adjustment when ADK changes internals.
    """
    enabled = os.getenv(
        "GOOGLE_ADK_SMOKE_DISABLE_NATIVE_AGENT_SPAN", ""
    ).lower()
    if enabled not in ("1", "true", "yes"):
        return

    from google.adk.telemetry import _instrumentation  # noqa: PLC0415

    import opentelemetry.context as context_api  # noqa: PLC0415

    @contextlib.asynccontextmanager
    async def _record_agent_invocation(ctx, agent):
        token = context_api.attach(context_api.Context())
        try:
            yield _instrumentation.TelemetryContext(
                otel_context=context_api.get_current()
            )
        finally:
            context_api.detach(token)

    _instrumentation.record_agent_invocation = _record_agent_invocation


def get_city_weather(city: str) -> str:
    """Return deterministic weather text so the model can exercise a tool."""
    return f"{city}: sunny, 24C, light wind"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _extract_event_text(event) -> str:
    content = getattr(event, "content", None)
    if not content:
        return ""
    parts = getattr(content, "parts", None) or []
    return "".join(
        getattr(part, "text", "") or "" for part in parts if part is not None
    )


def _last_non_empty_text(events: Iterable[object]) -> str:
    text = ""
    for event in events:
        event_text = _extract_event_text(event)
        if event_text:
            text = event_text
    return text


async def _create_runner() -> tuple[Runner, InMemorySessionService]:
    api_key = _require_env("DASHSCOPE_API_KEY")
    model = LiteLlm(
        model=os.getenv("DASHSCOPE_MODEL", "dashscope/qwen-plus"),
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=float(os.getenv("DASHSCOPE_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("DASHSCOPE_MAX_TOKENS", "256")),
    )
    weather_tool = FunctionTool(func=get_city_weather)
    agent = LlmAgent(
        name="google_adk_smoke_agent",
        model=model,
        instruction=(
            "You are a concise assistant. Use tools when a prompt asks for "
            "weather, then answer with one short sentence."
        ),
        description="Google ADK instrumentation smoke-test agent",
        tools=[weather_tool],
    )
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="google_adk_smoke",
        agent=agent,
        session_service=session_service,
    )
    return runner, session_service


async def _run_once(
    runner: Runner,
    session_service: InMemorySessionService,
    *,
    user_id: str,
    session_id: str,
    prompt: str,
    streaming: bool = False,
) -> str:
    session = await session_service.create_session(
        app_name="google_adk_smoke",
        user_id=user_id,
        session_id=session_id,
    )
    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])
    run_config = (
        RunConfig(streaming_mode=StreamingMode.SSE) if streaming else None
    )
    events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=user_message,
        run_config=run_config,
    ):
        events.append(event)
    return _last_non_empty_text(events)


async def run_non_stream() -> None:
    runner, session_service = await _create_runner()
    response = await _run_once(
        runner,
        session_service,
        user_id="otelgui_user",
        session_id=f"non_stream_{int(time.time() * 1000)}",
        prompt="Use get_city_weather for Hangzhou and summarize it.",
    )
    print(f"non_stream response: {response}")


async def run_stream() -> None:
    runner, session_service = await _create_runner()
    response = await _run_once(
        runner,
        session_service,
        user_id="otelgui_user",
        session_id=f"stream_{int(time.time() * 1000)}",
        prompt="Reply with a short streaming-friendly greeting.",
        streaming=True,
    )
    print(f"stream response: {response}")


async def run_concurrent(count: int) -> None:
    runner, session_service = await _create_runner()
    now_ms = int(time.time() * 1000)
    tasks = [
        _run_once(
            runner,
            session_service,
            user_id=f"otelgui_user_{index}",
            session_id=f"concurrent_{now_ms}_{index}",
            prompt=f"Use get_city_weather for city {index} and summarize it.",
        )
        for index in range(count)
    ]
    responses = await asyncio.gather(*tasks)
    for index, response in enumerate(responses):
        print(f"concurrent response {index}: {response}")


async def _amain(args: argparse.Namespace) -> None:
    tracer_provider = _configure_otlp_exporter_from_env()
    _disable_native_agent_span_for_smoke()
    GoogleAdkInstrumentor().instrument()
    try:
        if args.scenario in ("non-stream", "all"):
            await run_non_stream()
        if args.scenario in ("stream", "all"):
            await run_stream()
        if args.scenario in ("concurrent", "all"):
            await run_concurrent(args.concurrent_count)
    finally:
        if tracer_provider is not None:
            tracer_provider.force_flush()
            tracer_provider.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("non-stream", "stream", "concurrent", "all"),
        default="all",
    )
    parser.add_argument("--concurrent-count", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
