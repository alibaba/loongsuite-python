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

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from agno.agent import Agent
from agno.models.dashscope import DashScope


def get_weather(city: str) -> str:
    """Return a tiny deterministic weather summary."""
    return f"{city}: sunny, 24C"


def build_agent(name: str = "AgnoDashScopeSmoke") -> Agent:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required")

    model = DashScope(
        id=os.environ.get("DASHSCOPE_MODEL", "qwen-plus"),
        api_key=api_key,
        base_url=os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    return Agent(
        name=name,
        model=model,
        tools=[get_weather],
        instructions=[
            "Answer concisely.",
            "When a weather question is asked, use the get_weather tool.",
        ],
    )


def run_non_stream() -> None:
    agent = build_agent("AgnoDashScopeNonStream")
    response = agent.run("What is the weather in Hangzhou?")
    print("non_stream:", response.content)


def run_stream() -> None:
    agent = build_agent("AgnoDashScopeStream")
    print("stream:", end=" ", flush=True)
    for event in agent.run(
        "Stream a one sentence weather answer for Hangzhou.",
        stream=True,
        stream_events=True,
    ):
        content = getattr(event, "content", None)
        if content:
            print(content, end="", flush=True)
    print()


def run_concurrent() -> None:
    def call(index: int) -> str:
        agent = build_agent(f"AgnoDashScopeConcurrent{index}")
        response = agent.run(f"Reply with the word pong and index {index}.")
        return str(response.content)

    with ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(call, range(3)):
            print("concurrent:", result)


def main() -> None:
    mode = os.environ.get("AGNO_SMOKE_MODE", "all")
    if mode in ("all", "non_stream"):
        run_non_stream()
    if mode in ("all", "stream"):
        run_stream()
    if mode in ("all", "concurrent"):
        run_concurrent()


if __name__ == "__main__":
    main()
