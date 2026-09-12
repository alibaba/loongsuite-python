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

"""Real LiteLLM smoke traffic for LoongSuite GenAI telemetry.

Run this under ``loongsuite-instrument`` with OTLP configured. The script
exercises non-streaming, streaming, and concurrent async completion calls.
"""

from __future__ import annotations

import asyncio
import os

import litellm

MODEL = os.getenv("LITELLM_MODEL", "qwen-turbo")
API_BASE = os.getenv(
    "LITELLM_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
CUSTOM_PROVIDER = os.getenv("LITELLM_CUSTOM_LLM_PROVIDER", "openai")


def _configure_provider() -> None:
    litellm.telemetry = False


def _provider_kwargs() -> dict[str, str]:
    api_key = (
        os.getenv("LITELLM_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "Missing required API key: set LITELLM_API_KEY, "
            "DASHSCOPE_API_KEY, or OPENAI_API_KEY"
        )

    return {
        "custom_llm_provider": CUSTOM_PROVIDER,
        "api_key": api_key,
        "api_base": API_BASE,
    }


def run_non_streaming() -> None:
    response = litellm.completion(
        model=MODEL,
        **_provider_kwargs(),
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly one short sentence.",
            }
        ],
        temperature=0.1,
        max_tokens=64,
    )
    print("non_streaming:", response.choices[0].message.content[:80])


def run_streaming() -> None:
    stream = litellm.completion(
        model=MODEL,
        **_provider_kwargs(),
        messages=[
            {
                "role": "user",
                "content": "Count from one to five, separated by commas.",
            }
        ],
        stream=True,
        temperature=0.1,
        max_tokens=64,
    )

    chunks = []
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                chunks.append(delta.content)
    print("streaming:", "".join(chunks)[:80])


async def run_concurrent() -> None:
    prompts = [
        "Give one word for sky color.",
        "Give one word for ocean color.",
        "Give one word for grass color.",
    ]

    async def call(prompt: str):
        return await litellm.acompletion(
            model=MODEL,
            **_provider_kwargs(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=32,
        )

    responses = await asyncio.gather(*(call(prompt) for prompt in prompts))
    print(
        "concurrent:",
        ", ".join(
            response.choices[0].message.content[:24] for response in responses
        ),
    )


def main() -> None:
    _configure_provider()
    mode = os.getenv("LITELLM_SMOKE_MODE", "all").lower()

    if mode in ("all", "non_streaming"):
        run_non_streaming()
    if mode in ("all", "streaming"):
        run_streaming()
    if mode in ("all", "concurrent"):
        asyncio.run(run_concurrent())


if __name__ == "__main__":
    main()
