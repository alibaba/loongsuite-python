#!/usr/bin/env python3
"""Run the text-only e-commerce customer service example."""

import argparse
import os
import sys
from collections.abc import Iterator

from langchain_openai import ChatOpenAI
from workflow import EcommerceSupportWorkflow

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_model() -> ChatOpenAI:
    """Create an OpenAI-compatible chat model from environment variables."""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running the example."
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("MODEL_NAME", "qwen-plus"),
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def questions(initial_question: str | None) -> Iterator[str]:
    """Yield one command-line question or questions from an interactive loop."""
    if initial_question:
        yield initial_question
        return

    print("Enter a question. Submit an empty line to exit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        yield question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-only LangGraph e-commerce customer service example"
    )
    parser.add_argument(
        "--question",
        help="Ask one question and exit. Omit it to use the interactive loop.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        support = EcommerceSupportWorkflow(create_model())
    except (RuntimeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    for question in questions(args.question):
        try:
            result = support.run(question)
            print(f"[{result['route']}] {result['final_response']}")
        except KeyboardInterrupt:
            print()
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
