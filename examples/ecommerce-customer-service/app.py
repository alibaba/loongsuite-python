#!/usr/bin/env python3
"""运行纯文字中文电商客服示例。"""

import argparse
import os
import sys
from collections.abc import Iterator

from langchain_openai import ChatOpenAI
from workflow import EcommerceSupportWorkflow

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ROUTE_LABELS = {
    "presales": "售前",
    "aftersales": "售后",
    "clarify": "需澄清",
}


def create_model() -> ChatOpenAI:
    """Create an OpenAI-compatible chat model from environment variables."""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "运行示例前请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。"
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

    print("请输入问题，直接回车即可退出。")
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
        description="纯文字 LangGraph 中文电商客服示例"
    )
    parser.add_argument(
        "--question",
        help="提问一次后退出；不传该参数则进入交互模式。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        support = EcommerceSupportWorkflow(create_model())
    except (RuntimeError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    for question in questions(args.question):
        try:
            result = support.run(question)
            route_label = ROUTE_LABELS.get(result["route"], result["route"])
            print(f"[{route_label}] {result['final_response']}")
        except ValueError as exc:
            print(f"请求错误：{exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print()
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
