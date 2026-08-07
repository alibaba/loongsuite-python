"""Offline tests for the text-only e-commerce support workflow."""

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import app
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from tools import (
    check_inventory,
    lookup_order_history,
    query_after_sales_policy,
    search_product_catalog,
)
from workflow import EcommerceSupportWorkflow, IntentDecision, select_route


class KeywordClassifier:
    def invoke(self, messages: list[Any]) -> IntentDecision:
        question = messages[-1].content.casefold()
        if "订单" in question or "退款" in question:
            return IntentDecision(
                intent="aftersales", confidence=0.95, reason="已有订单"
            )
        if "鞋" in question or "库存" in question or "有货" in question:
            return IntentDecision(
                intent="presales", confidence=0.95, reason="商品咨询"
            )
        return IntentDecision(
            intent="other", confidence=0.4, reason="意图不明确"
        )


class FixedClassifier:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    def invoke(self, messages: list[Any]) -> IntentDecision:
        del messages
        return IntentDecision(
            intent=self.intent, confidence=0.99, reason="测试路由"
        )


class DeterministicToolModel(BaseChatModel):
    """Minimal real LangChain model that drives a LangGraph ReAct tool call."""

    tool_name: str
    tool_args: dict[str, Any]

    @property
    def _llm_type(self) -> str:
        return "deterministic-tool-model"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        del tools, tool_choice, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        is_review = any(
            isinstance(message, SystemMessage)
            and "最终回复审核员" in str(message.content)
            for message in messages
        )
        if is_review:
            response = AIMessage(content="已审核的确定性回复")
        elif any(isinstance(message, ToolMessage) for message in messages):
            response = AIMessage(content="确定性的专业客服回复")
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_name,
                        "args": self.tool_args,
                        "id": "deterministic-call",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])


class FakeModel:
    def __init__(self, *, fail_review: bool = False) -> None:
        self.fail_review = fail_review
        self.review_inputs: list[str] = []

    def with_structured_output(self, schema: Any) -> KeywordClassifier:
        assert schema is IntentDecision
        return KeywordClassifier()

    def invoke(self, messages: list[Any]) -> AIMessage:
        if self.fail_review:
            raise RuntimeError("虚构审核员异常")
        review_input = messages[-1].content
        self.review_inputs.append(review_input)
        return AIMessage(content=f"已审核：{review_input}")


class FakeAgent:
    def __init__(
        self, name: str, calls: list[str], *, fail: bool = False
    ) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail

    def invoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError("虚构专业客服异常")
        question = input_data["messages"][-1].content
        return {
            "messages": [
                ToolMessage(
                    content=f'{{"source": "{self.name}", "ok": true}}',
                    tool_call_id=f"call-{self.name}",
                    name=f"{self.name}_tool",
                ),
                AIMessage(content=f"{self.name} 针对“{question}”的回复草稿"),
            ]
        }


def make_workflow(
    *, fail_review: bool = False, fail_agent: str | None = None
) -> tuple[EcommerceSupportWorkflow, FakeModel, list[str]]:
    calls: list[str] = []
    model = FakeModel(fail_review=fail_review)

    def factory(**kwargs: Any) -> FakeAgent:
        name = kwargs["name"]
        return FakeAgent(name, calls, fail=name == fail_agent)

    return EcommerceSupportWorkflow(model, agent_factory=factory), model, calls


def test_synthetic_tools_return_bounded_results() -> None:
    assert '"P-DEMO-001"' in search_product_catalog.invoke({"query": "通勤鞋"})
    inventory = check_inventory.invoke(
        {"product_id": "P-DEMO-001", "size": "42"}
    )
    assert '"available": true' in inventory
    assert '"quantity": 4' in inventory
    assert '"window_days": 30' in query_after_sales_policy.invoke(
        {"issue_type": "quality_issue"}
    )


def test_missing_order_is_a_tool_result_not_an_exception() -> None:
    result = lookup_order_history.invoke({"order_id": "DEMO-404"})
    assert json.loads(result) == {
        "error": "未找到订单：DEMO-404",
        "ok": False,
    }


def test_cli_reports_a_chinese_error_for_a_blank_question(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class EmptyQuestionWorkflow:
        def run(self, question: str) -> dict[str, Any]:
            del question
            raise ValueError("问题不能为空")

    monkeypatch.setattr(
        app,
        "parse_args",
        lambda: argparse.Namespace(question="  "),
    )
    monkeypatch.setattr(app, "create_model", lambda: object())
    monkeypatch.setattr(
        app,
        "EcommerceSupportWorkflow",
        lambda model: EmptyQuestionWorkflow(),
    )

    assert app.main() == 2
    assert capsys.readouterr().err == "请求错误：问题不能为空\n"


def test_route_selection_uses_intent_and_confidence() -> None:
    assert (
        select_route({"intent": "presales", "confidence": 0.9}) == "presales"
    )
    assert (
        select_route({"intent": "aftersales", "confidence": 0.9})
        == "aftersales"
    )
    assert select_route({"intent": "presales", "confidence": 0.2}) == "clarify"
    assert select_route({"intent": "other", "confidence": 1.0}) == "clarify"


def test_presales_and_aftersales_use_different_agents() -> None:
    workflow, model, calls = make_workflow()
    presales = workflow.run("云步通勤鞋 42 码有货吗？")
    aftersales = workflow.run("订单 DEMO-1001 的鞋有质量问题，可以退款吗？")

    assert presales["route"] == "presales"
    assert aftersales["route"] == "aftersales"
    assert calls == ["presales_agent", "aftersales_agent"]
    assert "presales_agent_tool" in model.review_inputs[0]
    assert "aftersales_agent_tool" in model.review_inputs[1]


@pytest.mark.parametrize(
    ("intent", "tool_name", "tool_args", "question", "evidence"),
    [
        (
            "presales",
            "check_inventory",
            {"product_id": "P-DEMO-001", "size": "42"},
            "云步通勤鞋 42 码有货吗？",
            '"quantity": 4',
        ),
        (
            "aftersales",
            "lookup_order_history",
            {"order_id": "DEMO-1001"},
            "订单 DEMO-1001 现在是什么状态？",
            '"status": "已签收"',
        ),
    ],
)
def test_real_langgraph_react_agent_executes_a_synthetic_tool(
    intent: str,
    tool_name: str,
    tool_args: dict[str, Any],
    question: str,
    evidence: str,
) -> None:
    model = DeterministicToolModel(
        tool_name=tool_name,
        tool_args=tool_args,
    )
    workflow = EcommerceSupportWorkflow(
        model,
        classifier=FixedClassifier(intent),
    )

    result = workflow.run(question)

    assert result["route"] == intent
    assert result["final_response"] == "已审核的确定性回复"
    assert any(evidence in item for item in result["tool_evidence"])


def test_ambiguous_question_uses_clarification_without_a_specialist() -> None:
    workflow, _, calls = make_workflow()
    result = workflow.run("能帮帮我吗？")
    assert result["route"] == "clarify"
    assert calls == []
    assert "请问你想咨询" in result["final_response"]


def test_specialist_and_reviewer_fail_open() -> None:
    workflow, _, calls = make_workflow(
        fail_review=True, fail_agent="presales_agent"
    )
    result = workflow.run("云步通勤鞋有货吗？")
    assert calls == ["presales_agent"]
    assert result["route"] == "presales"
    assert result["final_response"].startswith("售前客服暂时不可用")


def test_concurrent_invocations_keep_routes_and_answers_isolated() -> None:
    workflow, _, _ = make_workflow()
    questions = [
        "云步通勤鞋有货吗？",
        "订单 DEMO-1002 到哪里了？",
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(workflow.run, questions))

    assert [result["route"] for result in results] == [
        "presales",
        "aftersales",
    ]
    assert questions[0] in results[0]["final_response"]
    assert questions[1] in results[1]["final_response"]
