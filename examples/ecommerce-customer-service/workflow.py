"""纯文字电商客服 LangGraph 工作流。"""

import json
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from tools import AFTERSALES_TOOLS, PRESALES_TOOLS

Route = Literal["presales", "aftersales", "clarify"]

PRESALES_PROMPT = """你是虚构电商店铺的售前客服。
只能使用提供的虚构商品目录、商品知识和库存工具。
回答前至少调用一个工具；工具未提供的信息要明确说明不知道。
不得编造价格、促销、到货承诺或商品能力。
请始终使用简体中文回复。
"""

AFTERSALES_PROMPT = """你是虚构电商店铺的售后客服。
只能使用提供的虚构订单、售后政策和问题评估工具。
讨论订单前必须先调用订单查询工具。不得直接承诺退款或换货，
只能说明政策依据和下一步核验流程。
请始终使用简体中文回复。
"""

REVIEW_PROMPT = """你是虚构电商店铺的最终回复审核员。
请把草稿润色成简洁、友善的中文客服回复，只保留工具证据支持的事实。
不得补充证据中没有的价格、库存、政策、订单状态、保证或处理动作。
证据不足时必须保留不确定性。只输出最终给客户看的中文回复。
"""


class IntentDecision(BaseModel):
    """Structured result produced by the intent router."""

    intent: Literal["presales", "aftersales", "other"]
    confidence: float = Field(ge=0, le=1)
    reason: str


class SupportState(TypedDict, total=False):
    """State shared by the outer customer-service graph."""

    question: str
    intent: str
    confidence: float
    route: Route
    draft_response: str
    tool_evidence: list[str]
    final_response: str


class StructuredClassifier(Protocol):
    def invoke(self, input_data: Any) -> IntentDecision | dict[str, Any]: ...


class ChatModel(Protocol):
    def with_structured_output(
        self, schema: type[BaseModel]
    ) -> StructuredClassifier: ...

    def invoke(self, input_data: Any) -> BaseMessage: ...


class AgentRunner(Protocol):
    def invoke(self, input_data: dict[str, Any]) -> dict[str, Any]: ...


AgentFactory = Callable[..., AgentRunner]


def default_agent_factory(
    *, name: str, model: ChatModel, tools: Sequence[BaseTool], prompt: str
) -> AgentRunner:
    """Build a LangGraph ReAct agent while keeping construction injectable for tests."""
    del name  # The outer graph node supplies the observable specialist name.
    return create_react_agent(model=model, tools=list(tools), prompt=prompt)


def select_route(
    state: SupportState, confidence_threshold: float = 0.6
) -> Route:
    """Map the structured decision to an explicit graph branch."""
    if state.get("confidence", 0) < confidence_threshold:
        return "clarify"
    intent = state.get("intent")
    if intent == "presales":
        return "presales"
    if intent == "aftersales":
        return "aftersales"
    return "clarify"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def extract_agent_result(result: dict[str, Any]) -> tuple[str, list[str]]:
    """Extract the final assistant text and tool evidence from an agent result."""
    messages = result.get("messages", [])
    answer = ""
    evidence = []
    for message in messages:
        message_type = getattr(message, "type", "")
        content = _content_to_text(getattr(message, "content", ""))
        if message_type == "tool":
            name = getattr(message, "name", None) or "tool"
            evidence.append(f"{name}: {content}")
        elif message_type == "ai" and content.strip():
            answer = content.strip()
    if not answer:
        raise ValueError("专业客服 Agent 未返回最终答案")
    return answer, evidence


def build_review_input(state: SupportState) -> str:
    """Build a bounded review request from the draft and observed tool results."""
    payload = {
        "route": state.get("route", "clarify"),
        "customer_question": state["question"],
        "draft_response": state.get("draft_response", ""),
        "tool_evidence": state.get("tool_evidence", []),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class EcommerceSupportWorkflow:
    """Route a text question to a specialist and review its response."""

    def __init__(
        self,
        model: ChatModel,
        *,
        classifier: StructuredClassifier | None = None,
        agent_factory: AgentFactory = default_agent_factory,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.model = model
        self.classifier = classifier or model.with_structured_output(
            IntentDecision
        )
        self.confidence_threshold = confidence_threshold
        self.presales_agent = agent_factory(
            name="presales_agent",
            model=model,
            tools=PRESALES_TOOLS,
            prompt=PRESALES_PROMPT,
        )
        self.aftersales_agent = agent_factory(
            name="aftersales_agent",
            model=model,
            tools=AFTERSALES_TOOLS,
            prompt=AFTERSALES_PROMPT,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(SupportState)
        graph.add_node("intent_router", self._route_intent)
        graph.add_node("presales_agent", self._run_presales)
        graph.add_node("aftersales_agent", self._run_aftersales)
        graph.add_node("clarify", self._clarify)
        graph.add_node("response_review", self._review_response)
        graph.add_edge(START, "intent_router")
        graph.add_conditional_edges(
            "intent_router",
            lambda state: select_route(state, self.confidence_threshold),
            {
                "presales": "presales_agent",
                "aftersales": "aftersales_agent",
                "clarify": "clarify",
            },
        )
        graph.add_edge("presales_agent", "response_review")
        graph.add_edge("aftersales_agent", "response_review")
        graph.add_edge("clarify", "response_review")
        graph.add_edge("response_review", END)
        return graph.compile()

    def _route_intent(self, state: SupportState) -> SupportState:
        question = state["question"]
        try:
            raw_decision = self.classifier.invoke(
                [
                    SystemMessage(
                        content=(
                            "请将这个电商问题分类为 presales、aftersales 或 other。"
                            "presales（售前）包括商品选择、规格和库存；"
                            "aftersales（售后）包括已有订单、物流、退换、质保或"
                            "商品问题。只判断意图，不要回答用户问题。"
                        )
                    ),
                    HumanMessage(content=question),
                ]
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, IntentDecision)
                else IntentDecision.model_validate(raw_decision)
            )
        except Exception:  # noqa: BLE001 - an unavailable model must fail open
            decision = IntentDecision(
                intent="other",
                confidence=0,
                reason="意图识别暂时不可用",
            )
        route = select_route(
            {"intent": decision.intent, "confidence": decision.confidence},
            self.confidence_threshold,
        )
        return {
            "intent": decision.intent,
            "confidence": decision.confidence,
            "route": route,
        }

    def _run_specialist(
        self, state: SupportState, agent: AgentRunner, label: str
    ) -> SupportState:
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=state["question"])]}
            )
            draft, evidence = extract_agent_result(result)
            return {"draft_response": draft, "tool_evidence": evidence}
        except Exception:  # noqa: BLE001 - an unavailable agent must fail open
            return {
                "draft_response": (
                    f"{label}客服暂时不可用，请稍后重试或补充更多信息。"
                ),
                "tool_evidence": [],
            }

    def _run_presales(self, state: SupportState) -> SupportState:
        return self._run_specialist(state, self.presales_agent, "售前")

    def _run_aftersales(self, state: SupportState) -> SupportState:
        return self._run_specialist(state, self.aftersales_agent, "售后")

    def _clarify(self, state: SupportState) -> SupportState:
        return {
            "draft_response": (
                "请问你想咨询商品选择、规格或库存，还是想查询已有订单、"
                "物流或退换货问题？"
            ),
            "tool_evidence": [],
        }

    def _review_response(self, state: SupportState) -> SupportState:
        try:
            response = self.model.invoke(
                [
                    SystemMessage(content=REVIEW_PROMPT),
                    HumanMessage(content=build_review_input(state)),
                ]
            )
            final_response = _content_to_text(response.content).strip()
            if not final_response:
                raise ValueError("回复审核员返回了空内容")
        except Exception:  # noqa: BLE001 - review failure returns the safe draft
            final_response = state.get("draft_response") or (
                "暂时无法生成回复，请稍后重试。"
            )
        return {"final_response": final_response}

    def run(self, question: str) -> SupportState:
        normalized = question.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return self.graph.invoke({"question": normalized})
