"""LangGraph workflow for a text-only e-commerce support example."""

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

PRESALES_PROMPT = """You are the pre-sales specialist for a fictional store.
Use only the supplied synthetic catalog, product knowledge, and inventory tools.
Call at least one tool before answering. If a requested fact is unavailable, say so.
Never invent prices, promotions, delivery promises, or product capabilities.
"""

AFTERSALES_PROMPT = """You are the after-sales specialist for a fictional store.
Use only the supplied synthetic order, policy, and issue-assessment tools.
Call the order tool before discussing an order. Never promise a refund or replacement;
describe the policy and the next verification step instead.
"""

REVIEW_PROMPT = """You are the final response reviewer for a fictional store.
Rewrite the draft as a concise, helpful customer reply. Preserve only facts supported
by the tool evidence. Do not add prices, stock, policy, order status, guarantees, or
actions that are absent from the evidence. If evidence is missing, keep the uncertainty.
Return only the final customer-facing answer.
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
        raise ValueError("specialist agent returned no final answer")
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
                            "Classify this e-commerce question as presales, aftersales, "
                            "or other. Presales covers product selection, specifications, "
                            "and stock. Aftersales covers an existing order, delivery, "
                            "returns, warranty, or a product issue."
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
                reason="intent classification was unavailable",
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
                    f"The {label} specialist is temporarily unavailable. "
                    "Please try again or provide more details."
                ),
                "tool_evidence": [],
            }

    def _run_presales(self, state: SupportState) -> SupportState:
        return self._run_specialist(state, self.presales_agent, "pre-sales")

    def _run_aftersales(self, state: SupportState) -> SupportState:
        return self._run_specialist(
            state, self.aftersales_agent, "after-sales"
        )

    def _clarify(self, state: SupportState) -> SupportState:
        return {
            "draft_response": (
                "Could you clarify whether you are asking about choosing a product "
                "or about an existing order?"
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
                raise ValueError("reviewer returned an empty response")
        except Exception:  # noqa: BLE001 - review failure returns the safe draft
            final_response = state["draft_response"]
        return {"final_response": final_response}

    def run(self, question: str) -> SupportState:
        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return self.graph.invoke({"question": normalized})
