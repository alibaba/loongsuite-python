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

import json
import logging
from typing import Any

from opentelemetry.util.genai.extended_types import RetrievalDocument
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------

AGENT_RUN_NAMES = frozenset(
    {
        "AgentExecutor",
        "MRKLChain",
        "ReActChain",
        "ReActTextWorldAgent",
        "SelfAskWithSearchChain",
    }
)

_LANGGRAPH_REACT_METADATA_KEY = "_loongsuite_react_agent"
_DEEPAGENTS_METADATA_KEY = "_loongsuite_deepagents_agent"

LANGGRAPH_REACT_STEP_NODE = "agent"
DEEPAGENTS_REACT_STEP_NODE = "model"


def _is_agent_run(run: Any) -> bool:
    """Return *True* for classic LangChain agents (name-based check only).

    LangGraph agents are detected separately via metadata — see
    ``_has_langgraph_react_metadata`` — because their metadata propagates
    to ALL child callbacks and must be disambiguated in the tracer.
    """
    name = getattr(run, "name", "") or ""
    return name in AGENT_RUN_NAMES


def _has_langgraph_react_metadata(run: Any) -> bool:
    """Return *True* if *run* carries the LangGraph ReAct agent metadata.

    This flag is injected by ``loongsuite-instrumentation-langgraph``
    into ``config["metadata"]`` when ``Pregel.stream`` is called on a
    graph marked with ``_loongsuite_react_agent = True``.

    Note: the metadata propagates to child runs, so the caller must
    distinguish the top-level graph from child nodes.
    """
    metadata = getattr(run, "metadata", None) or {}
    return bool(metadata.get(_LANGGRAPH_REACT_METADATA_KEY))


def _has_deepagents_metadata(run: Any) -> bool:
    """Return *True* if *run* carries the DeepAgents agent metadata.

    DeepAgents builds on ``langchain.agents.create_agent`` rather than
    ``langgraph.prebuilt.create_react_agent``. Its model decision node is named
    ``"model"`` instead of LangGraph ReAct's ``"agent"`` node, so the tracer
    uses this separate marker to preserve the framework-specific STEP rule.
    """
    metadata = getattr(run, "metadata", None) or {}
    return bool(metadata.get(_DEEPAGENTS_METADATA_KEY))


# ---------------------------------------------------------------------------
# Run data extraction helpers
# ---------------------------------------------------------------------------


def _extract_model_name(run: Any) -> str | None:
    extra = getattr(run, "extra", None) or {}
    params = extra.get("invocation_params") or {}
    return (
        params.get("model_name")
        or params.get("model")
        or params.get("model_id")
    )


def _extract_provider(run: Any) -> str:
    serialized = getattr(run, "serialized", None) or {}
    id_list = serialized.get("id") or []
    if len(id_list) >= 3:
        return id_list[2]
    return "langchain"


def _extract_invocation_params(run: Any) -> dict[str, Any]:
    extra = getattr(run, "extra", None) or {}
    return extra.get("invocation_params") or {}


def _extract_tool_definitions(run: Any) -> list[FunctionToolDefinition]:
    """Extract tool definitions from LangChain Run for LLM spans.

    Tools may appear in:
    - run.extra["invocation_params"]["tools"] (e.g. from bind_tools)
    - run.inputs["tools"]

    Supports OpenAI-style format: {"type": "function", "function": {...}}
    and flat format: {"name": ..., "description": ..., "parameters": ...}.
    """
    tool_definitions: list[FunctionToolDefinition] = []
    tools: list[Any] = []

    params = _extract_invocation_params(run)
    if params and "tools" in params:
        raw = params["tools"]
        if isinstance(raw, list):
            tools = raw
        elif hasattr(raw, "__iter__") and not isinstance(raw, (str, dict)):
            tools = list(raw)

    if not tools:
        inputs = getattr(run, "inputs", None) or {}
        raw = inputs.get("tools")
        if isinstance(raw, list):
            tools = raw
        elif hasattr(raw, "__iter__") and not isinstance(raw, (str, dict)):
            tools = list(raw)

    for tool in tools:
        if isinstance(tool, FunctionToolDefinition):
            tool_definitions.append(tool)
            continue
        if isinstance(tool, dict):
            func = tool.get("function", {})
            if isinstance(func, dict) and func.get("name"):
                tool_definitions.append(
                    FunctionToolDefinition(
                        name=func.get("name", ""),
                        description=func.get("description"),
                        parameters=func.get("parameters"),
                        type="function",
                    )
                )
            elif "name" in tool:
                tool_definitions.append(
                    FunctionToolDefinition(
                        name=tool.get("name", ""),
                        description=tool.get("description"),
                        parameters=tool.get("parameters"),
                        type="function",
                    )
                )
        elif hasattr(tool, "name") and hasattr(tool, "description"):
            tool_definitions.append(
                FunctionToolDefinition(
                    name=getattr(tool, "name", ""),
                    description=getattr(tool, "description"),
                    parameters=getattr(tool, "args_schema", None)
                    or getattr(tool, "parameters", None),
                    type="function",
                )
            )

    return tool_definitions


# ---------------------------------------------------------------------------
# LangChain message ↔ util-genai message conversion
# ---------------------------------------------------------------------------


def _convert_lc_message_to_input(msg: Any) -> InputMessage | None:
    """Convert a LangChain message dict (dumpd format) to InputMessage."""
    if isinstance(msg, dict):
        kwargs = msg.get("kwargs") or {}
        role = msg.get("id", ["", "", ""])
        if isinstance(role, list) and len(role) >= 3:
            role_name = role[-1].lower().replace("message", "")
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "function": "tool",
                "tool": "tool",
                "chat": "user",
            }
            role_str = role_map.get(role_name, role_name)
        else:
            role_str = "user"

        content = kwargs.get("content", "")
        parts = []

        if role_str == "tool":
            # ToolMessage: use ToolCallResponse with tool_call_id
            tool_call_id = kwargs.get("tool_call_id")
            if isinstance(content, str):
                response_content = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                response_content = "\n".join(text_parts) if text_parts else ""
            else:
                response_content = str(content) if content else ""
            parts.append(
                ToolCallResponse(response=response_content, id=tool_call_id)
            )
        else:
            if isinstance(content, str) and content:
                parts.append(Text(content=content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(Text(content=part.get("text", "")))
                    elif isinstance(part, str):
                        parts.append(Text(content=part))

            tool_calls = kwargs.get("tool_calls") or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parts.append(
                        ToolCall(
                            name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                            id=tc.get("id"),
                        )
                    )
        if parts:
            return InputMessage(role=role_str, parts=parts)
    return None


def _extract_llm_input_messages(run: Any) -> list[InputMessage]:
    """Extract input messages from a Run's inputs."""
    inputs = getattr(run, "inputs", None) or {}
    messages: list[InputMessage] = []

    raw_messages = inputs.get("messages")
    if raw_messages:
        for batch in raw_messages:
            if isinstance(batch, list):
                for msg in batch:
                    converted = _convert_lc_message_to_input(msg)
                    if converted:
                        messages.append(converted)
        if messages:
            return messages

    prompts = inputs.get("prompts")
    if prompts and isinstance(prompts, list):
        for p in prompts:
            if isinstance(p, str):
                messages.append(
                    InputMessage(role="user", parts=[Text(content=p)])
                )
        return messages

    return messages


def _extract_llm_output_messages(run: Any) -> list[OutputMessage]:
    """Extract output messages from a completed Run."""
    outputs = getattr(run, "outputs", None) or {}
    result: list[OutputMessage] = []

    generations = outputs.get("generations") or []
    for gen_list in generations:
        if not isinstance(gen_list, list):
            continue
        for gen in gen_list:
            if not isinstance(gen, dict):
                continue
            text = gen.get("text", "")
            parts = []
            if text:
                parts.append(Text(content=text))

            msg_data = gen.get("message") or {}
            msg_kwargs = {}
            if isinstance(msg_data, dict):
                msg_kwargs = msg_data.get("kwargs") or {}

            tool_calls = msg_kwargs.get("tool_calls") or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parts.append(
                        ToolCall(
                            name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                            id=tc.get("id"),
                        )
                    )

            finish_reason = (gen.get("generation_info") or {}).get(
                "finish_reason", "stop"
            )
            if parts:
                result.append(
                    OutputMessage(
                        role="assistant",
                        parts=parts,
                        finish_reason=finish_reason or "stop",
                    )
                )
    return result


def _parse_token_usage_dict(token_usage: Any) -> tuple[int | None, int | None]:
    """Parse a token_usage/usage dict into (input_tokens, output_tokens)."""
    if not isinstance(token_usage, dict):
        return None, None
    inp = (
        token_usage.get("prompt_tokens")
        or token_usage.get("PromptTokens")
        or token_usage.get("input_tokens")
    )
    out = (
        token_usage.get("completion_tokens")
        or token_usage.get("CompletionTokens")
        or token_usage.get("output_tokens")
    )
    return (
        int(inp) if inp is not None else None,
        int(out) if out is not None else None,
    )


def _extract_token_usage(run: Any) -> tuple[int | None, int | None]:
    """Return (input_tokens, output_tokens) from a completed LLM Run.

    Tries multiple LangChain formats in order:
    1. outputs["llm_output"]["token_usage"] or ["usage"]
    2. generations[i][j]["generation_info"]["token_usage"] or ["usage"]
    3. generations[i][j]["message"].response_metadata["token_usage"] or generations[i][j]["message"].response_metadata["usage"] or generations[i][j]["message"]["kwargs"]["response_metadata"]["token_usage"] or generations[i][j]["message"]["kwargs"]["response_metadata"]["usage"]
    4. generations[i][j]["message"].usage_metadata or generations[i][j]["message"]["kwargs"]["usage_metadata"]
    """
    outputs = getattr(run, "outputs", None) or {}

    # 1. Primary: llm_output.token_usage / llm_output.usage
    llm_output = outputs.get("llm_output") or {}
    token_usage = (
        llm_output.get("token_usage") or llm_output.get("usage") or {}
    )
    inp, out = _parse_token_usage_dict(token_usage)
    if inp is not None or out is not None:
        return inp, out

    # 2. Fallback: generations[][].generation_info["token_usage"] or ["usage"]
    # 3. Fallback: generations[][].message.response_metadata["token_usage"] or generations[][].message.response_metadata["usage"] or generations[][].message["kwargs"]["response_metadata"]["token_usage"] or generations[][].message["kwargs"]["response_metadata"]["usage"]
    # 4. Fallback: generations[][].message.usage_metadata or generations[][].message["kwargs"]["usage_metadata"]
    for gen_list in outputs.get("generations") or []:
        if not isinstance(gen_list, list):
            continue
        for gen in gen_list:
            if not isinstance(gen, dict):
                continue
            # Try generation_info
            gen_info = gen.get("generation_info") or {}
            token_usage = (
                gen_info.get("token_usage") or gen_info.get("usage") or {}
            )
            inp, out = _parse_token_usage_dict(token_usage)
            if inp is not None or out is not None:
                return inp, out
            msg = gen.get("message")
            if msg is None:
                continue
            # Try message.response_metadata (serialized: kwargs.response_metadata)
            if isinstance(msg, dict):
                metadata = (msg.get("kwargs") or {}).get(
                    "response_metadata"
                ) or {}
            else:
                metadata = getattr(msg, "response_metadata", None) or {}
            if isinstance(metadata, dict):
                token_usage = (
                    metadata.get("token_usage") or metadata.get("usage") or {}
                )
                inp, out = _parse_token_usage_dict(token_usage)
                if inp is not None or out is not None:
                    return inp, out
            # Try message.usage_metadata (serialized: kwargs.usage_metadata)
            if isinstance(msg, dict):
                metadata = (msg.get("kwargs") or {}).get(
                    "usage_metadata"
                ) or {}
            else:
                metadata = getattr(msg, "usage_metadata", None) or {}
            inp, out = _parse_token_usage_dict(metadata)
            if inp is not None or out is not None:
                return inp, out

    return None, None


def _extract_finish_reasons(run: Any) -> list[str] | None:
    outputs = getattr(run, "outputs", None) or {}
    reasons: list[str] = []
    for gen_list in outputs.get("generations") or []:
        if not isinstance(gen_list, list):
            continue
        for gen in gen_list:
            if not isinstance(gen, dict):
                continue
            info = gen.get("generation_info") or {}
            reason = info.get("finish_reason")
            if reason:
                reasons.append(reason)
    return reasons or None


def _extract_response_model(run: Any) -> str | None:
    outputs = getattr(run, "outputs", None) or {}
    llm_output = outputs.get("llm_output") or {}
    return llm_output.get("model_name") or llm_output.get("model")


# ---------------------------------------------------------------------------
# Retriever document conversion
# ---------------------------------------------------------------------------


def _documents_to_retrieval_documents(documents: Any) -> list:
    """Convert retriever output documents to List[RetrievalDocument].

    Accepts LangChain Document objects (page_content, metadata) or similar.
    Extracts id from doc.id, metadata.id, metadata.doc_id, metadata.document_id.
    Extracts score from metadata.score, metadata.relevance_score, metadata.similarity_score.
    """

    result = []
    if not documents:
        return result
    for doc in documents:
        meta = getattr(doc, "metadata", None) or {}
        doc_id = (
            getattr(doc, "id", None)
            or meta.get("id")
            or meta.get("doc_id")
            or meta.get("document_id")
        )
        score = (
            meta.get("score")
            or meta.get("relevance_score")
            or meta.get("similarity_score")
        )
        content = getattr(doc, "page_content", None) or getattr(
            doc, "content", None
        )
        result.append(
            RetrievalDocument(
                id=doc_id,
                score=score,
                content=content,
                metadata=meta if meta else None,
            )
        )
    return result


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _safe_json(obj: Any, max_len: int = 4096) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        logger.debug(
            "Failed to JSON serialize object, using str()", exc_info=True
        )
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + "...[truncated]"
    return s
