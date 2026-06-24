"""Semantic convention constants and MAF→ARMS attribute mapping table.

Centralizes:
- ``gen_ai.span.kind`` enum values (per ARMS ``gen-ai.md``).
- ``gen_ai.operation.name`` enum values.
- MAF private-prefix attribute → ``gen_ai.*`` rename map (kept as a local constant,
  aligned with the pattern in
  ``opentelemetry-instrumentation-openai-agents-v2/.../span_processor.py``).
"""

from __future__ import annotations

from typing import Final


class GenAISpanKind:
    """``gen_ai.span.kind`` enumeration (ARMS gen-ai.md)."""

    AGENT = "AGENT"
    LLM = "LLM"
    TOOL = "TOOL"
    EMBEDDING = "EMBEDDING"
    CHAIN = "CHAIN"
    TASK = "TASK"
    STEP = "STEP"
    ENTRY = "ENTRY"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    CLIENT = "CLIENT"


class GenAIOperation:
    """``gen_ai.operation.name`` enumeration (ARMS gen-ai.md)."""

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    GENERATE_CONTENT = "generate_content"
    EMBEDDINGS = "embeddings"
    EXECUTE_TOOL = "execute_tool"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    RETRIEVAL = "retrieval"
    WORKFLOW = "workflow"
    TASK = "task"
    REACT = "react"
    MCP = "mcp"


# MAF span-name prefixes (from observability.py OtelAttr) — used to classify a
# span when ``gen_ai.operation.name`` is not already set on it.
MAF_SPAN_NAME_PREFIXES: Final[dict[str, str]] = {
    "chat ": GenAIOperation.CHAT,
    "embeddings ": GenAIOperation.EMBEDDINGS,
    "execute_tool ": GenAIOperation.EXECUTE_TOOL,
    "invoke_agent ": GenAIOperation.INVOKE_AGENT,
    "create_agent ": GenAIOperation.CREATE_AGENT,
    "workflow.run": GenAIOperation.WORKFLOW,
    "workflow.build": GenAIOperation.WORKFLOW,
    "message.send": GenAIOperation.WORKFLOW,
    "executor.process": GenAIOperation.WORKFLOW,
    "edge_group.process": GenAIOperation.WORKFLOW,
    "react step": GenAIOperation.REACT,
}

# MAF writes some attributes with its own private prefix (no ``gen_ai.`` prefix).
# We rename them to the ARMS-spec key in ``MAFSemanticProcessor.on_end`` so that
# downstream platforms see a single canonical key. The mapping is idempotent —
# if MAF later writes the canonical key directly, the rename becomes a no-op
# because the source key won't be present.
MAF_ATTR_RENAME_MAP: Final[dict[str, str]] = {
    # Workflow / chain attributes (observability.py:247-281)
    "workflow.id": "gen_ai.workflow.id",
    "workflow.name": "gen_ai.workflow.name",
    "workflow.description": "gen_ai.workflow.description",
    "workflow.definition": "gen_ai.workflow.definition",
    "workflow_builder.name": "gen_ai.workflow.builder.name",
    "workflow_builder.description": "gen_ai.workflow.builder.description",
    # Executor / task attributes (observability.py:266-272)
    "executor.id": "gen_ai.task.name",
    "executor.type": "gen_ai.task.type",
    # Edge group attributes (observability.py:270-274)
    "edge_group.type": "gen_ai.edge_group.type",
    "edge_group.id": "gen_ai.edge_group.id",
    # Message attributes (observability.py:276-281)
    "message.source_id": "gen_ai.message.source_id",
    "message.target_id": "gen_ai.message.target_id",
    "message.type": "gen_ai.message.type",
    "message.payload_type": "gen_ai.message.payload_type",
    "message.destination_executor_id": "gen_ai.message.destination_executor_id",
}

# Provider name normalization — collapse MAF-specific provider spellings to the
# canonical OTel/ARMS value to avoid dimension sprawl in metrics.
PROVIDER_NAME_NORMALIZE: Final[dict[str, str]] = {
    "azure_openai": "openai",
    "azure_ai_openai": "openai",
    "azure.openai": "openai",
    "microsoft.agent_framework": "openai",
}

# Attribute keys we read off the span. Centralized so tests can import them.
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_TTFT = "gen_ai.response.time_to_first_token"
GEN_AI_USER_TTFT = "gen_ai.user.time_to_first_token"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_REACT_ROUND = "gen_ai.react.round"
GEN_AI_REACT_FINISH_REASON = "gen_ai.react.finish_reason"
GEN_AI_FRAMEWORK = "gen_ai.framework"
ERROR_TYPE = "error.type"

FRAMEWORK_NAME = "microsoft-agent-framework"
