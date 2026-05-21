# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared constants for deepagents instrumentation."""

from __future__ import annotations

FRAMEWORK_NAME = "deepagents"

GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_TYPE = "gen_ai.agent.type"
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_FRAMEWORK_VERSION = "gen_ai.framework.version"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN = "gen_ai.response.time_to_first_token"
GEN_AI_SESSION_ID = "gen_ai.session.id"
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
    "gen_ai.usage.cache_creation.input_tokens"
)
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = (
    "gen_ai.usage.cache_read.input_tokens"
)
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

SPAN_KIND_AGENT = "AGENT"
SPAN_KIND_CHAIN = "CHAIN"
SPAN_KIND_EMBEDDING = "EMBEDDING"
SPAN_KIND_ENTRY = "ENTRY"
SPAN_KIND_LLM = "LLM"
SPAN_KIND_RERANKER = "RERANKER"
SPAN_KIND_RETRIEVER = "RETRIEVER"
SPAN_KIND_STEP = "STEP"
SPAN_KIND_TASK = "TASK"
SPAN_KIND_TOOL = "TOOL"

ENTRY_PARENT_KINDS = {SPAN_KIND_ENTRY, SPAN_KIND_AGENT}
GENAI_SPAN_KINDS = {
    SPAN_KIND_AGENT,
    SPAN_KIND_CHAIN,
    SPAN_KIND_EMBEDDING,
    SPAN_KIND_ENTRY,
    SPAN_KIND_LLM,
    SPAN_KIND_RERANKER,
    SPAN_KIND_RETRIEVER,
    SPAN_KIND_STEP,
    SPAN_KIND_TASK,
    SPAN_KIND_TOOL,
}

METADATA_LS_INTEGRATION = "ls_integration"
METADATA_LS_AGENT_TYPE = "ls_agent_type"
METADATA_LC_AGENT_NAME = "lc_agent_name"
METADATA_VERSIONS = "versions"
METADATA_DEEPAGENTS_VERSION = "deepagents"

SUBAGENT_TYPE = "subagent"
TASK_TOOL_NAME = "task"
TOOL_TYPE_AGENT = "agent"

GRAPH_ATTR = "_loongsuite_deepagents_graph"
GRAPH_VERSION_ATTR = "_loongsuite_deepagents_version"
GRAPH_METADATA_ATTR = "_loongsuite_deepagents_metadata"
GRAPH_REGISTRY_ATTR = "_loongsuite_deepagents_subagent_registry"
GRAPH_ORIGINAL_METHODS_ATTR = "_loongsuite_deepagents_original_methods"
GRAPH_METHODS_WRAPPED_ATTR = "_loongsuite_deepagents_methods_wrapped"
LANGGRAPH_REACT_AGENT_METADATA_KEY = "_loongsuite_react_agent"

CREATE_DEEP_AGENT_MODULE = "deepagents.graph"
CREATE_DEEP_AGENT_NAME = "create_deep_agent"

METRIC_CALLS_COUNT = "genai_calls_count"
METRIC_CALLS_DURATION_SECONDS = "genai_calls_duration_seconds"
METRIC_CALLS_ERROR_COUNT = "genai_calls_error_count"
METRIC_CALLS_SLOW_COUNT = "genai_calls_slow_count"
METRIC_LLM_FIRST_TOKEN_SECONDS = "genai_llm_first_token_seconds"
METRIC_LLM_USAGE_TOKENS = "genai_llm_usage_tokens"

DEFAULT_SLOW_THRESHOLDS_SECONDS = {
    SPAN_KIND_ENTRY: 60.0,
    SPAN_KIND_AGENT: 30.0,
    SPAN_KIND_CHAIN: 10.0,
    SPAN_KIND_STEP: 10.0,
    SPAN_KIND_LLM: 10.0,
    SPAN_KIND_TOOL: 10.0,
    SPAN_KIND_RETRIEVER: 5.0,
    SPAN_KIND_RERANKER: 5.0,
    SPAN_KIND_EMBEDDING: 5.0,
    SPAN_KIND_TASK: 30.0,
}

USAGE_TOKEN_ATTRIBUTES = {
    "input": GEN_AI_USAGE_INPUT_TOKENS,
    "output": GEN_AI_USAGE_OUTPUT_TOKENS,
    "total": GEN_AI_USAGE_TOTAL_TOKENS,
    "cache_creation": GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    "cache_read": GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
}
