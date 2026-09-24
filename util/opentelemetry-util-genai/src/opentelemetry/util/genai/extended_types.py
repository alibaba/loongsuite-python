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

"""
Extended types for GenAI operations.
This module defines invocation types for additional GenAI operations
that are not supported by the base types module.
This is an extension module that does not modify the original types.py,
allowing for easy upstream synchronization without conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from opentelemetry.trace import Span
from opentelemetry.util.genai.types import (
    ContextToken,
    InputMessage,
    MessagePart,
    OutputMessage,
    ToolDefinition,
)


@dataclass()
class RetrievalDocument:
    """
    Represents a single document retrieved from a vector database or search system.
    """

    id: str | None
    score: float | None
    content: str | None = None
    metadata: dict[str, Any] | None = None


def _new_str_any_dict() -> Dict[str, Any]:
    """Helper function to create a new empty dict for default factory."""
    return {}


def _new_input_messages() -> List[InputMessage]:
    """Helper function to create a new empty list for default factory."""
    return []


def _new_output_messages() -> List[OutputMessage]:
    """Helper function to create a new empty list for default factory."""
    return []


def _new_tool_definitions() -> List[ToolDefinition]:
    """Helper function to create a new empty list for default factory."""
    return []


def _new_system_instruction() -> List[MessagePart]:
    """Helper function to create a new empty list for default factory."""
    return []


def _new_retrieval_documents() -> List[RetrievalDocument]:
    """Helper function to create a new empty list for default factory."""
    return []


@dataclass
class EmbeddingInvocation:
    """
    Represents a single embedding invocation.
    When creating an EmbeddingInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    request_model: str
    context_token: ContextToken | None = None
    span: Span | None = None
    provider: str | None = None
    response_model_name: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Embedding-specific attributes
    dimension_count: int | None = None
    encoding_formats: List[str] | None = None
    server_address: str | None = None
    server_port: int | None = None
    monotonic_start_s: float | None = None


@dataclass
class ExecuteToolInvocation:
    """
    Represents a single tool execution invocation.
    When creating an ExecuteToolInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    tool_name: str
    context_token: ContextToken | None = None
    span: Span | None = None
    provider: str | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Tool-specific attributes
    tool_call_id: str | None = None
    tool_description: str | None = None
    tool_type: str | None = None  # function, extension, datastore
    tool_call_arguments: Any = None
    tool_call_result: Any = None
    # Skill attributes (conditionally set when this tool execution loaded a skill)
    skill_name: str | None = None
    skill_id: str | None = None
    skill_description: str | None = None
    skill_version: str | None = None
    monotonic_start_s: float | None = None


@dataclass
class CreateAgentInvocation:
    """
    Represents a single agent creation invocation.
    When creating a CreateAgentInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    provider: str
    context_token: ContextToken | None = None
    span: Span | None = None
    agent_name: str | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Agent-specific attributes
    agent_id: str | None = None
    agent_description: str | None = None
    request_model: str | None = None
    # Server information
    server_address: str | None = None
    server_port: int | None = None
    monotonic_start_s: float | None = None


@dataclass
class InvokeAgentInvocation:
    """
    Represents a single agent invocation.
    When creating an InvokeAgentInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    provider: str
    context_token: ContextToken | None = None
    span: Span | None = None
    agent_name: str | None = None
    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )
    output_messages: List[OutputMessage] = field(
        default_factory=_new_output_messages
    )
    tool_definitions: List[ToolDefinition] = field(
        default_factory=_new_tool_definitions
    )
    system_instruction: List[MessagePart] = field(
        default_factory=_new_system_instruction
    )
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Agent-specific attributes
    agent_id: str | None = None
    agent_description: str | None = None
    conversation_id: str | None = None
    data_source_id: str | None = None
    request_model: str | None = None
    response_model_name: str | None = None
    response_id: str | None = None
    finish_reasons: List[str] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_cache_creation_input_tokens: int | None = None
    usage_cache_read_input_tokens: int | None = None
    # Request parameters
    output_type: str | None = None
    choice_count: int | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    stop_sequences: List[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Server information
    server_address: str | None = None
    server_port: int | None = None
    monotonic_start_s: float | None = None
    monotonic_end_s: float | None = None
    monotonic_first_token_s: float | None = None
    """Streaming: first-token time; TTFT attribute derived like ``LLMInvocation``."""


@dataclass
class RetrievalInvocation:
    """
    Represents a single document retrieval invocation.
    When creating a RetrievalInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.

    Per LoongSuite semantic convention:
    - gen_ai.operation.name: retrieval
    - gen_ai.span.kind: RETRIEVER
    - Span name: retrieval {data_source_id}
    """

    context_token: ContextToken | None = None
    span: Span | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Retrieval-specific attributes
    query: str | None = None
    documents: List[RetrievalDocument] = field(
        default_factory=_new_retrieval_documents
    )
    data_source_id: str | None = None
    provider: str | None = None
    request_model: str | None = None
    top_k: float | None = None
    server_address: str | None = None
    server_port: int | None = None
    monotonic_start_s: float | None = None


@dataclass
class RerankInvocation:
    """
    Represents a single rerank invocation.
    When creating a RerankInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    provider: str
    context_token: ContextToken | None = None
    span: Span | None = None
    request_model: str | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Rerank-specific attributes
    top_k: int | None = None  # gen_ai.request.top_k
    documents_count: int | None = None  # gen_ai.rerank.documents_count
    temperature: float | None = None  # For LLM Reranker
    max_tokens: int | None = None  # For LLM Reranker
    scoring_prompt: str | None = None  # For LLM Reranker
    return_documents: bool | None = None  # For Cohere
    max_chunks_per_doc: int | None = None  # For Cohere
    device: str | None = None  # For HuggingFace/SentenceTransformer
    batch_size: int | None = None  # For HuggingFace/SentenceTransformer
    max_length: int | None = None  # For HuggingFace
    normalize: bool | None = None  # For HuggingFace
    input_documents: Any = (
        None  # gen_ai.rerank.input_documents (optional, sensitive)
    )
    output_documents: Any = (
        None  # gen_ai.rerank.output_documents (optional, sensitive)
    )
    monotonic_start_s: float | None = None


@dataclass
class EntryInvocation:
    """
    Represents a single AI application system entry invocation.

    Entry identifies the call entry point to an AI application system.
    Follows LoongSuite semantic conventions: gen_ai.span.kind=ENTRY,
    gen_ai.operation.name=enter.

    When creating an EntryInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    context_token: ContextToken | None = None
    span: Span | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    session_id: str | None = None  # gen_ai.session.id
    user_id: str | None = None  # gen_ai.user.id
    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )
    output_messages: List[OutputMessage] = field(
        default_factory=_new_output_messages
    )
    system_instruction: List[MessagePart] = field(
        default_factory=_new_system_instruction
    )
    tool_definitions: List[ToolDefinition] = field(
        default_factory=_new_tool_definitions
    )
    response_time_to_first_token: int | None = None  # nanoseconds
    monotonic_start_s: float | None = None


@dataclass
class ReactStepInvocation:
    """
    Represents a single ReAct step invocation.

    ReAct Step identifies one Reasoning-Acting iteration in an Agent.
    Follows LoongSuite semantic conventions: gen_ai.span.kind=STEP,
    gen_ai.operation.name=react.

    When creating a ReactStepInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    context_token: ContextToken | None = None
    span: Span | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    finish_reason: str | None = None  # gen_ai.react.finish_reason
    round: int | None = None  # gen_ai.react.round, 1-based
    monotonic_start_s: float | None = None
