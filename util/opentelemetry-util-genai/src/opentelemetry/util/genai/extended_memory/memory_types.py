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

from dataclasses import dataclass, field
from typing import Any, Dict

from opentelemetry.util.genai.types import ContextToken, Span


def _new_str_any_dict() -> Dict[str, Any]:
    """Helper function to create a new empty dict for default factory."""
    return {}


@dataclass
class MemoryInvocation:
    """
    Represents a single memory operation invocation.
    When creating a MemoryInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    operation: str  # Memory operation type (add, search, update, etc.)
    context_token: ContextToken | None = None
    span: Span | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Memory identifiers (conditionally required)
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    app_id: str | None = None
    # Memory operation parameters (optional)
    memory_id: str | None = None
    limit: int | None = None
    page: int | None = None
    page_size: int | None = None
    top_k: int | None = None
    memory_type: str | None = None
    threshold: float | None = None
    rerank: bool | None = None
    # Memory content (optional, controlled by content capturing mode)
    input_messages: Any = None  # Original memory content
    output_messages: Any = None  # Query results
    # Server information
    server_address: str | None = None
    server_port: int | None = None
    monotonic_start_s: float | None = None
