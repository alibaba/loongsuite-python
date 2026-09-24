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
Memory operation semantic convention attributes for GenAI operations.
These attributes are defined according to the memory-span.md and memory-log.md specifications.
"""

from enum import Enum
from typing import Final

# Memory operation type
GEN_AI_MEMORY_OPERATION: Final = "gen_ai.memory.operation"
"""
The type of memory operation being performed.
Values: add, search, update, batch_update, get, get_all, history, delete, batch_delete, delete_all
"""

# Memory identifiers
GEN_AI_MEMORY_USER_ID: Final = "gen_ai.memory.user_id"
"""
User identifier for the memory operation.
"""

GEN_AI_MEMORY_AGENT_ID: Final = "gen_ai.memory.agent_id"
"""
Agent identifier for the memory operation.
"""

GEN_AI_MEMORY_RUN_ID: Final = "gen_ai.memory.run_id"
"""
Run identifier for the memory operation.
"""

GEN_AI_MEMORY_APP_ID: Final = "gen_ai.memory.app_id"
"""
Application identifier for the memory operation (for managed platforms).
"""

GEN_AI_MEMORY_ID: Final = "gen_ai.memory.id"
"""
Memory ID for the operation.
"""

# Memory operation parameters
GEN_AI_MEMORY_LIMIT: Final = "gen_ai.memory.limit"
"""
Limit on the number of results to return.
"""

GEN_AI_MEMORY_PAGE: Final = "gen_ai.memory.page"
"""
Page number for pagination.
"""

GEN_AI_MEMORY_PAGE_SIZE: Final = "gen_ai.memory.page_size"
"""
Page size for pagination.
"""

GEN_AI_MEMORY_TOP_K: Final = "gen_ai.memory.top_k"
"""
Number of top K results to return (for managed APIs).
"""

GEN_AI_MEMORY_MEMORY_TYPE: Final = "gen_ai.memory.memory_type"
"""
Type of memory (e.g., procedural_memory).
"""

GEN_AI_MEMORY_THRESHOLD: Final = "gen_ai.memory.threshold"
"""
Similarity threshold for search operations.
"""

GEN_AI_MEMORY_RERANK: Final = "gen_ai.memory.rerank"
"""
Whether reranking is enabled.
"""

# Memory content
GEN_AI_MEMORY_INPUT_MESSAGES: Final = "gen_ai.memory.input.messages"
"""
The original memory content for the operation.
"""

GEN_AI_MEMORY_OUTPUT_MESSAGES: Final = "gen_ai.memory.output.messages"
"""
The query results returned from the memory operation.
"""


class GenAiMemoryOperationValues(Enum):
    """Memory operation type values."""

    ADD = "add"
    """Add a memory record."""

    SEARCH = "search"
    """Search memory records."""

    UPDATE = "update"
    """Update a memory record."""

    BATCH_UPDATE = "batch_update"
    """Batch update memory records."""

    GET = "get"
    """Get a specific memory record."""

    GET_ALL = "get_all"
    """Get all memory records."""

    HISTORY = "history"
    """Get memory history."""

    DELETE = "delete"
    """Delete a memory record."""

    BATCH_DELETE = "batch_delete"
    """Batch delete memory records."""

    DELETE_ALL = "delete_all"
    """Delete all memory records."""
