"""Cognee-specific attribute constants and span-name prefix rules.

Source: cognee/modules/observability/tracing.py:32-58 (v1.2.1)
"""

COGNEE_DB_SYSTEM = "cognee.db.system"
COGNEE_DB_QUERY = "cognee.db.query"
COGNEE_DB_ROW_COUNT = "cognee.db.row_count"
COGNEE_LLM_MODEL = "cognee.llm.model"
COGNEE_LLM_PROVIDER = "cognee.llm.provider"
COGNEE_SEARCH_TYPE = "cognee.search.type"
COGNEE_SEARCH_QUERY = "cognee.search.query"
COGNEE_PIPELINE_TASK_NAME = "cognee.pipeline.task_name"
COGNEE_VECTOR_COLLECTION = "cognee.vector.collection"
COGNEE_VECTOR_RESULT_COUNT = "cognee.vector.result_count"
COGNEE_SPAN_CATEGORY = "cognee.span.category"
COGNEE_RESULT_SUMMARY = "cognee.result.summary"
COGNEE_RESULT_COUNT = "cognee.result.count"
COGNEE_PIPELINE_NAME = "cognee.pipeline.name"

COGNEE_LLM_PROMPT_PATH = "cognee.llm.prompt_path"
COGNEE_LLM_CONTEXT_LENGTH = "cognee.llm.context_length"
COGNEE_LLM_QUERY_LENGTH = "cognee.llm.query_length"
COGNEE_LLM_RESPONSE_LENGTH = "cognee.llm.response_length"

COGNEE_SEARCH_TOP_K = "cognee.search.top_k"
COGNEE_RETRIEVAL_TOP_K = "cognee.retrieval.top_k"
COGNEE_RECALL_TOP_K = "cognee.recall.top_k"

# Default prompt-path constant used by AgenticRetriever (v1.2.1)
AGENTIC_USER_PROMPT_FILENAME = "agentic_user.txt"
