"""
Semantic conventions for MemPalace instrumentation.

Covers Gen-AI common, Memory, Vector, Graph, MCP and Tool attributes from
/apsara/semantic-conventions/arms_docs/trace/{gen-ai,gen-ai_memory,gen-ai_mcp}.md.
"""


class SemanticAttributes:
    # ===== Gen-AI common =====
    GEN_AI_SPAN_KIND = "gen_ai.span.kind"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_FRAMEWORK = "gen_ai.framework"
    GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"

    # ===== Tool =====
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
    GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
    GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

    # ===== Memory common =====
    GEN_AI_MEMORY_OPERATION = "gen_ai.memory.operation"
    GEN_AI_MEMORY_USER_ID = "gen_ai.memory.user_id"
    GEN_AI_MEMORY_AGENT_ID = "gen_ai.memory.agent_id"
    GEN_AI_MEMORY_RUN_ID = "gen_ai.memory.run_id"
    GEN_AI_MEMORY_APP_ID = "gen_ai.memory.app_id"
    GEN_AI_MEMORY_RESULT_COUNT = "gen_ai.memory.result_count"
    GEN_AI_MEMORY_MEMORY_TYPE = "gen_ai.memory.memory_type"
    GEN_AI_MEMORY_INPUT_MESSAGES = "gen_ai.memory.input.messages"
    GEN_AI_MEMORY_OUTPUT_MESSAGES = "gen_ai.memory.output.messages"
    GEN_AI_MEMORY_METADATA = "gen_ai.memory.metadata"
    GEN_AI_MEMORY_ID = "gen_ai.memory.id"
    GEN_AI_MEMORY_LIMIT = "gen_ai.memory.limit"
    GEN_AI_MEMORY_TOP_K = "gen_ai.memory.top_k"
    GEN_AI_MEMORY_THRESHOLD = "gen_ai.memory.threshold"
    GEN_AI_MEMORY_RERANK = "gen_ai.memory.rerank"
    GEN_AI_MEMORY_KEYWORD_SEARCH = "gen_ai.memory.keyword_search"
    GEN_AI_MEMORY_FILTER_KEYS = "gen_ai.memory.filter_keys"
    GEN_AI_MEMORY_INNER_NAME = "gen_ai.memory.inner_name"

    # ===== Data source (Vector/Graph common) =====
    GEN_AI_MEMORY_DATA_SOURCE_TYPE = "gen_ai.memory.data_source.type"
    GEN_AI_MEMORY_DATA_SOURCE_URL = "gen_ai.memory.data_source.url"

    # ===== Vector =====
    GEN_AI_MEMORY_VECTOR_COLLECTION = "gen_ai.memory.vector.collection"
    GEN_AI_MEMORY_VECTOR_METHOD = "gen_ai.memory.vector.method"
    GEN_AI_MEMORY_VECTOR_LIMIT = "gen_ai.memory.vector.limit"
    GEN_AI_MEMORY_VECTOR_FILTERS_KEYS = "gen_ai.memory.vector.filter_keys"
    GEN_AI_MEMORY_VECTOR_FILTERS_OPERATORS = "gen_ai.memory.vector.filters_operators"
    GEN_AI_MEMORY_VECTOR_RESULT_COUNT = "gen_ai.memory.vector.result_count"
    GEN_AI_MEMORY_VECTOR_METRIC_TYPE = "gen_ai.memory.vector.metric_type"
    GEN_AI_MEMORY_VECTOR_EMBEDDING_DIMS = "gen_ai.memory.vector.embedding_dims"

    # ===== Graph =====
    GEN_AI_MEMORY_GRAPH_METHOD = "gen_ai.memory.graph.method"
    GEN_AI_MEMORY_GRAPH_RESULT_COUNT = "gen_ai.memory.graph.result_count"

    # ===== Embedding =====
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"

    # ===== Retrieval =====
    GEN_AI_RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"
    GEN_AI_RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"

    # ===== Task / Chain =====
    GEN_AI_TASK_NAME = "gen_ai.task.name"
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"

    # ===== MCP =====
    MCP_METHOD_NAME = "mcp.method.name"
    MCP_TOOL_NAME = "mcp.tool.name"
    MCP_SESSION_ID = "mcp.session.id"
    MCP_OUTPUT_SIZE = "mcp.output.size"
    MCP_CLIENT_VERSION = "mcp.client.version"
    MCP_ARGUMENTS = "mcp.arguments"

    # ===== RPC =====
    RPC_JSONRPC_REQUEST_ID = "rpc.jsonrpc.request_id"
    RPC_JSONRPC_ERROR_CODE = "rpc.jsonrpc.error_code"

    # ===== Network / server =====
    NETWORK_PROTOCOL_VERSION = "network.protocol.version"
    NETWORK_TRANSPORT = "network.transport"
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"

    # ===== Error =====
    ERROR_TYPE = "error.type"


class SpanKindValues:
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    TOOL = "TOOL"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    CHAIN = "CHAIN"
    TASK = "TASK"


class OperationNameValues:
    MCP_SERVER = "mcp.server"
    MEMORY_OPERATION = "memory_operation"
    EXECUTE_TOOL = "execute_tool"
    CHAT = "chat"
    RETRIEVAL = "retrieval"
    EMBEDDINGS = "embeddings"
    WORKFLOW = "workflow"
    RUN_TASK = "run_task"


# tool_name -> (memory.operation, memory_type)
TOOL_MEMORY_MAP: dict[str, tuple[str, str | None]] = {
    "mempalace_add_drawer": ("add", "procedural_memory"),
    "mempalace_diary_write": ("add", "episodic_memory"),
    "mempalace_kg_add": ("add", "entity_memory"),
    "mempalace_search": ("search", None),
    "mempalace_check_duplicate": ("search", None),
    "mempalace_kg_query": ("search", None),
    "mempalace_get_drawer": ("get", None),
    "mempalace_list_drawers": ("get_all", None),
    "mempalace_list_wings": ("get_all", None),
    "mempalace_list_rooms": ("get_all", None),
    "mempalace_get_taxonomy": ("get_all", None),
    "mempalace_kg_timeline": ("get_all", None),
    "mempalace_kg_stats": ("get_all", None),
    "mempalace_graph_stats": ("get_all", None),
    "mempalace_update_drawer": ("update", None),
    "mempalace_delete_drawer": ("delete", None),
    "mempalace_delete_by_source": ("delete", None),
    "mempalace_delete_tunnel": ("delete", None),
    "mempalace_delete_hallway": ("delete", None),
}

# tools that represent Chain/workflow orchestration rather than memory op
TOOL_CHAIN_SET: set[str] = {
    "mempalace_mine",
    "mempalace_sync",
    "mempalace_checkpoint",
}
