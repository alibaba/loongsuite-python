# LoongSuite MemPalace Instrumentation

Automatic OpenTelemetry instrumentation for the [MemPalace](https://github.com/MemPalace/mempalace) AI Agent memory platform (v3.x).

MemPalace implements its own JSON-RPC MCP server loop, calls LLM endpoints
through `urllib.request.urlopen` (no OpenAI/LiteLLM/Anthropic SDK), and wraps
ChromaDB in a custom `ChromaCollection` adapter. As a result, the existing
`loongsuite-instrumentation-mcp`, `opentelemetry-instrumentation-openai-v2`,
and OpenLLMetry chromadb instrumentations do not cover MemPalace's traffic.
This package wraps **9 stable anchor points** directly so the emitted spans
satisfy `/apsara/semantic-conventions/arms_docs/trace/{gen-ai,gen-ai_memory,gen-ai_mcp}.md`.

## Anchor points

| Span type | Anchor | Span kind | Operation name |
|---|---|---|---|
| MCP SERVER | `mempalace.mcp_server.handle_request` | SERVER | `mcp.server` |
| Tool + Memory (merged) | `mempalace.mcp_server.TOOLS[*].handler` | TOOL | `memory_operation` / `execute_tool` |
| Retriever | `mempalace.searcher.search_memories` / `search` | CLIENT | `retrieval` |
| Vector sub-phase | `ChromaCollection.{add,upsert,query,get,delete}` | CLIENT | `memory_operation` (`gen_ai.memory.inner_name=vector`) |
| Graph sub-phase | `mempalace.mcp_server._call_kg` | CLIENT | `memory_operation` (`inner_name=graph`) |
| Embedding | `mempalace.embedding.EmbeddinggemmaONNX.__call__` | INTERNAL | `embeddings` |
| LLM (main) | `mempalace.llm_client._http_post_json` | CLIENT | `chat` |
| LLM (closet) | `mempalace.closet_llm._call_llm` | CLIENT | `chat` |
| Task | `mempalace.service.execute_job` | INTERNAL | `run_task` |
| Chain | `mempalace.miner.mine` | INTERNAL | `workflow` |

## Installation

```bash
pip install -e instrumentation-loongsuite/loongsuite-instrumentation-mempalace/
```

## Configuration

| Env | Default | Description |
|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Capture `gen_ai.*.messages` / `tool.call.arguments` / `tool.call.result`. When on, values are second-pass redacted with `mempalace.wal._WAL_REDACT_KEYS`. |
| `OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED` | `false` | Enable Vector / Graph / Embedding sub-phase spans. |
| `OTEL_INSTRUMENTATION_MEMPALACE_EMBEDDING_SAMPLE_RATE` | `0.1` | Embedding span sampling rate (DEBUG log level → 100%). |
| `OTEL_INSTRUMENTATION_MEMPALACE_SLOW_SEARCH_THRESHOLD_S` | `2.0` | Slow-search counter threshold. |
| `OTEL_INSTRUMENTATION_MEMPALACE_SLOW_ADD_THRESHOLD_S` | `1.0` | Slow-add counter threshold. |
| `OTEL_INSTRUMENTATION_MEMPALACE_LLM_SLOW_THRESHOLD_S` | `10.0` | Slow-LLM counter threshold. |
| `OTEL_INSTRUMENTATION_MEMPALACE_ATTR_MAX_BYTES` | `4096` | Per-attribute byte cap (truncated with `…`). |
| `MEMPALACE_USER_ID` | — | Derived `gen_ai.memory.user_id` (MemPalace does not collect one). |

## Coexistence

- `ChromaCollection.*` and `_call_kg` wrappers attach
  `_SUPPRESS_INSTRUMENTATION_KEY` so OpenLLMetry chromadb / urllib
  instrumentations do not produce duplicate child spans.
- `loongsuite-instrumentation-mcp` is auto-skipped (MemPalace does not
  depend on the `mcp` SDK).

## Span hierarchy example

```
[SERVER] tools/call mempalace_search          (handle_request)
  └─ [TOOL+Memory] execute_tool mempalace_search
       ├  gen_ai.memory.operation=search
       └─ [RETRIEVER] retrieval a1b2c3d4      (search_memories)
            ├─ [EMBEDDING] embeddings embeddinggemma-300m  (sampled)
            └─ [Vector] chroma.search         (ChromaCollection.query)
```
