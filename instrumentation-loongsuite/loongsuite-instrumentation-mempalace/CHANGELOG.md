# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- Initial `loongsuite-instrumentation-mempalace` package.
- Wraps 9 stable MemPalace anchor points (MCP SERVER, Tool+Memory merged,
  Retriever, Vector sub-phase, Graph sub-phase, Embedding, two LLM paths,
  Task, Chain) with `gen_ai.*` / `mcp.*` semantic attributes per
  `trace/{gen-ai,gen-ai_memory,gen-ai_mcp}.md`.
- Metrics: `mcp_server_operation_duration`, `gen_ai_memory_operation_*`,
  `gen_ai_memory_inner_operation_*`, `genai_calls_*`, `genai_llm_usage_tokens`.
- Sub-phase spans (Vector/Graph/Embedding) gated by
  `OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED` (default off).
- `_SUPPRESS_INSTRUMENTATION_KEY` attached inside Vector / Graph / Embedding
  wrappers to avoid double-nesting with OpenLLMetry chromadb / urllib.
- `capture-message-content` default off; second-pass redaction with
  `mempalace.wal._WAL_REDACT_KEYS`; `palace_path` hashed to sha256[:8].
