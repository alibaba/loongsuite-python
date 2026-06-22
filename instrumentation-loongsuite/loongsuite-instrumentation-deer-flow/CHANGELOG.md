# Changelog

## 0.1.0.dev

* Initial release: DeerFlow 2.1+ instrumentation following the hybrid
  LangChain-callback-reuse + 7-wrapt-monkey-patch plan from
  `llm-dev/deer-flow/investigate/execute.md`.
* Covers Entry / Agent (subagent) / Task (task_tool, sandbox lifecycle,
  memory load/save/update) / Tool (Sandbox ABC) spans. LLM / Tool (LangChain)
  / ReAct Step spans are delegated to `loongsuite-instrumentation-langchain`.
* PII handling: Memory content gated by a two-level switch
  (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` +
  `OTEL_DEER_FLOW_CAPTURE_MEMORY_CONTENT`).
