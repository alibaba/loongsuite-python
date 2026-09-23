# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-llama-index`: automatic
  instrumentation for LlamaIndex (`llama-index-core`) via its native
  instrumentation dispatcher, projecting LlamaIndex spans/events onto
  OpenTelemetry spans that follow the ARMS gen-ai semantic conventions
  (LLM / EMBEDDING / RETRIEVER / RERANKER / TASK / CHAIN / AGENT / TOOL), with
  parent/child trace relationships preserved from `parent_span_id`.
  ([#18](https://github.com/alibaba/loongsuite-python/issues/18))
- Span-kind and content-capture semantics are sourced from the shared GenAI
  util (`opentelemetry-util-genai`): span kinds come from `GenAiSpanKindValues`
  and content capture is governed by the standard
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` switch, so this package
  behaves consistently with the rest of loongsuite.

### Fixed

- A class name containing `Agent` no longer forces internal methods to `AGENT`
  spans: `call_tool` is now a `TOOL` span, agent-loop steps (`setup_agent`,
  `take_step`, `finalize`, `handle_tool_call_results`, ...) are `CHAIN` spans,
  and only a genuine agent invocation (`run`/`chat` on an agent) is `AGENT`.
- `astructured_predict` / `stream_structured_predict` /
  `astream_structured_predict` are now classified as `LLM` calls.
- `uninstrument()` no longer strands spans that were open when it ran: the
  handler stops creating new spans and drains (ends) any still-open spans
  before it is detached from the dispatcher.
