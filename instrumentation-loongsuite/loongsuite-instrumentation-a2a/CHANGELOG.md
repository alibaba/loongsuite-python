# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-a2a`: automatic
  instrumentation for the official A2A (Agent2Agent) Python SDK (`a2a-sdk`).
  Produces an ARMS gen-ai `AGENT` span around each server-side
  `AgentExecutor.execute` invocation (covering both existing and
  future-defined executor subclasses), complementing — rather than
  duplicating — the SDK's built-in transport/request-handler tracing.
  ([#28](https://github.com/alibaba/loongsuite-python/issues/28))
- Span kind and content-capture are sourced from the shared GenAI util
  (`opentelemetry-util-genai`): the span kind comes from `GenAiSpanKindValues`
  and content capture is governed by the standard
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` switch (default
  `NO_CONTENT`), so user prompts are never exported without explicit opt-in.
- Task context available at the execution boundary is attached using the A2A
  semantic-conventions draft keys `a2a.task.id` / `a2a.task.state`
  ([semantic-conventions-genai#195](https://github.com/open-telemetry/semantic-conventions-genai/pull/195)),
  keeping agent execution distinct from A2A protocol operations; full protocol
  instrumentation can follow separately.
- Telemetry is fail-safe: any error while recording span attributes is
  swallowed so instrumentation can never interrupt agent execution.
- The AGENT span name uses the concrete executor class
  (`invoke_agent {ClassName}`), matching sibling agent instrumentations.
