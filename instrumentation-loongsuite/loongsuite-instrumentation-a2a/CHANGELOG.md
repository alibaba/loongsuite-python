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
- The agent execution span is owned end-to-end by the shared
  `opentelemetry-util-genai` `ExtendedTelemetryHandler`: the
  instrumentation builds an `InvokeAgentInvocation` at the executor
  boundary and drives it through `start_invoke_agent` /
  `stop_invoke_agent` / `fail_invoke_agent`, so span start/end,
  attributes, content capture, events, metrics and error recording all
  come from the shared util.
- No extra structural span is emitted: the handler-owned `invoke_agent`
  AGENT span is itself the single executor-boundary span (matching the
  Hermes instrumentation). No client-side or A2A protocol spans are
  produced.
- The A2A `contextId` is mapped to the standard
  `gen_ai.conversation.id` attribute (via
  `InvokeAgentInvocation.conversation_id`), aligned with the A2A
  semantic-conventions draft
  ([semantic-conventions-genai#195](https://github.com/open-telemetry/semantic-conventions-genai/pull/195)).
  Task id/state remain attached via the draft's `a2a.task.id` /
  `a2a.task.state` keys, set on the handler-owned span.
- Content capture (span, event and the `EVENT_ONLY`
  details-event-only mode) and `NO_CONTENT` default are governed
  entirely by the shared GenAI util via
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, so user
  prompts are never exported without explicit opt-in.

### Changed

- Every telemetry step — invocation construction,
  handler start/stop/fail, attribute and error recording — is fully
  fail-safe: a telemetry error never blocks the executor, never alters
  its result, and never replaces the business exception (the original
  exception is re-raised unchanged).
