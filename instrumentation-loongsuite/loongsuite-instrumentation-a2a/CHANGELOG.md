# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- A2A protocol CLIENT/SERVER instrumentation aligned with the experimental
  OpenTelemetry A2A proposal #195, including method, message/task, conversation,
  Agent Card and endpoint attributes.
- HTTP trace-context propagation when transport instrumentation is absent, and
  enrichment of an existing HTTP SERVER span when it is present.
- Real SDK HTTP and streaming regression coverage, concurrent multi-turn
  isolation, cancellation and probe failure tests.

### Changed

- Replace the initial `AgentExecutor.execute` AGENT span with protocol spans.
  Higher-level Agent/LLM/Tool spans belong to framework instrumentation and
  continue to use the shared GenAI util.
- Suppress native SDK internal-method spans by default while instrumented;
  preserve explicit `OTEL_INSTRUMENTATION_A2A_SDK_ENABLED=true` debugging.
- Restore context before yielding streamed events and finalize owned spans at
  most once without changing business results or exception identity.
