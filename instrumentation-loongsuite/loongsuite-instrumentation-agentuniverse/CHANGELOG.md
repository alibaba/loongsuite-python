# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

### Added

- Independent Agent, LLM and Tool instrumentation using the framework's six
  wrapper extension points, with shared GenAI attributes and 26 native metric
  families. Existing framework wrappers are restored on uninstrumentation.
- Stable agentUniverse 0.0.19.1 API compatibility for sync/async calls and the
  native LangChain ReAct executor. Optional alpha round and dispatcher hooks
  are not required to enable the three core layers.
- Representative stable runtime coverage: two user turns per session, four
  model decisions and three actual tools per turn, concurrent full-stream
  conversations, public reasoning enabled/disabled and usage-only SSE tails.

### Fixed

- Install and restore wrapper/provider hooks transactionally, retaining later
  hook owners and allowing a failed installation to be retried.
- Read response IDs, public reasoning, finish reasons and tool-call fragments
  from the stable SDK's retained OpenAI payloads without changing business
  objects. Read JSON arguments from native text ReAct tool dispatch.
- Use provider-inclusive Token totals without counting cache/reasoning subsets
  twice, and aggregate each stream onto its parent exactly once.
- Keep Agent telemetry limited to its actual final answer. Preserve explicit
  session IDs, message roles/history, channel provider/model and temperature zero.
- Preserve business exceptions and cancellation, chunk identity/order and
  context restoration across tasks, unread close and collector failures.
- Keep the name-preserving proxy compatible with the OpenTelemetry Span API,
  preserving trace/span correlation for GenAI events.
- Apply content controls to both native and GenAI content attributes.

### Release blocker

- The published agentUniverse 0.0.19.1 wheel pins `grpcio==1.63.0`, conflicting
  with the aligned OTel exporter's `grpcio>=1.63.2` requirement. Runtime tests
  with unchanged stable source and a visibly versioned metadata-only dependency
  correction are diagnostic evidence. Normal installation and release remain
  blocked until upstream supplies a compatible stable package. No dependency
  bypass is advertised as supported installation.
