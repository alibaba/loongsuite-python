# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

### Added

- Independent Agent, LLM and Tool instrumentation using the framework's six
  wrapper extension points, with shared GenAI attributes and 26 native metric
  families. Existing framework wrappers are restored on uninstrumentation.
- Instrumentation for the published agentUniverse 0.1.0a2 preview on Python
  3.10–3.12. The installer, bootstrap registry and test matrix target the same
  framework version.

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
