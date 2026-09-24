# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial SWE-agent instrumentation: `SweagentInstrumentor` emits GenAI spans
  (entry, `invoke_agent`, react step, `execute_tool` with LLM `tool_calls` when
  available) via `ExtendedTelemetryHandler`; includes tests, examples, and
  tox/CI wiring.
  ([#165](https://github.com/alibaba/loongsuite-python-agent/pull/165))
