# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.6.0 (2026-06-03)

### Changed

- Route Google ADK `AGENT`, `LLM`, and `TOOL` spans through
  `opentelemetry-util-genai`, emitting current GenAI attributes such as
  `gen_ai.input.messages`, `gen_ai.output.messages`,
  `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`,
  `gen_ai.span.kind`, and `gen_ai.provider.name=google_adk`.
  ([#199](https://github.com/alibaba/loongsuite-python-agent/pull/199))

### Fixed

- Keep Google ADK streaming model spans open until the final response and
  protect same-session concurrent invocations from cross-finishing spans.
- Ensure Google ADK spans include LoongSuite `gen_ai.span.kind` values such as
  `AGENT`.

## Version 0.5.0 (2026-05-11)

There are no changelog entries for this release.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Changed

- Update README integration flow to align with the root recommended LoongSuite pattern using Option C (`pip install loongsuite-instrumentation-google-adk`) and `loongsuite-instrument`.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

## Version 0.2.0 (2026-03-12)

There are no changelog entries for this release.

## Version 0.1.0 (2026-02-28)

### Added

- Initialize the instrumentation for Google Agent Development Kit (ADK)
  ([#71](https://github.com/alibaba/loongsuite-python-agent/pull/71))
