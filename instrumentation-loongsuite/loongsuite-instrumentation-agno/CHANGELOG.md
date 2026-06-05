# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.6.0 (2026-06-03)

### Removed

- Drop Agno 1.x support and require Agno 2.x public `Agent.run`/`Agent.arun`
  APIs. Users that still depend on Agno 1.x should pin
  `loongsuite-instrumentation-agno < 0.6`.

### Changed

- Align message content capture with `opentelemetry-util-genai` controls such as
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`.
- Migrate Agno instrumentation to Agno 2.x public `Agent.run`/`Agent.arun`
  APIs and `opentelemetry-util-genai` `ExtendedTelemetryHandler`.
- Emit standardized AGENT, LLM, and TOOL GenAI spans for agent runs, model
  calls, streaming calls, async streaming calls, and function executions.

### Added

- Add a DashScope smoke example that exercises non-streaming, streaming, and
  concurrent Agno calls.
- Add local test requirements for the Agno LoongSuite tox environment.

## Version 0.5.0 (2026-05-11)

There are no changelog entries for this release.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Changed

- Update README integration flow to align with the root recommended LoongSuite pattern using Option A (`loongsuite-bootstrap -a install --latest`) for this package not yet on PyPI.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

## Version 0.2.0 (2026-03-12)

There are no changelog entries for this release.

## Version 0.1.0 (2026-02-28)

### Fixed

- Fix aresponse missing await and double wrapped() calls
  ([#107](https://github.com/alibaba/loongsuite-python-agent/pull/107))
- Fix broken trace caused by the improper setting of the parent context
  ([#23](https://github.com/alibaba/loongsuite-python-agent/pull/23))
- Correct span name of tool call
  ([#21](https://github.com/alibaba/loongsuite-python-agent/pull/21))

### Added

- Initial implementation of Agno instrumentation
  ([#13](https://github.com/alibaba/loongsuite-python-agent/pull/13))
