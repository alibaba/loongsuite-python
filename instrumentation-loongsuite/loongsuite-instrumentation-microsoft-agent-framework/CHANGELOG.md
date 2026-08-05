# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Enable native telemetry on Microsoft Agent Framework 1.0, whose
  ``enable_instrumentation`` API does not yet accept ``force``.

### Tests

- Add replay-only VCR contracts against the real framework for content modes,
  streaming, tools, provider errors, and concurrent trace isolation.

## Version 0.8.0 (2026-07-31)

### Fixed

- Avoid enriching non-MAF spans that use overlapping GenAI operation names.
- Preserve parent-child context for legacy Microsoft Agent Framework streaming
  agent spans.
- Stop exporting process-cumulative GenAI ObservableGauges and use Microsoft
  Agent Framework's native metric instruments instead.
- Keep non-streaming MAF spans current for nested work and scope Robin provider
  suppression to the active LLM call when that optional integration is present.

## Version 0.7.0 (2026-07-03)

### Added

- Add Microsoft Agent Framework instrumentation aligned with LoongSuite GenAI semantic conventions.
  ([#229](https://github.com/alibaba/loongsuite-python/pull/229))

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.
