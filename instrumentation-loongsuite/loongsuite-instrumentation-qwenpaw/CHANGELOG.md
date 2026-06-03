# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.

## Version 0.5.0 (2026-05-11)

### Added

- Renamed the primary instrumentation package to QwenPaw and kept runtime
  compatibility for `copaw <= 1.0.2`.
- Added latest and legacy runtime test dependency sets.

### Changed

- Kept QwenPaw as the single auto-instrumentation entry point while retaining
  legacy import/runtime compatibility.

## Version 0.4.0 (2026-04-03)

### Added

- **CoPaw instrumentation initialization**: ``CoPawInstrumentor`` registers
  automatic instrumentation for CoPaw when ``instrument()`` is called (included
  in LoongSuite distro automatic injection).
  ([#162](https://github.com/alibaba/loongsuite-python-agent/pull/162))

### Changed

- Instrumentor depends on ``opentelemetry-util-genai`` and passes
  ``tracer_provider``, ``meter_provider``, and ``logger_provider`` from
  ``instrument()`` into the shared GenAI telemetry handler.
  ([#162](https://github.com/alibaba/loongsuite-python-agent/pull/162))
