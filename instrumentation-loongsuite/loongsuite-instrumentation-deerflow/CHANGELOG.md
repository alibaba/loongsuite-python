# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Treat user cancellation, early stream closure, and DeerFlow interrupted runs
  as non-error entry completions while retaining their interrupted status.
- Clarify the initial sandbox, skill, and memory observability boundaries.
- Document verified local, daemon, Docker, and embedded integration paths,
  including DeerFlow's exact `uv sync` behavior.
- Keep DeerFlow trace-correlation helper failures from interrupting embedded
  client streams.

## Version 0.8.0 (2026-07-14)

### Added

- Initial DeerFlow 2.x graph identification and application `ENTRY` spans for
  Gateway and embedded client execution.
- Propagate DeerFlow session, user, agent, assistant, run, and request trace
  identities without replacing the OpenTelemetry trace id.
- Map Gateway and embedded stream success, error, timeout, interruption, and
  early-close lifecycles onto a single application entry span.
- Preserve call-time OpenTelemetry and DeerFlow correlation contexts while a
  synchronous embedded iterator is advanced or closed from another thread.
