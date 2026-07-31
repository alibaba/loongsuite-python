# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.8.0 (2026-07-31)

There are no changelog entries for this release.

## Version 0.7.0 (2026-07-03)

### Fixed

- Record token usage from Qwen-Agent DashScope response metadata on streaming
  and non-streaming chat spans.
- Roll up child LLM token usage to Qwen-Agent invoke-agent spans, preserve
  nested agent spans, and record only the final agent answer as output.

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.

## Version 0.5.0 (2026-05-11)

There are no changelog entries for this release.

## Version 0.4.0.dev

### Added

- Initial implementation of Qwen-Agent instrumentation
  ([#154](https://github.com/alibaba/loongsuite-python/pull/154))
