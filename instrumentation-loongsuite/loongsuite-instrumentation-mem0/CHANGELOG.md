# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.

## Version 0.5.0 (2026-05-11)

### Fixed

- Limit supported `mem0ai` versions to `>=1.0.0,<2.0.0`, test the true
  minimum supported version in `requirements.oldest.txt`, and keep the
  latest test environment on the supported `1.x` line.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Changed

- Adapt imports to `opentelemetry-util-genai` module layout change
  ([#158](https://github.com/alibaba/loongsuite-python-agent/pull/158))
- Update README integration flow to align with the root recommended LoongSuite pattern using Option C (`pip install loongsuite-instrumentation-mem0`) and `loongsuite-instrument`.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

## Version 0.2.0 (2026-03-12)

There are no changelog entries for this release.

## Version 0.1.0 (2026-02-28)

### Fixed

- Fix unit tests
  ([#98](https://github.com/alibaba/loongsuite-python-agent/pull/98))

### Added

- Refactor capture logic with memory handler
  ([#89](https://github.com/alibaba/loongsuite-python-agent/pull/89))
- Add hook extensions
  ([#95](https://github.com/alibaba/loongsuite-python-agent/pull/95))
- Initialize the instrumentation for mem0
  ([#67](https://github.com/alibaba/loongsuite-python-agent/pull/67))
