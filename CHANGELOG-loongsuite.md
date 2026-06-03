# Changelog for LoongSuite

All notable changes to loongsuite components will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
> The following components are released independently and maintain individual CHANGELOG files.
> Use [this search for a list of all CHANGELOG-loongsuite.md files in this repo](https://github.com/search?q=repo%3Aalibaba%2Floongsuite-python-agent+path%3A**%2FCHANGELOG-loongsuite.md&type=code).

## Unreleased

## Version 0.6.0 (2026-06-03)

### Changed

- Publish LoongSuite GenAI utilities as `loongsuite-otel-util-genai` for new
  releases. New LoongSuite distro and instrumentation packages depend on
  `loongsuite-otel-util-genai`; the previous `loongsuite-util-genai`
  distribution remains available for existing installations but is no longer
  the package receiving new LoongSuite GenAI utility updates. The Python import
  namespace remains `opentelemetry.util.genai`.

## Version 0.5.0 (2026-05-11)

### Fixed

- Pin the default upstream core SHA used by CI to a 0.62-compatible commit, and
  cap `mem0ai` support to `<2.0.0` while keeping Mem0 oldest/latest test
  coverage aligned with the supported `1.x` range.
- Constrain LoongSuite instrumentation `wrapt` dependencies to `<2.0.0` to
  avoid wrapper compatibility regressions from the wrapt 2.x line.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Added

- Release tooling: build and publish **`instrumentation-loongsuite/*`** as separate PyPI wheels (with **`loongsuite_pypi_manifest.py`** defining which distributions are uploaded; some remain tar-only until ready), and add a **PyPI packages** section to aggregated release notes
  ([#155](https://github.com/alibaba/loongsuite-python-agent/pull/155))
- **`loongsuite-site-bootstrap`**: initialize .pth-based OTel auto-instrumentation package
  ([#156](https://github.com/alibaba/loongsuite-python-agent/pull/156))
- **Top-level docs**: add Chinese README (**`README-zh.md`**) translated from **`README.md`**.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/158))

### Changed

- **`instrumentation-loongsuite/*`**, **`loongsuite-distro`**, and **`util/opentelemetry-util-genai`**: `pyproject.toml` metadata and dependencies for standalone PyPI installs
  ([#155](https://github.com/alibaba/loongsuite-python-agent/pull/155))
- **`loongsuite-site-bootstrap`**, **`loongsuite-distro`** docs: update **`README.md`**.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/158))

## Version 0.2.0 (2026-03-12)

There are no changelog entries for this release.

## Version 0.1.0 (2026-02-28)

### Fixed

- Add `from __future__ import annotations` to fix Python 3.9 compatibility for union type syntax (`X | Y`)
  ([#80](https://github.com/alibaba/loongsuite-python-agent/pull/80))

### Added

- `loongsuite-distro`: initialize loongsuite python agent distro
  ([#126](https://github.com/alibaba/loongsuite-python-agent/pull/126))
