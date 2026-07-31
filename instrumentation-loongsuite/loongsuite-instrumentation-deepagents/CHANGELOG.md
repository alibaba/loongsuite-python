# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.8.0 (2026-07-31)

There are no changelog entries for this release.

## Version 0.7.0 (2026-07-03)

### Added

- Initial DeepAgents instrumentation that marks `create_deep_agent` graphs so
  LangChain instrumentation emits an `AGENT` span for the DeepAgents root.
- DeepAgents skill-load telemetry: when an agent reads a registered skill's
  top-level `SKILL.md` through the built-in `read_file` tool, the tool span
  carries `gen_ai.skill.name`, `gen_ai.skill.id`,
  `gen_ai.skill.description`, and `gen_ai.skill.version`.

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.
