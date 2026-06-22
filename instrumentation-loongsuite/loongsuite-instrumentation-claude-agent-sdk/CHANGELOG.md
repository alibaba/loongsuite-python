# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Capture `gen_ai.skill.name`, `gen_ai.skill.id`, `gen_ai.skill.description`
  and `gen_ai.skill.version` on the `execute_tool` span of the built-in
  `Skill` tool. Skill metadata is read best-effort from the project-level
  `SKILL.md` frontmatter (located via `SystemMessage.data.cwd`); `skill.id`
  is reported as `claude:project:<skill-name>`. Metadata read failures never
  affect the SDK call.

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.

## Version 0.5.0 (2026-05-11)

There are no changelog entries for this release.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Changed

- Adapt imports to `opentelemetry-util-genai` module layout change
  ([#158](https://github.com/alibaba/loongsuite-python/pull/158))

## Version 0.2.0 (2026-03-12)

There are no changelog entries for this release.

## Version 0.1.0 (2026-02-28)

### Added

- Initial implementation of Claude Agent SDK instrumentation
  ([#104](https://github.com/alibaba/loongsuite-python/pull/104))
  - Support for agent query sessions via Hooks mechanism
  - Support for tool execution tracing (PreToolUse/PostToolUse hooks)
  - Integration with `opentelemetry-util-genai` ExtendedTelemetryHandler
  - Span attributes following OpenTelemetry GenAI Semantic Conventions
  - Support for Alibaba Cloud DashScope Anthropic-compatible API
