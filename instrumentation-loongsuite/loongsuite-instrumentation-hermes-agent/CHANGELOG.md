# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.

## Version 0.5.0 (2026-05-11)

### Added

- Create `ENTRY` spans for Hermes `AIAgent` platform requests,
  then parent the agent invocation under the entry span.
- Add `gen_ai.skill.*` semantic attributes to Hermes skill tool spans when
  `skill_view` or `skill_manage` is executed.

### Fixed

- Fix nested-agent state corruption: `RunConversationWrapper` now uses a
  push/reset token pattern (`push_state` / `reset_state`) so each invocation
  gets an isolated `_HERMES_STATE` ContextVar frame. Parent agent state is
  restored when a child agent returns, preventing child runs from overwriting
  parent `last_response_id`, token counters, and step metadata.

## Version 0.5.0.dev (2026-04-24)

### Added

- Initial Hermes Agent instrumentation package
  - Agent, step, LLM, and tool spans built on `opentelemetry-util-genai`
  - Live and spec test coverage for Hermes agent conversations
  - Provider normalization for Hermes agent spans and underlying LLM spans
