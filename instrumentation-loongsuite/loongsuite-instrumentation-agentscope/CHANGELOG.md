# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.5.0 (2026-05-11)

### Added

- Detect skill-load tool executions by matching reads of registered skills'
  top-level `SKILL.md`, and enrich the corresponding `execute_tool` span with
  `gen_ai.skill.name`, `gen_ai.skill.id`, `gen_ai.skill.description`, and
  `gen_ai.skill.version`.

### Fixed

- Pin `wrapt` to `< 2.0.0` for AgentScope instrumentation compatibility with
  the current wrapper API usage.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Changed

- Adapt imports to `opentelemetry-util-genai` module layout change
  ([#158](https://github.com/alibaba/loongsuite-python-agent/pull/158))

### Fixed

- Avoid duplicate LLM / Agent spans when multiple `ChatModelBase` or
  `AgentBase` subclasses stack (e.g. proxy layers that each implement `__call__`
  and forward inward), by tracking per-task `__call__` depth with
  `contextvars` and only instrumenting the outermost frame
  ([#153](https://github.com/alibaba/loongsuite-python-agent/pull/153))
- Avoid duplicate `react step` spans when ReAct hook wrappers nest (e.g.
  subclasses or mixins that override `_reasoning` / `_acting` and call
  `super()`), by only opening steps and updating tool-act counts on the
  outermost wrapper
  ([#153](https://github.com/alibaba/loongsuite-python-agent/pull/153))

### Changed

- Update README integration flow to align with the root recommended LoongSuite pattern using Option C (`pip install loongsuite-instrumentation-agentscope`) and `loongsuite-instrument`.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

### Added

- Add ReAct step span instrumentation for ReAct agents
  ([#140](https://github.com/alibaba/loongsuite-python-agent/pull/140))
  - Each ReAct iteration is wrapped in a `react step` span with `gen_ai.react.round` and `gen_ai.react.finish_reason` attributes
  - Uses AgentScope's instance-level hook system for robust, non-invasive instrumentation

## Version 0.2.0 (2026-03-12)

There are no changelog entries for this release.

## Version 0.1.0 (2026-02-28)

### Fixed

- Fix tool call response parsing
  ([#118](https://github.com/alibaba/loongsuite-python-agent/pull/118))
- Fix LLM message content capture in spans
  ([#91](https://github.com/alibaba/loongsuite-python-agent/pull/91))
- Fix spell mistake in pyproject.toml
  ([#8](https://github.com/alibaba/loongsuite-python-agent/pull/8))

### Breaking Changes

- Deprecate the support for AgentScope v0
  ([#82](https://github.com/alibaba/loongsuite-python-agent/pull/82))

### Changed

- Refactor the instrumentation for AgentScope with `genai-util`
  ([#82](https://github.com/alibaba/loongsuite-python-agent/pull/82))
  - **Refactored to use opentelemetry-util-genai**: Migrated to `ExtendedTelemetryHandler` and `ExtendedInvocationMetricsRecorder` from `opentelemetry-util-genai` for unified metrics and tracing management
  - **Architecture Simplification**: Removed redundant code and consolidated instrumentation logic
  - **Tool Tracing Enhancement**: Rewritten tool execution tracing to use `ExtendedTelemetryHandler` for full feature support (see HANDLER_INTEGRATION.md)
    - Now properly leverages `_apply_execute_tool_finish_attributes` for standardized attribute handling
    - Automatic metrics recording for tool executions
    - Content capturing mode support (respects experimental mode and content capturing settings)
    - Unified error handling with proper error attributes
  - Removed "V1" prefix from class names (AgentScopeV1ChatModelWrapper → AgentScopeChatModelWrapper, etc.)
  - Updated to use Apache License 2.0 headers across all source files
- Refactor the instrumentation for AgentScope
  ([#14](https://github.com/alibaba/loongsuite-python-agent/pull/14))

### Added

- Add support for agentscope v1.0.0
  ([#45](https://github.com/alibaba/loongsuite-python-agent/pull/45))
- Initialize the instrumentation for AgentScope
  ([#2](https://github.com/alibaba/loongsuite-python-agent/pull/2))
