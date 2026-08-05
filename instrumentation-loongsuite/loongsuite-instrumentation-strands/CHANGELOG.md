# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Add Strands Agents instrumentation for Agent, ReAct step, model, and tool
  spans using the shared LoongSuite GenAI telemetry utility.
- Preserve application parent context while suppressing duplicate Strands SDK
  native spans.

### Changed

- Report `strands-agents` as the framework provider on agent spans while
  keeping the model provider on LLM spans.
- Always emit framework-level LLM spans without suppressing nested provider
  instrumentation in the open-source package.
- Finalize remaining LLM, tool, step, and agent spans best-effort when one
  telemetry finalizer fails during invocation or stream cleanup.
- Isolate failures raised by an underlying stream's `aclose()` while still
  finalizing the active invocation and restoring application context.
- Preserve cancellation and generator-exit error types, retain the original
  model error when a failure reporter also fails, and mark model-span startup
  failures as an explicit ReAct-step telemetry error.
- Capture the first model call's input from the Strands agent message snapshot,
  before invocation state is populated by tool execution.
