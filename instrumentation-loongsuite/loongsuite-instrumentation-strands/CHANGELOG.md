# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Normalize public Strands reasoning text as GenAI reasoning parts on model
  spans and conversation history, excluding signatures and redacted content.
- Keep agent output limited to the final response while retaining reasoning
  and tool transcripts on the corresponding model and tool spans.
- Tolerate the framework-owned baggage context detach that runs when an
  application closes a strand stream from another context, so `strands-agents`
  1.60.0 and later no longer surface a spurious `Failed to detach context`
  error while the instrumentation's own context hygiene is unchanged.

## Version 0.9.0 (2026-09-07)

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
