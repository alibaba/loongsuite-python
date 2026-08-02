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
