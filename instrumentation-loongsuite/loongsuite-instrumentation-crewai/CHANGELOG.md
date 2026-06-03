# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.6.0 (2026-06-03)

### Breaking

- Align CrewAI GenAI span names with `opentelemetry-util-genai`
  extended semantic conventions. `gen_ai.operation.name` now reports
  `enter`, `invoke_agent`, or `execute_tool`; the CrewAI framework operation
  is reported in `gen_ai.crewai.operation` instead.
- Replace the legacy `gen_ai.system=crewai` attribute with
  `gen_ai.provider.name=crewai`.
- Use the current content-capture environment values:
  `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` and
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`.

### Changed

- Migrate CrewAI entry, task, agent, and tool spans to
  `opentelemetry-util-genai` `ExtendedTelemetryHandler`.
- Keep instrumentation post-processing failures from changing successful
  CrewAI calls into user-visible errors, avoid duplicate nested agent spans,
  and gate content-like CrewAI task and agent attributes behind util-genai
  content capture controls.

### Added

- Add a real CrewAI smoke example covering sync, streaming, and concurrent
  calls for local otel-gui and Robin/ARMS verification.

## Version 0.5.0 (2026-05-11)

There are no changelog entries for this release.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Changed

- Adapt imports to `opentelemetry-util-genai` module layout change
  ([#158](https://github.com/alibaba/loongsuite-python-agent/pull/158))
- Update README integration flow to align with the root recommended LoongSuite pattern using Option C (`pip install loongsuite-instrumentation-crewai`) and `loongsuite-instrument`.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

### Added

- Initialize the instrumentation for CrewAI
  ([#87](https://github.com/alibaba/loongsuite-python-agent/pull/87))
