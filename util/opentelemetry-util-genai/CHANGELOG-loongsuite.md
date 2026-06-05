# Changelog

All notable changes to loongsuite project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.6.0 (2026-06-03)

There are no changelog entries for this release.

## Version 0.5.0 (2026-05-11)

### Added

- Add `gen_ai.skill.name`, `gen_ai.skill.id`, `gen_ai.skill.description`, and
  `gen_ai.skill.version` semantic attributes for `execute_tool` spans, plus the
  corresponding optional fields on `ExecuteToolInvocation`.

### Changed

- Relax OpenTelemetry package dependency ranges so LoongSuite GenAI utilities
  can be installed with recent AgentScope-based runtimes such as QwenPaw and
  CoPaw.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Breaking Change

- Remove package ``opentelemetry.util.genai._extended_common``. ``EntryInvocation`` and ``ReactStepInvocation`` now live in ``extended_types``; ``_apply_entry_finish_attributes`` and ``_apply_react_step_finish_attributes`` live in ``extended_span_utils``.
  ([#158](https://github.com/alibaba/loongsuite-python-agent/pull/158))
- Rename packages ``opentelemetry.util.genai._extended_memory`` → ``extended_memory`` and ``opentelemetry.util.genai._extended_semconv`` → ``extended_semconv`` (public module paths).
  ([#158](https://github.com/alibaba/loongsuite-python-agent/pull/158))

### Fixed

- Add bypass logic around instrumentation-specific initialization so `opentelemetry-util-genai` can work correctly as a standalone SDK without depending on instrumentation package bootstrap flow.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

## Version 0.2.0 (2026-03-12)

### Added

- Add `RetrievalDocument` dataclass for typed retrieval document representation (id, score, content, metadata). 
  ([#145](https://github.com/alibaba/loongsuite-python-agent/pull/145))
- Control RetrievalDocument serialization: when content capturing is NO_CONTENT, only serialize id and score; when SPAN_ONLY/SPAN_AND_EVENT, serialize full (id, score, content, metadata)
  ([#145](https://github.com/alibaba/loongsuite-python-agent/pull/145))
- Add Entry span (`gen_ai.span.kind=ENTRY`) and ReAct Step span (`gen_ai.span.kind=STEP`) support in `ExtendedTelemetryHandler` with types, utilities, and context-manager APIs
  ([#135](https://github.com/alibaba/loongsuite-python-agent/pull/135))
- Propagate `gen_ai.session.id` and `gen_ai.user.id` into Baggage during `start_entry`, enabling traffic coloring via `BaggageSpanProcessor` for all child spans within the entry block
  ([#135](https://github.com/alibaba/loongsuite-python-agent/pull/135))

### Changed

- **Retrieval semantic convention**: Align retrieval spans with LoongSuite spec
  ([#145](https://github.com/alibaba/loongsuite-python-agent/pull/145))
  - `gen_ai.operation.name`: `retrieve_documents` → `retrieval`
  - `gen_ai.retrieval.query` → `gen_ai.retrieval.query.text` for query text
  - Span name: `retrieval {gen_ai.data_source.id}` when `data_source_id` is set
  - Add `RetrievalInvocation` fields: `data_source_id`, `provider`, `request_model`, `top_k`
- Add optional `context` parameter to all `start_*` methods in `TelemetryHandler` and `ExtendedTelemetryHandler` for explicit parent-child span linking
  ([#135](https://github.com/alibaba/loongsuite-python-agent/pull/135))
- Unify `attach`/`detach` strategy in `ExtendedTelemetryHandler`: always `attach` regardless of whether `context` is provided; `stop_*`/`fail_*` guards restored to `context_token is None or span is None`
  ([#135](https://github.com/alibaba/loongsuite-python-agent/pull/135))

### Fixed

- Fix `gen_ai.retrieval.query` to respect content capturing mode: when `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is `NO_CONTENT`, both query and documents are now omitted from retrieve spans (previously only documents were gated)
  ([#139](https://github.com/alibaba/loongsuite-python-agent/pull/139))
- Fix `_safe_detach` to use `_RUNTIME_CONTEXT.detach` directly, avoiding noisy `ERROR` log from OTel SDK's `context_api.detach` wrapper
  ([#135](https://github.com/alibaba/loongsuite-python-agent/pull/135))
- Fix undefined `otel_context` reference in `_multimodal_processing.py` `process_multimodal_fail`, replaced with `_safe_detach`
  ([#135](https://github.com/alibaba/loongsuite-python-agent/pull/135))

## Version 0.1.0 (2026-02-28)

### Fixed

- Fix compatibility with Python 3.8 hashlib usage
  ([#102](https://github.com/alibaba/loongsuite-python-agent/pull/102))

### Added

- Add support for memory operations
  ([#83](https://github.com/alibaba/loongsuite-python-agent/pull/83))
- Add multimodal separation and upload support for GenAI utils
  ([#94](https://github.com/alibaba/loongsuite-python-agent/pull/94))
- Add `gen_ai.usage.total_tokens` attribute for LLM, Agent, and Embedding operations
  ([#108](https://github.com/alibaba/loongsuite-python-agent/pull/108))
- Add `gen_ai.response.time_to_first_token` attribute for LLM operations
  ([#113](https://github.com/alibaba/loongsuite-python-agent/pull/113))
- Enhance the capture and upload process of multimodal data
  ([#119](https://github.com/alibaba/loongsuite-python-agent/pull/119))
  - Enhance multimodal pre-upload pipeline with Data URI and local path support
  - Add AgentInvocation multimodal data handling
  - Introduce configurable pre-upload hooks and uploader entry points, add graceful shutdown processor for GenAI components
  - Improve multimodal metadata extraction and docs
