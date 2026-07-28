# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Align provider hooks and tests with the canonical
  ``opentelemetry-instrumentation-google-genai==1.0b1`` baseline.
- Support ``interactions.create`` and automatic function-call
  ``execute_tool`` spans in addition to generation, streaming, and embeddings.
- Retain LoongSuite ``ExtendedTelemetryHandler`` metrics, multimodal handling,
  standard instrumentation suppression, reasoning-part capture, Python 3.9,
  and completion-hook support through an isolated compatibility layer.
- Document that the provider runs on LoongSuite's shared GenAI util, while
  Robin adds ARMS metrics and private lower-level SDK suppression as a
  commercial overlay.
- Fix reusable-config mutation, stream-construction span leaks, Google async
  stream closing, streaming TTFT, real Interactions SSE completion parsing,
  and embedding raw-response state isolation found during the upstream delta
  audit.
- Add local opt-in real Gemini API tests and redacted VCR coverage for public
  CI without provider credentials.
- Apply the shared ``hook_advice`` fail-open contract to generation,
  Interactions, embeddings, automatic tool calls, and sync/async streams so
  probe failures cannot change SDK call count, results, chunks, cancellation,
  ``GeneratorExit``, or original provider exceptions.
- Detach streaming context before returning SDK streams, make finalization
  idempotent across close/aclose/error/GC paths, and cover cross-Context and
  one-fault-among-many isolation.

## Version 0.8.0.dev

### Added

- Add LoongSuite Google GenAI instrumentation backed by
  `ExtendedTelemetryHandler`.
- Cover sync/async generation, streaming, and embeddings with extended GenAI
  attributes, metrics, multimodal messages, response identity, and TTFT.
