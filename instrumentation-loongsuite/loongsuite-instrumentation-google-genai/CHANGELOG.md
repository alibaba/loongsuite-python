# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Support ``google-genai`` 2.x while retaining compatibility with 1.x and
  Python 3.9.
- Add local opt-in real Gemini API tests and redacted VCR coverage for public
  CI without provider credentials.

## Version 0.8.0.dev

### Added

- Add LoongSuite Google GenAI instrumentation backed by
  `ExtendedTelemetryHandler`.
- Cover sync/async generation, streaming, and embeddings with extended GenAI
  attributes, metrics, multimodal messages, response identity, and TTFT.
