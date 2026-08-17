# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Add DSPy instrumentation emitting the framework layer of a GenAI trace —
  ENTRY, CHAIN, AGENT, ReAct STEP, TOOL and RETRIEVER spans — through the
  shared LoongSuite GenAI telemetry utility.
- Attach every framework span to the OpenTelemetry context so the LLM spans
  produced by `loongsuite-instrumentation-litellm` deeper in the DSPy call
  stack join the same trace as children.
- Emit one STEP span per `dspy.ReAct` round (excluding the trailing `extract`
  call), spanning the round's reasoning call and the tool call it selected.
- Fill `gen_ai.request.model` on framework spans best-effort from
  `dspy.settings.lm.model`, normalized the same way the LiteLLM instrumentation
  normalizes it so the `modelName` dimension does not split.
- Add `OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO` to bound span volume during
  optimizer compilation; the decision covers a whole outermost call subtree.
