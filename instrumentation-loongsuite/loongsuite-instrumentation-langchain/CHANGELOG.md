# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.5.0 (2026-05-11)

There are no changelog entries for this release.

## Version 0.4.0 (2026-04-03)

There are no changelog entries for this release.

## Version 0.3.0 (2026-03-27)

### Added

- Rerank / document-compressor span support
  ([#149](https://github.com/alibaba/loongsuite-python-agent/pull/149))

### Changed

- Adapt imports to `opentelemetry-util-genai` module layout change
  ([#158](https://github.com/alibaba/loongsuite-python-agent/pull/158))
- Set `run_inline = True` on the tracer so LangChain callbacks run inline for correct OpenTelemetry context propagation
  ([#148](https://github.com/alibaba/loongsuite-python-agent/pull/148))
- Improved token usage extraction to support multiple LangChain/LLM provider formats
  ([#148](https://github.com/alibaba/loongsuite-python-agent/pull/148))
- Update README integration flow to align with the root recommended LoongSuite pattern using Option C (`pip install loongsuite-instrumentation-langchain`) and `loongsuite-instrument`.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

## Version 0.2.0 (2026-03-12)

### Added

- ReAct Step instrumentation for AgentExecutor
  ([#139](https://github.com/alibaba/loongsuite-python-agent/pull/139))
  - Monkey-patch `AgentExecutor._iter_next_step` and `_aiter_next_step` to instrument each ReAct iteration
  - Dual patch: patch both `langchain.agents` (0.x) and `langchain_classic.agents` (1.x) when available, so either import path works
  - Covers invoke, ainvoke, stream, astream, batch, abatch
  - ReAct Step spans: `gen_ai.span.kind=STEP`, `gen_ai.operation.name=react`, `gen_ai.react.round`, `gen_ai.react.finish_reason`
  - Span hierarchy: Agent > ReAct Step > LLM/Tool

- LangGraph ReAct agent support (requires `loongsuite-instrumentation-langgraph`)
  ([#139](https://github.com/alibaba/loongsuite-python-agent/pull/139))
  - Detect LangGraph agents via `Run.metadata["_loongsuite_react_agent"]`
    (metadata injected by the LangGraph instrumentation)
  - Disambiguate the top-level graph (Agent span) from child nodes (chain
    spans) using `inside_langgraph_react` propagation
  - Agent name resolution: when the ReAct agent is invoked inside an outer
    graph node, inherit the node's name (e.g. `invoke_agent product_agent`)
    instead of the generic default (`invoke_agent LangGraph`)
  - Track ReAct step boundaries via callback-based detection of the
    `"agent"` node within the graph
  - Span hierarchy: Agent > ReAct Step > LLM/Tool (same as AgentExecutor)

### Breaking Changes

- Rewrite the instrumentation for LangChain with `genai-util`
  ([#139](https://github.com/alibaba/loongsuite-python-agent/pull/139))
  - Replaced the legacy `wrapt`-based function wrapping with `BaseTracer` callback mechanism
  - Migrated to `ExtendedTelemetryHandler` from `opentelemetry-util-genai` for standardized GenAI semantic conventions
  - Added Agent detection by `run.name`, TTFT tracking, content capture gating, and `RLock` thread safety
  - Added new test suite with `oldest`/`latest` dependency matrices

## Version 0.1.0 (2026-02-28)

### Added

- Initialize the instrumentation for langchain
  ([#34](https://github.com/alibaba/loongsuite-python-agent/pull/34))
