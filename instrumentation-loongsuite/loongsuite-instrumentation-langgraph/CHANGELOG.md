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

### Changed

- Update README integration flow to align with the root recommended LoongSuite pattern using Option C (`pip install loongsuite-instrumentation-langgraph`) and `loongsuite-instrument`.
  ([#159](https://github.com/alibaba/loongsuite-python-agent/pull/159))

## Version 0.2.0 (2026-03-12)

### Added

- Initial instrumentation framework for LangGraph
  ([#143](https://github.com/alibaba/loongsuite-python-agent/pull/143))
  - Patch `create_react_agent` to set `_loongsuite_react_agent = True` flag
    on `CompiledStateGraph`
  - Patch `Pregel.stream` / `Pregel.astream` to inject
    `metadata["_loongsuite_react_agent"]` into `RunnableConfig`, enabling
    LangChain instrumentation to detect ReAct agents via callback metadata
  - All patches use `wrapt.wrap_function_wrapper` /
    `opentelemetry.instrumentation.utils.unwrap` (consistent with
    `loongsuite-instrumentation-langchain`)
