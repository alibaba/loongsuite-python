# LoongSuite deepagents Instrumentation

This package adds the deepagents-specific telemetry that is not already
covered by `loongsuite-instrumentation-langchain` and
`loongsuite-instrumentation-langgraph`.

It intentionally adds only three integration points:

- wraps `deepagents.graph.create_deep_agent` and the returned graph instance's
  `invoke`, `ainvoke`, `stream`, and `astream` methods to create the outer
  `ENTRY` span;
- injects a sidecar LangChain callback handler that enriches existing
  LoongSuite LangChain spans with deepagents framework and SubAgent metadata;
- installs a `SpanProcessor` that emits the `genai_calls_*` and
  `genai_llm_*` metrics from completed GenAI spans.

The `task` tool remains a `TOOL` span and is marked with
`gen_ai.tool.type=agent`. The SubAgent itself is represented by the nested
`AGENT` span emitted by the LangChain/LangGraph instrumentation.

## Local Install

Install the shared GenAI utility from the same source tree first, then install
the dependent LangChain, LangGraph, and deepagents instrumentations:

```bash
pip install -e ./util/opentelemetry-util-genai
pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-langchain
pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-langgraph
pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-deepagents
```
