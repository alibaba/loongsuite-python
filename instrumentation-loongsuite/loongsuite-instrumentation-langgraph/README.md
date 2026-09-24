# LongSuLoongSuiteite LangGraph Instrumentation

LoongSuite instrumentation for [LangGraph](https://github.com/langchain-ai/langgraph).

## Installation

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-langgraph
```

## Usage

```python
from opentelemetry.instrumentation.langgraph import LangGraphInstrumentor

LangGraphInstrumentor().instrument()
```

## What it does

This instrumentation patches two targets to enable LangChain instrumentation
(`loongsuite-instrumentation-langchain`) to recognise LangGraph ReAct agents
and create proper Agent / ReAct Step spans.

All patches use `wrapt.wrap_function_wrapper` (consistent with
`loongsuite-instrumentation-langchain`).

### 1. `create_react_agent` patch

Wraps `langgraph.prebuilt.create_react_agent` to set a boolean flag
`_loongsuite_react_agent = True` on the compiled `CompiledStateGraph`.
The flag itself is not accessible inside LangChain's callback system — it
only serves as the trigger for the second patch below.

### 2. `Pregel.stream` / `Pregel.astream` patch

Wraps the graph execution entry points so that when a graph carrying
`_loongsuite_react_agent = True` is invoked, the metadata
`{"_loongsuite_react_agent": True}` is injected into the `RunnableConfig`
**before** execution begins.

The data flow:

```
graph._loongsuite_react_agent = True       # set by patch 1
            │
            ▼
Pregel.stream() wrapper intercepts call    # patch 2
            │
            ▼
config["metadata"]["_loongsuite_react_agent"] = True
            │
            ▼
LangChain callback manager reads metadata
            │
            ▼
Run.metadata["_loongsuite_react_agent"]    # LoongsuiteTracer reads this
```

LangChain's callback system automatically propagates `config["metadata"]`
to all child callbacks, so every sub-node within the graph also carries the
flag. The `LoongsuiteTracer` disambiguates the top-level graph (Agent span)
from child nodes (chain spans) by tracking an internal
`inside_langgraph_react` flag that propagates through the run hierarchy.

`Pregel.invoke()` / `ainvoke()` internally delegate to `stream` / `astream`,
so only the latter two need to be patched.

## How it works with LangChain instrumentation

When both instrumentors are active, the `LoongsuiteTracer` in the LangChain
instrumentation:

1. **Detects the agent** — `_has_langgraph_react_metadata(run)` checks
   `Run.metadata` for the flag. If the parent is not already inside a
   LangGraph agent, this run becomes an Agent span.

2. **Resolves agent name** — when the ReAct agent is invoked inside an
   outer graph node (e.g. `product_agent`), the agent span inherits the
   node's name (`invoke_agent product_agent`) instead of the generic
   default (`invoke_agent LangGraph`).

3. **Tracks ReAct steps** — each time the `"agent"` node fires inside
   the graph, a new ReAct Step span is created, with the hierarchy:
   `Agent > ReAct Step > LLM / Tool`.

## Compatibility

- `langgraph >= 0.2`
- `langchain_core >= 0.1.0`
- Python 3.9+
