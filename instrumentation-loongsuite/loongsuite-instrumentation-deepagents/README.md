# LoongSuite DeepAgents Instrumentation

LoongSuite instrumentation for [DeepAgents](https://github.com/langchain-ai/deepagents).

## Installation

```bash
pip install loongsuite-instrumentation-deepagents
```

## Usage

```python
from opentelemetry.instrumentation.deepagents import DeepAgentsInstrumentor

DeepAgentsInstrumentor().instrument()

from deepagents import create_deep_agent
```

When instrumenting manually, call `DeepAgentsInstrumentor().instrument()` before
importing or binding `deepagents.create_deep_agent`. Auto-instrumentation runs
before application imports and does not need this extra ordering step.

## What it does

This instrumentation patches `deepagents.graph.create_deep_agent` so the final
graph returned by DeepAgents is marked with `_loongsuite_react_agent = True`.
It also marks the graph as a DeepAgents agent and injects the same flags into
call-time `RunnableConfig` metadata for `invoke`, `ainvoke`, `stream`, and
`astream`.

The marker is consumed by `loongsuite-instrumentation-langchain`, which routes
the DeepAgents root graph span as an `AGENT` span instead of a generic `CHAIN`
span.

The LangChain tracer also maps each DeepAgents `model` decision node to a
`react step` span. Multi-round tool flows end the previous step with
`gen_ai.react.finish_reason = "tool_calls"` and the final step with `"stop"`.
This package intentionally does not create separate ENTRY spans or GenAI
metrics.

For DeepAgents skills, the framework first loads skill metadata through
`SkillsMiddleware.before_agent` and exposes the available skills in the system
prompt. Loading the full skill instructions happens when the agent calls the
built-in filesystem tool to read the skill file, for example
`read_file(file_path="/skills/foo/SKILL.md")`. That `execute_tool read_file`
span is annotated with `gen_ai.skill.name`, `gen_ai.skill.id`,
`gen_ai.skill.description`, and `gen_ai.skill.version` when the file path
matches a registered top-level `SKILL.md`. Reading helper files under the same
skill directory is recorded as a normal file read, not as a skill load.

## Compatibility

- `deepagents >= 0.6.0, < 0.7.0`
- Python 3.11+
