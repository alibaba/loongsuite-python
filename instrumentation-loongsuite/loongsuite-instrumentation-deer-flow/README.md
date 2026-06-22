# LoongSuite Instrumentation for DeerFlow

OpenTelemetry instrumentation for [ByteDance DeerFlow](https://github.com/bytedance/deer-flow)
(`deer-flow >= 2.1, < 3.0`).

## Design

This package follows the hybrid plan documented in
`llm-dev/deer-flow/investigate/execute.md`:

* **LLM / Tool / ReAct Step spans are delegated** to
  `loongsuite-instrumentation-langchain` — DeerFlow builds on
  `langchain.agents.create_agent`, so LangChain's `BaseCallbackManager` wrap
  already emits those spans. This package **does not** wrap
  `BaseCallbackManager`, avoiding duplicate spans.
* **DeerFlow-specific Entry / Agent / Task / Sandbox / Memory spans** are
  produced by seven `wrapt` monkey patches on DeerFlow public APIs:

| # | Module / Symbol | Span kind | Operation name |
| --- | --- | --- | --- |
| 1 | `deerflow.runtime.runs.worker.run_agent` | ENTRY | `enter` |
| 2 | `deerflow.subagents.executor.SubagentExecutor._aexecute` | AGENT | `invoke_agent` |
| 3 | `deerflow.tools.builtins.task_tool.task_tool` | TASK | `run_task` |
| 4 | `deerflow.sandbox.sandbox.Sandbox.execute_command` (+ `read_file` / `write_file` / `glob` / `grep` / `list_dir`) | TOOL | `execute_tool` |
| 5 | `deerflow.sandbox.sandbox_provider.SandboxProvider.acquire` / `acquire_async` / `release` | TASK | `run_task` |
| 6 | `deerflow.agents.memory.storage.FileMemoryStorage.load` / `save` | TASK | `run_task` |
| 7 | `deerflow.agents.memory.updater.MemoryUpdater.aupdate_memory` | TASK | `run_task` |

Span creation for ENTRY / AGENT / TOOL uses `opentelemetry.util.genai.ExtendedTelemetryHandler`,
which also auto-emits the `genai_*` metrics via its built-in
`ExtendedInvocationMetricsRecorder`. TASK spans are created manually with
`tracer.start_as_current_span` because `ExtendedTelemetryHandler` does not
expose `start_task` / `stop_task` (TASK is a LoongSuite extension kind that
predates the handler).

## Configuration

| Env | Default | Effect |
| --- | --- | --- |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Standard gen-ai content capture switch. Controls `gen_ai.input.messages` / `gen_ai.output.messages` / `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result`. |
| `OTEL_DEER_FLOW_CAPTURE_MEMORY_CONTENT` | `false` | Second-level switch: when true (and the standard switch is also true), Memory spans will populate `input.value` / `output.value`. Memory content contains PII, hence the double gate. |

## Context propagation

DeerFlow propagates `contextvars` into subagent isolated event loops via
`contextvars.copy_context()` (executor.py:749/845). OTel Context follows
`ContextVar` automatically, so the ENTRY span's parent/child relationship to
the subagent AGENT span is preserved without any manual injection here.

## Usage

```python
from opentelemetry.instrumentation.deer_flow import DeerFlowInstrumentor

DeerFlowInstrumentor().instrument()
```

Or via the `opentelemetry-instrument` CLI entry point:

```
opentelemetry-instrument --instrumentation_modules deer_flow
```

## Limitations

* IM Channel entry points (Slack / Feishu / Discord) do not go through
  `run_agent`; they are not instrumented in v1.
* `task_tool` produces both a TASK span (this package, parent) and a TOOL
  span (langchain instrumentation, child). This is intentional — the TASK
  span describes "dispatch to subagent" semantics while the TOOL span covers
  the LangChain `@tool` invocation mechanics.
