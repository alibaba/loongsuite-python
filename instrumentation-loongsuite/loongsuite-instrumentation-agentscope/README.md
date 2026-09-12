# LoongSuite AgentScope Instrumentation

LoongSuite instrumentation for [AgentScope](https://github.com/agentscope-ai/agentscope) framework.

## Features

- **Traces**: Distributed tracing for agents, LLMs, tools, and formatters
- **Metrics**: Performance metrics following OpenTelemetry GenAI semantic conventions
  - `gen_ai.client.operation.duration`: Operation duration in seconds
  - `gen_ai.client.token.usage`: Token usage for input and output
- **Events**: Detailed event logging for messages and choices

## Installation

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-agentscope

# Optional app dependency
pip install agentscope
```

## Usage

### Auto-instrumentation

```bash
loongsuite-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    python your_agentscope_app.py
```

### Manual instrumentation

```python
from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from agentscope.models import DashScopeChatModel
import agentscope

AgentScopeInstrumentor().instrument()

agentscope.init(project="my_project")
model = DashScopeChatModel(model_name="qwen-max")
result = await model(messages)

AgentScopeInstrumentor().uninstrument()
```

## Configuration

### Export to OTLP Backend

```bash
export OTEL_SERVICE_NAME=my-agentscope-app
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=<metrics_endpoint>

loongsuite-instrument python your_app.py
```

### Content Capture

Control message content capture using environment variables:

```bash
# Enable experimental GenAI semantic conventions
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

# Capture content in spans only
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

# Capture content in events only
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=EVENT_ONLY

# Capture in both spans and events
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

# Disable content capture (default)
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT
```

## Supported Components

- **Models**: ChatModelBase and all subclasses (nested `__call__` / proxy chains emit a single LLM span per logical invocation)
- **Agents**: AgentBase and all subclasses (same for stacked `AgentBase.__call__`)
- **ReAct Steps**: Per-iteration `react step` spans for ReActAgent (via instance hooks); nested overrides that call `super()._reasoning` / `super()._acting` still produce one step per logical iteration
- **Tools**: Toolkit.call_tool_function
- **Formatters**: TruncatedFormatterBase.format

### Skill metadata on tool spans

When a tool execution reads the top-level `SKILL.md` of a skill already
registered in `toolkit.skills`, AgentScope enriches that `execute_tool` span
with:

- `gen_ai.skill.name`
- `gen_ai.skill.id`
- `gen_ai.skill.description`
- `gen_ai.skill.version`

AgentScope v2 exposes skill loading through its built-in `Skill` viewer tool.
The v2 middleware records the requested `skill` argument as
`gen_ai.skill.name` and `gen_ai.skill.id`; when QwenPaw or AgentScope exposes
the registered skill directory, it also resolves the available description,
workspace-scoped id, and version metadata.

The matching is intentionally conservative:

- only registered skills can match
- only the top-level `SKILL.md` counts as a skill load
- reads under other paths such as `scripts/` or `references/` do not emit
  `gen_ai.skill.*`

For CoPaw-style workspace layouts, `gen_ai.skill.version` is resolved from
`SKILL.md` frontmatter first and falls back to workspace `skill.json`
`metadata.version_text` when needed.

## Concurrency (agents)

ReAct step tracing stores temporary state on the agent instance for the
duration of each `AgentBase.__call__`. **AgentScope itself models a single
in-flight reply per instance** (see `AgentBase._reply_task` / `_reply_id` in
agentscope): concurrent overlapping `await agent(...)` on the **same**
instance is unsupported and can corrupt traces, hook state, and interrupt
behavior. Use one instance per concurrent conversation, or serialize calls
with a lock or queue.

## Visualization

Export telemetry data to:
- [Alibaba Cloud Managed Service for OpenTelemetry](https://www.aliyun.com/product/xtrace)
- [AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio)
- Any OpenTelemetry-compatible backend (Jaeger, Zipkin, etc.)

## Examples

See the [main README](https://github.com/alibaba/loongsuite-python/blob/main/README.md) for complete usage examples.

## License

Apache License 2.0
